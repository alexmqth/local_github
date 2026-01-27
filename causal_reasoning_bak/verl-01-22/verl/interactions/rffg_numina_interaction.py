# file: verl/interactions/numina_interaction_local.py
"""
使用本地模型进行最终评估的 Interaction 模块。

特点：
1. 中间步骤不调用模型，返回 0
2. 最终步骤调用一次本地模型，评估所有步骤
3. 使用简单的 0/1 数列格式，适合指令跟随能力较弱的模型
4. Robust 的解析方法，容忍格式偏差
"""

from typing import Dict, Any, List, Tuple, Optional
from uuid import uuid4
import re
import json
import os
import logging
import torch
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed")

from verl.interactions.base import BaseInteraction


# ============== 正则表达式 ==============
READY_RE = re.compile(r"<READY>", re.IGNORECASE)


def _flatten_messages(messages: List[Dict[str, Any]]) -> str:
    """把多轮 messages 展平为一个长文本。"""
    parts: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        text = str(m.get("content", ""))
        parts.append(f"[{role.upper()}] {text}")
    return "\n".join(parts)


# ============== 评估 Prompt 模板 ==============
FINAL_EVALUATION_PROMPT = """You are a mathematical proof checker.

## Problem
{problem_text}

## Reference Solution
{ground_truth}

## Proof Steps to Evaluate
{numbered_steps}

## Task
For EACH step above, give a score of 0 or 1:
- 1 = step is correct and makes real progress
- 0 = step has errors, is empty, or uses circular reasoning

Also give a FINAL score (0 or 1) for the whole proof:
- 1 = proof is complete and correct
- 0 = proof is incomplete or has errors

## Output Format (IMPORTANT)
Return ONLY numbers in this format:
STEPS: [score1, score2, score3, ...]
FINAL: score

Example for 3 steps:
STEPS: [1, 1, 0]
FINAL: 0

Now evaluate the {num_steps} steps above:"""


@dataclass
class EvaluationResult:
    """评估结果"""
    step_scores: List[int]      # 每一步的分数 (0 或 1)
    final_score: int            # 最终分数 (0 或 1)
    num_steps: int              # 总步数
    error_count: int            # 错误步数
    raw_response: str           # 原始响应


class RFFGInteractionLocal(BaseInteraction):
    """
    使用本地模型进行最终评估的 Interaction 模块。
    
    特点：
    1. 中间步骤不调用模型，返回 0
    2. 最终步骤调用一次本地模型
    3. 使用简单的数列格式
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._instances: Dict[str, Dict[str, Any]] = {}
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not available")

        # ---- 模型配置 ----
        self.model_name = config.get("verifier_model_name_or_path", "Qwen/Qwen2.5-Math-7B-Instruct")
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = int(config.get("max_new_tokens", 256))
        
        # ---- 加载模型 ----
        self._load_model()
        
        # ---- 奖励权重 ----
        self.step_weight = float(config.get("step_weight", 0.5))  # 步骤分权重
        self.final_weight = float(config.get("final_weight", 0.5))  # 最终分权重

        # ---- 日志配置 ----
        self.log_path = config.get("log_path", "rffg_local_log.jsonl")
        abs_log_dir = os.path.dirname(os.path.abspath(self.log_path))
        if abs_log_dir:
            os.makedirs(abs_log_dir, exist_ok=True)
        logger.info(f"[RFFGInteractionLocal] Logging to: {self.log_path}")

    def _load_model(self):
        """加载本地验证模型"""
        print(f"[RFFGInteractionLocal] Loading model: {self.model_name}")
        print(f"[RFFGInteractionLocal] Device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"[RFFGInteractionLocal] Model loaded successfully")

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        problem_id = kwargs.get("problem_id")
        problem_text = kwargs.get("problem_text")
        ground_truth = kwargs.get("ground_truth")
        goals = kwargs.get("goals", {}) or {}
        final_goal_id = kwargs.get("final_goal_id")
        goal_order = kwargs.get("goal_order", []) or []

        max_turns = int(kwargs.get("max_turns", self.config.get("max_turns", 8)))
        stop_regex = kwargs.get("stop_regex", self.config.get("stop_regex", r"<READY>"))
        stop_re = re.compile(stop_regex, re.IGNORECASE)

        self._instances[instance_id] = {
            "problem_id": problem_id,
            "problem_text": problem_text,
            "ground_truth": ground_truth,
            "goals": goals,
            "final_goal_id": final_goal_id,
            "goal_order": goal_order,
            "max_turns": max_turns,
            "stop_re": stop_re,
            "turn": 0,
            "last_assistant_text": "",
            "accumulated_steps": [],  # 累积的步骤
        }
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> Tuple[bool, str, float, Dict[str, Any]]:
        state = self._instances[instance_id]
        state["turn"] += 1
        current_turn = state["turn"]

        # 抽取最后一条 assistant 的内容
        last_assistant = ""
        for item in reversed(messages):
            if item.get("role") == "assistant":
                last_assistant = item.get("content", "")
                break
        state["last_assistant_text"] = last_assistant
        state["accumulated_steps"].append(last_assistant)

        # 获取当前 turn 的 goal 信息
        goal_order = state.get("goal_order", [])
        final_goal_id = state.get("final_goal_id")
        turn_index = current_turn - 1
        
        extra_info = {}
        if turn_index < len(goal_order):
            current_goal_info = goal_order[turn_index]
            current_to_goal_id = current_goal_info.get("to")
            extra_info["is_final_step"] = (current_to_goal_id == final_goal_id)
        else:
            extra_info["is_final_step"] = True

        # 计算奖励
        reward = await self.calculate_score(instance_id, messages=messages, extra_info=extra_info)

        # 终止条件
        should_stop_by_pattern = bool(state["stop_re"].search(last_assistant or ""))
        should_stop_by_turn = state["turn"] >= state["max_turns"]
        should_terminate = should_stop_by_pattern or should_stop_by_turn

        response_to_user = "Interaction finished." if should_terminate else ""

        metadata = {
            "turn": state["turn"],
            "ready_tag_detected": should_stop_by_pattern,
            "max_turns_exceeded": should_stop_by_turn,
        }
        return should_terminate, response_to_user, float(reward), metadata

    async def calculate_score(
        self,
        instance_id: str,
        messages: List[Dict[str, Any]],
        extra_info: Dict[str, Any],
        **kwargs
    ) -> float:
        """
        计算奖励分数。
        
        中间步骤：返回 0
        最终步骤：调用模型评估所有步骤，计算奖励
        """
        state = self._instances[instance_id]
        is_final_step = bool(extra_info.get("is_final_step", False))
        problem_id = state["problem_id"]
        
        # ============ 中间步骤：不调用模型，返回 0 ============
        if not is_final_step:
            self._log_simple(problem_id, state["turn"], is_final_step, 0.0, "intermediate step")
            return 0.0
        
        # ============ 最终步骤：调用模型评估 ============
        problem_text = state["problem_text"] or ""
        ground_truth = state["ground_truth"] or "Not provided"
        all_steps = state["accumulated_steps"]
        
        # 构建带编号的步骤
        numbered_steps = []
        for i, step in enumerate(all_steps):
            numbered_steps.append(f"Step {i+1}: {step}")
        numbered_steps_text = "\n\n".join(numbered_steps)
        
        # 调用模型评估
        eval_result = self._evaluate_with_model(
            problem_text=problem_text,
            ground_truth=ground_truth,
            numbered_steps=numbered_steps_text,
            num_steps=len(all_steps),
        )
        
        # 计算最终奖励
        # step_reward = 1.0 - (错误步数 / 总步数)
        if eval_result.num_steps > 0:
            step_reward = 1.0 - (eval_result.error_count / eval_result.num_steps)
        else:
            step_reward = 0.0
        
        final_reward = float(eval_result.final_score)
        
        # 加权组合
        total_reward = self.step_weight * step_reward + self.final_weight * final_reward
        
        # 限幅到 [0, 1]
        total_reward = max(0.0, min(1.0, total_reward))
        
        # 日志
        self._log_simple(
            problem_id, state["turn"], is_final_step, total_reward,
            f"steps={eval_result.step_scores}, final={eval_result.final_score}, "
            f"errors={eval_result.error_count}/{eval_result.num_steps}"
        )
        
        print(f"[Reward] Steps: {eval_result.step_scores}, Final: {eval_result.final_score}")
        print(f"[Reward] step_reward={step_reward:.3f}, final_reward={final_reward:.3f}, total={total_reward:.3f}")
        
        return total_reward

    def _evaluate_with_model(
        self,
        problem_text: str,
        ground_truth: str,
        numbered_steps: str,
        num_steps: int,
    ) -> EvaluationResult:
        """调用本地模型进行评估"""
        
        prompt = FINAL_EVALUATION_PROMPT.format(
            problem_text=problem_text,
            ground_truth=ground_truth,
            numbered_steps=numbered_steps,
            num_steps=num_steps,
        )
        
        # 生成
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 解析结果
        step_scores, final_score = self._extract_scores(response, num_steps)
        
        error_count = sum(1 for s in step_scores if s == 0)
        
        return EvaluationResult(
            step_scores=step_scores,
            final_score=final_score,
            num_steps=num_steps,
            error_count=error_count,
            raw_response=response,
        )

    def _extract_scores(self, response: str, expected_num_steps: int) -> Tuple[List[int], int]:
        """
        Robust 地从模型响应中提取分数。
        
        尝试多种格式：
        1. STEPS: [1, 0, 1] FINAL: 1
        2. [1, 0, 1] 1
        3. 纯数字序列
        4. 逐行数字
        
        Returns:
            (step_scores, final_score)
        """
        response = response.strip()
        
        # 尝试解析 STEPS: [...] FINAL: X 格式
        step_scores, final_score = self._try_parse_labeled_format(response)
        if step_scores is not None:
            return self._validate_and_fix(step_scores, final_score, expected_num_steps)
        
        # 尝试解析 [...] 数组格式
        step_scores, final_score = self._try_parse_array_format(response)
        if step_scores is not None:
            return self._validate_and_fix(step_scores, final_score, expected_num_steps)
        
        # 尝试解析纯数字格式
        step_scores, final_score = self._try_parse_numbers_format(response)
        if step_scores is not None:
            return self._validate_and_fix(step_scores, final_score, expected_num_steps)
        
        # 尝试逐行解析
        step_scores, final_score = self._try_parse_line_by_line(response, expected_num_steps)
        if step_scores is not None:
            return self._validate_and_fix(step_scores, final_score, expected_num_steps)
        
        # 全部失败，返回默认值（保守策略：全错）
        print(f"[Extract] Failed to parse, returning defaults. Response: {response[:200]}")
        return [0] * expected_num_steps, 0

    def _try_parse_labeled_format(self, response: str) -> Tuple[Optional[List[int]], Optional[int]]:
        """解析 STEPS: [...] FINAL: X 格式"""
        # 匹配 STEPS: [...]
        steps_match = re.search(r'STEPS\s*[:：]\s*\[([^\]]+)\]', response, re.IGNORECASE)
        # 匹配 FINAL: X
        final_match = re.search(r'FINAL\s*[:：]\s*(\d)', response, re.IGNORECASE)
        
        if steps_match:
            try:
                # 解析数组内容
                array_content = steps_match.group(1)
                # 提取所有数字
                numbers = re.findall(r'\d', array_content)
                step_scores = [int(n) for n in numbers]
                step_scores = [min(1, max(0, s)) for s in step_scores]  # 限制到 0/1
                
                final_score = 0
                if final_match:
                    final_score = min(1, max(0, int(final_match.group(1))))
                
                return step_scores, final_score
            except:
                pass
        
        return None, None

    def _try_parse_array_format(self, response: str) -> Tuple[Optional[List[int]], Optional[int]]:
        """解析 [1, 0, 1] 格式"""
        # 查找所有数组
        array_matches = re.findall(r'\[([^\]]+)\]', response)
        
        for match in array_matches:
            numbers = re.findall(r'\d', match)
            if len(numbers) >= 1:
                step_scores = [min(1, max(0, int(n))) for n in numbers]
                
                # 找最后一个独立数字作为 final
                remaining = response[response.rfind(']')+1:] if ']' in response else ""
                final_numbers = re.findall(r'\b(\d)\b', remaining)
                final_score = min(1, max(0, int(final_numbers[-1]))) if final_numbers else 0
                
                return step_scores, final_score
        
        return None, None

    def _try_parse_numbers_format(self, response: str) -> Tuple[Optional[List[int]], Optional[int]]:
        """解析纯数字序列，如 "1 0 1 0 1" """
        # 提取所有 0 和 1
        numbers = re.findall(r'\b([01])\b', response)
        
        if len(numbers) >= 2:
            # 最后一个是 final，其余是 steps
            step_scores = [int(n) for n in numbers[:-1]]
            final_score = int(numbers[-1])
            return step_scores, final_score
        
        return None, None

    def _try_parse_line_by_line(self, response: str, expected_num_steps: int) -> Tuple[Optional[List[int]], Optional[int]]:
        """逐行解析，查找每行的数字"""
        lines = response.strip().split('\n')
        scores = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 查找行中的 0 或 1
            # 优先查找明确的格式如 "Step 1: 1" 或 "score: 0"
            match = re.search(r'[:：=]\s*([01])\b', line)
            if match:
                scores.append(int(match.group(1)))
                continue
            
            # 查找行末的 0 或 1
            match = re.search(r'\b([01])\s*$', line)
            if match:
                scores.append(int(match.group(1)))
                continue
            
            # 查找任意 0 或 1
            match = re.search(r'\b([01])\b', line)
            if match:
                scores.append(int(match.group(1)))
        
        if len(scores) >= 2:
            # 假设最后一个是 final
            return scores[:-1], scores[-1]
        
        return None, None

    def _validate_and_fix(
        self, 
        step_scores: List[int], 
        final_score: int, 
        expected_num_steps: int
    ) -> Tuple[List[int], int]:
        """验证并修正分数列表长度"""
        # 如果步骤数不匹配，尝试修正
        if len(step_scores) < expected_num_steps:
            # 填充 0（保守策略）
            step_scores = step_scores + [0] * (expected_num_steps - len(step_scores))
        elif len(step_scores) > expected_num_steps:
            # 截断
            step_scores = step_scores[:expected_num_steps]
        
        # 确保 final_score 是 0 或 1
        final_score = 1 if final_score > 0 else 0
        
        return step_scores, final_score

    def _log_simple(
        self,
        problem_id: Optional[str],
        turn: int,
        is_final_step: bool,
        reward: float,
        note: str,
    ) -> None:
        """简化的日志记录"""
        rec = {
            "problem_id": problem_id,
            "turn": turn,
            "is_final_step": is_final_step,
            "reward": reward,
            "note": note,
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Log write failed: {e}")

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)

