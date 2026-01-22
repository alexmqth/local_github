"""
Reframe/Split subproblem "meta-tool" agent loop.

Motivation
----------
Compared to `step_verify_agent_loop.py`, this loop adds an *abstract tool* behavior:
when the assistant receives a low external score (e.g., LLM judge returns 0),
the assistant can request an external LLM to:
  1) rewrite the current subproblem more clearly, or
  2) split the current subproblem into finer subproblems.

This is implemented as two *pseudo tools* (no real environment tool):
  - `reframe_subproblem`
  - `split_subproblem`

The policy requests the action using Hermes tool-call markup:
  <tool_call>{"name":"reframe_subproblem","arguments":{...}}</tool_call>
  <tool_call>{"name":"split_subproblem","arguments":{...}}</tool_call>

Then this loop calls an external OpenAI-compatible endpoint (or local/policy backend)
to produce the rewritten/split subproblem(s), and injects them back as the next user turn(s).

Token-level credit assignment
-----------------------------
This loop emits:
  - extra_fields["turn_scores"]: list[float]              # step rewards for assistant generations
  - extra_fields["step_token_spans"]: list[(int,int)]     # spans in response token space
  - extra_fields["final_reward"]: float                   # optional final reward

`AgentLoopWorkerBase._postprocess()` will convert them into per-token rm_scores.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


RuleVerifierFn = Callable[[str, dict[str, Any]], Optional[float]]


def _import_by_path(dotted_path: str) -> Any:
    if ":" in dotted_path:
        module_name, attr = dotted_path.split(":", 1)
    else:
        module_name, attr = dotted_path.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, attr)


def _extract_last_assistant_text(messages: list[dict[str, Any]]) -> str:
    for item in reversed(messages):
        if item.get("role") == "assistant":
            return str(item.get("content", ""))
    return ""


def _parse_binary_judge(text: str) -> Optional[float]:
    """Parse judge output into 0/1 (robust to surrounding text)."""
    if not isinstance(text, str):
        return None
    t = text.strip()
    if not t:
        return None
    import re

    m = re.search(r"(?<!\d)([01])(?!\d)", t)
    if not m:
        return None
    return 1.0 if m.group(1) == "1" else 0.0


def _extract_json_obj(text: str) -> Optional[dict[str, Any]]:
    """Best-effort extraction of a JSON object from text."""
    if not isinstance(text, str):
        return None
    t = text.strip()
    if not t:
        return None
    # Fast path
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # Try to find the first {...} block
    import re

    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


@dataclass
class _VerifierConfig:
    # Optional rule verifier
    rule_verifier_fn: Optional[str] = None

    # Step verifier (LLM judge)
    llm_enable: bool = True
    llm_max_new_tokens: int = 4
    # Sampling params for step LLM judge (default deterministic)
    llm_temperature: float = 0.0
    llm_top_p: float = 1.0
    llm_prompt_template: str = (
        "You are a strict verifier. Judge whether the assistant answer correctly solves the given subproblem.\n"
        "Return ONLY one token: 1 (correct) or 0 (incorrect).\n\n"
        "Subproblem:\n{question}\n\n"
        "Assistant answer:\n{answer}\n"
    )

    # Judge backend
    judge_backend: str = "policy"  # policy | local | remote
    judge_model_name_or_path: Optional[str] = None
    judge_device: str = "cpu"  # cpu | cuda
    judge_dtype: str = "auto"
    judge_base_url: Optional[str] = None
    judge_api_key: Optional[str] = None
    judge_timeout_s: float = 120.0
    judge_max_concurrency: int = 8

    # Whether to emit process(step) rewards from the step verifier.
    # If False, the judge score is still computed and can be used to trigger reframe/split,
    # but `turn_scores` (and thus token-level rm_scores) will not include step rewards.
    step_reward_enable: bool = True

    # Final reward (optional; answer-extraction not implemented here, keep simple)
    final_enable: bool = False
    final_weight: float = 1.0


@dataclass
class _ReframeToolConfig:
    enable: bool = True

    # Trigger when step_score <= threshold (e.g., 0.0)
    trigger_score_threshold: float = 0.0

    # How many meta-tool uses are allowed per sample
    max_tool_uses: int = 2

    # If True, require the policy to explicitly emit a tool_call.
    # If False, auto-select action based on `auto_action`.
    require_model_request: bool = True

    # Auto selection when require_model_request=False or when model emits no tool_call
    auto_action: str = "reframe"  # reframe | split

    # Tool call parsing
    tool_call_format: str = "hermes"  # hermes (recommended)

    # External LLM backend for reframe/split
    # If not set, defaults to verifier's judge backend/model settings.
    tool_backend: Optional[str] = None  # policy | local | remote
    tool_model_name_or_path: Optional[str] = None
    tool_device: Optional[str] = None
    tool_dtype: Optional[str] = None
    tool_base_url: Optional[str] = None
    tool_api_key: Optional[str] = None
    tool_timeout_s: Optional[float] = None
    tool_max_concurrency: Optional[int] = None

    tool_max_new_tokens: int = 256
    tool_temperature: float = 0.2
    tool_top_p: float = 0.95

    # Policy-side instruction to request the meta-tool
    request_prompt: str = (
        "你刚刚的回答被外部评估为低分/0分。\n"
        "你可以调用一个“元工具”来让外部LLM把当前子问题表达得更清楚，或拆分成更细的子问题。\n"
        "请只输出一个工具调用（Hermes 格式），不要输出任何解释文字：\n"
        "- reframe_subproblem：重写当前子问题，使其更清晰、可执行。\n"
        "- split_subproblem：把当前子问题拆成多个更细的子问题（按顺序）。\n"
    )

    # External LLM prompt templates (must return JSON only)
    reframe_prompt_template: str = (
        "You rewrite subproblems to be clearer and more specific.\n"
        "Return ONLY valid JSON: {\"subproblem\": \"...\"}.\n\n"
        "Original subproblem:\n{question}\n\n"
        "Assistant answer (may be wrong):\n{answer}\n"
    )
    split_prompt_template: str = (
        "You split a subproblem into smaller, sequential subproblems.\n"
        "Return ONLY valid JSON: {\"subproblems\": [\"...\", \"...\", ...]}.\n\n"
        "Original subproblem:\n{question}\n\n"
        "Assistant answer (may be wrong):\n{answer}\n"
    )


@register("reframe_step_verify_agent")
class ReframeStepVerifyAgentLoop(AgentLoopBase):
    """
    A multi-turn loop:
    - solve current user subproblem
    - LLM judge each assistant turn => step score
    - if score is low, allow the policy to request a meta-tool:
        - reframe_subproblem
        - split_subproblem
      then call external LLM to produce improved subproblem(s) and continue.
    """

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        logger.warning("Performing class-level ReframeStepVerifyAgentLoop initialization (pid=%s)", os.getpid())

        cls.tokenizer = tokenizer
        cls.processor = processor

        rollout = config.actor_rollout_ref.rollout
        cls.prompt_length = rollout.prompt_length
        cls.response_length = rollout.response_length
        cls.max_user_turns = rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = rollout.multi_turn.max_assistant_turns
        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})

        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )

        # step_verify config comes via kwargs (agent_loop_config_path)
        step_cfg = kwargs.get("step_verify") or {}
        if step_cfg and not isinstance(step_cfg, dict):
            try:
                from omegaconf import OmegaConf

                step_cfg = OmegaConf.to_container(step_cfg, resolve=True)  # type: ignore[assignment]
            except Exception:
                step_cfg = {}

        cls.verifier_cfg = _VerifierConfig(
            rule_verifier_fn=step_cfg.get("rule_verifier_fn"),
            llm_enable=bool(step_cfg.get("llm_enable", True)),
            llm_max_new_tokens=int(step_cfg.get("llm_max_new_tokens", 4)),
            llm_temperature=float(step_cfg.get("llm_temperature", 0.0)),
            llm_top_p=float(step_cfg.get("llm_top_p", 1.0)),
            llm_prompt_template=str(step_cfg.get("llm_prompt_template", _VerifierConfig.llm_prompt_template)),
            judge_backend=str(step_cfg.get("judge_backend", "policy")),
            judge_model_name_or_path=step_cfg.get("judge_model_name_or_path"),
            judge_device=str(step_cfg.get("judge_device", "cpu")),
            judge_dtype=str(step_cfg.get("judge_dtype", "auto")),
            judge_base_url=step_cfg.get("judge_base_url"),
            judge_api_key=step_cfg.get("judge_api_key"),
            judge_timeout_s=float(step_cfg.get("judge_timeout_s", 120.0)),
            judge_max_concurrency=int(step_cfg.get("judge_max_concurrency", 8)),
            step_reward_enable=bool(step_cfg.get("step_reward_enable", True)),
            final_enable=bool(step_cfg.get("final_enable", False)),
            final_weight=float(step_cfg.get("final_weight", 1.0)),
        )

        cls.rule_verifier: Optional[RuleVerifierFn] = None
        if cls.verifier_cfg.rule_verifier_fn:
            fn = _import_by_path(cls.verifier_cfg.rule_verifier_fn)
            if not callable(fn):
                raise TypeError(f"rule_verifier_fn must be callable, got {type(fn).__name__}")
            cls.rule_verifier = fn  # type: ignore[assignment]

        # reframe_tool config comes via kwargs
        rt_cfg = kwargs.get("reframe_tool") or {}
        if rt_cfg and not isinstance(rt_cfg, dict):
            try:
                from omegaconf import OmegaConf

                rt_cfg = OmegaConf.to_container(rt_cfg, resolve=True)  # type: ignore[assignment]
            except Exception:
                rt_cfg = {}
        cls.reframe_cfg = _ReframeToolConfig(
            enable=bool(rt_cfg.get("enable", True)),
            trigger_score_threshold=float(rt_cfg.get("trigger_score_threshold", 0.0)),
            max_tool_uses=int(rt_cfg.get("max_tool_uses", 2)),
            require_model_request=bool(rt_cfg.get("require_model_request", True)),
            auto_action=str(rt_cfg.get("auto_action", "reframe")),
            tool_call_format=str(rt_cfg.get("tool_call_format", "hermes")),
            tool_backend=rt_cfg.get("tool_backend"),
            tool_model_name_or_path=rt_cfg.get("tool_model_name_or_path"),
            tool_device=rt_cfg.get("tool_device"),
            tool_dtype=rt_cfg.get("tool_dtype"),
            tool_base_url=rt_cfg.get("tool_base_url"),
            tool_api_key=rt_cfg.get("tool_api_key"),
            tool_timeout_s=float(rt_cfg["tool_timeout_s"]) if "tool_timeout_s" in rt_cfg else None,
            tool_max_concurrency=int(rt_cfg["tool_max_concurrency"]) if "tool_max_concurrency" in rt_cfg else None,
            tool_max_new_tokens=int(rt_cfg.get("tool_max_new_tokens", 256)),
            tool_temperature=float(rt_cfg.get("tool_temperature", 0.2)),
            tool_top_p=float(rt_cfg.get("tool_top_p", 0.95)),
            request_prompt=str(rt_cfg.get("request_prompt", _ReframeToolConfig.request_prompt)),
            reframe_prompt_template=str(rt_cfg.get("reframe_prompt_template", _ReframeToolConfig.reframe_prompt_template)),
            split_prompt_template=str(rt_cfg.get("split_prompt_template", _ReframeToolConfig.split_prompt_template)),
        )

        cls.tool_parser = ToolParser.get_tool_parser(cls.reframe_cfg.tool_call_format, cls.tokenizer)

        # Judge backend shared resources
        cls._judge_tokenizer = None
        cls._judge_model = None
        cls._remote_judge_semaphore = None

        backend = (cls.verifier_cfg.judge_backend or "policy").lower()
        if backend == "local":
            if not cls.verifier_cfg.judge_model_name_or_path:
                raise ValueError("judge_backend='local' requires judge_model_name_or_path")
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                judge_path = cls.verifier_cfg.judge_model_name_or_path
                cls._judge_tokenizer = AutoTokenizer.from_pretrained(judge_path, trust_remote_code=True)
                dtype = cls.verifier_cfg.judge_dtype.lower()
                if dtype == "float16":
                    torch_dtype = torch.float16
                elif dtype == "bfloat16":
                    torch_dtype = torch.bfloat16
                elif dtype == "float32":
                    torch_dtype = torch.float32
                else:
                    torch_dtype = "auto"

                cls._judge_model = AutoModelForCausalLM.from_pretrained(
                    judge_path, trust_remote_code=True, torch_dtype=torch_dtype
                )
                cls._judge_model.eval()
                device = cls.verifier_cfg.judge_device.lower()
                if device != "cpu":
                    cls._judge_model = cls._judge_model.to(device)
            except Exception as e:
                raise RuntimeError(f"Failed to load local judge model: {e}") from e
        elif backend == "remote":
            if not cls.verifier_cfg.judge_base_url:
                raise ValueError("judge_backend='remote' requires judge_base_url")
            if not cls.verifier_cfg.judge_model_name_or_path:
                raise ValueError("judge_backend='remote' requires judge_model_name_or_path (remote model name)")
            try:
                import asyncio

                mc = int(cls.verifier_cfg.judge_max_concurrency)
                cls._remote_judge_semaphore = asyncio.Semaphore(mc) if mc and mc > 0 else None
            except Exception:
                cls._remote_judge_semaphore = None

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages: list[dict[str, Any]] = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info") or {}
        reward_model = kwargs.get("reward_model") or {}

        # Subproblem source:
        # - preferred: extra_info.subproblems
        # - legacy: backward_hints
        subproblems: Optional[list[str]] = extra_info.get("subproblems")
        if subproblems is None:
            subproblems = extra_info.get("backward_hints")
        if subproblems is None:
            interaction_kwargs = extra_info.get("interaction_kwargs") or {}
            subproblems = interaction_kwargs.get("backward_hints")
        if subproblems is not None and not isinstance(subproblems, list):
            subproblems = None

        expected_answers = extra_info.get("expected_answers")

        # If no explicit subproblems list, treat the last user turn as the single subproblem.
        if not subproblems:
            last_user = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_user = str(m.get("content", ""))
                    break
            subproblems = [last_user] if last_user else []

        # Remaining future subproblems (queue). Assume subproblems[0] is already in raw_prompt if provided.
        queue: list[str] = list(subproblems[1:]) if len(subproblems) > 1 else []
        current_question = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                current_question = str(m.get("content", ""))
                break
        if not current_question and subproblems:
            current_question = str(subproblems[0])

        request_id = uuid4().hex
        metrics: dict[str, Any] = {}

        prompt_ids = await self._encode_full_messages(messages)

        response_mask: list[int] = []
        response_logprobs: list[float] = []
        step_token_spans: list[tuple[int, int]] = []
        turn_scores: list[float] = []

        # Debug metadata
        turn_scores_rule: list[Optional[float]] = []
        turn_scores_llm: list[Optional[float]] = []
        reframe_events: list[dict[str, Any]] = []

        assistant_turns = 0
        user_turns = 0
        tool_uses = 0

        while True:
            # 1) policy generate an answer
            with simple_timer("generate_sequences", metrics):
                out = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, image_data=None
                )

            assistant_turns += 1
            gen_ids = out.token_ids
            start = len(response_mask)
            end = start + len(gen_ids)
            step_token_spans.append((start, end))

            prompt_ids += gen_ids
            response_mask += [1] * len(gen_ids)
            if out.log_probs:
                response_logprobs += out.log_probs

            assistant_text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            )
            messages.append({"role": "assistant", "content": assistant_text})

            # 2) verify this step
            step_ctx = {
                "step_idx": len(turn_scores),
                "messages": messages,
                "question": current_question,
                "expected": (
                    expected_answers[len(turn_scores)]
                    if isinstance(expected_answers, list) and len(turn_scores) < len(expected_answers)
                    else None
                ),
            }
            verdict = await self._verify_step(request_id, assistant_text, step_ctx, sampling_params)
            judge_score = float(verdict["final"])
            step_reward = judge_score if bool(getattr(self.verifier_cfg, "step_reward_enable", True)) else 0.0
            turn_scores.append(float(step_reward))
            turn_scores_rule.append(verdict.get("rule"))
            turn_scores_llm.append(verdict.get("llm"))

            # termination: token budget / turn limits
            if len(response_mask) >= self.response_length:
                break
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                break
            if self.max_user_turns and user_turns >= self.max_user_turns:
                break

            # 3) low-score => allow meta-tool
            did_inject_new_question = False
            if (
                bool(getattr(self.reframe_cfg, "enable", False))
                and tool_uses < int(getattr(self.reframe_cfg, "max_tool_uses", 0) or 0)
                and judge_score <= float(getattr(self.reframe_cfg, "trigger_score_threshold", 0.0))
                and current_question
            ):
                # Ensure `action` is always defined even when we skip meta-tool triggering.
                action: Optional[dict[str, Any]] = None
                # If the low score is caused by a non-binary judge output (parse failure),
                # skip triggering meta-tool (avoid cascading on flaky judge formatting).
                if verdict.get("llm_parse_ok") is False and verdict.get("rule") is None:
                    # Still continue the normal queue progression below.
                    pass
                else:
                    # Option A: policy explicitly requests the meta-tool via <tool_call>...</tool_call>
                    if bool(getattr(self.reframe_cfg, "require_model_request", True)):
                        add_messages = [
                            {"role": "user", "content": str(getattr(self.reframe_cfg, "request_prompt", ""))}
                        ]
                        messages.extend(add_messages)
                        user_ids = await self._encode_incremental_messages(add_messages)
                        if len(response_mask) + len(user_ids) >= self.response_length:
                            break
                        prompt_ids += user_ids
                        response_mask += [0] * len(user_ids)
                        if response_logprobs:
                            response_logprobs += [0.0] * len(user_ids)
                        user_turns += 1

                        tool_sampling = dict(sampling_params)
                        tool_sampling["temperature"] = 0.0
                        tool_sampling["top_p"] = 1.0
                        # vLLM rollout server expects OpenAI-style `max_tokens` (NOT `max_new_tokens`).
                        tool_sampling.pop("max_new_tokens", None)
                        tool_sampling["max_tokens"] = min(128, int(tool_sampling.get("max_tokens", 128)))

                        with simple_timer("generate_sequences", metrics):
                            tool_out = await self.server_manager.generate(
                                request_id=f"tool_req_{request_id}_{len(turn_scores)}",
                                prompt_ids=prompt_ids,
                                sampling_params=tool_sampling,
                                image_data=None,
                            )

                        assistant_turns += 1
                        tool_ids = tool_out.token_ids
                        t_start = len(response_mask)
                        t_end = t_start + len(tool_ids)
                        step_token_spans.append((t_start, t_end))

                        prompt_ids += tool_ids
                        response_mask += [1] * len(tool_ids)
                        if tool_out.log_probs:
                            response_logprobs += tool_out.log_probs

                        # Add tool-request assistant message for context/debugging
                        tool_text = await self.loop.run_in_executor(
                            None, lambda: self.tokenizer.decode(tool_ids, skip_special_tokens=True)
                        )
                        messages.append({"role": "assistant", "content": tool_text})

                        # No direct verification for tool-request turn; allow future rewards to credit it.
                        turn_scores.append(0.0)
                        turn_scores_rule.append(None)
                        turn_scores_llm.append(None)

                        # Parse tool calls from this assistant output
                        _, tool_calls = await self.tool_parser.extract_tool_calls(tool_ids)
                        action = self._select_action_from_tool_calls(
                            tool_calls=tool_calls, current_question=current_question
                        )

                        # termination after tool-request turn
                        if len(response_mask) >= self.response_length:
                            break
                        if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                            break
                        if self.max_user_turns and user_turns >= self.max_user_turns:
                            break
                    else:
                        # Option B: auto action
                        action = self._auto_reframe_action(current_question=current_question)

                if action is not None:
                    tool_uses += 1
                    try:
                        new_questions = await self._execute_reframe_action(
                            action=action, question=current_question, answer=assistant_text
                        )
                    except Exception as e:
                        logger.warning("meta-tool execution failed: %s", e)
                        new_questions = []

                    reframe_events.append(
                        {
                            "step_idx": int(step_ctx["step_idx"]),
                            "trigger_score": float(judge_score),
                            "action": str(action.get("name")),
                            "arguments": action.get("arguments"),
                            "new_questions": new_questions,
                        }
                    )

                    if new_questions:
                        # Inject the first new question as the next user turn; push remaining to front of queue.
                        next_q = str(new_questions[0])
                        queue = [str(x) for x in new_questions[1:]] + queue
                        add_messages = [{"role": "user", "content": next_q}]
                        messages.extend(add_messages)
                        user_turns += 1

                        user_ids = await self._encode_incremental_messages(add_messages)
                        if len(response_mask) + len(user_ids) >= self.response_length:
                            break
                        prompt_ids += user_ids
                        response_mask += [0] * len(user_ids)
                        if response_logprobs:
                            response_logprobs += [0.0] * len(user_ids)

                        current_question = next_q
                        did_inject_new_question = True

            if did_inject_new_question:
                continue

            # 4) normal progression: feed next queued subproblem
            if not queue:
                break
            next_user = queue.pop(0)
            add_messages = [{"role": "user", "content": next_user}]
            messages.extend(add_messages)
            user_turns += 1

            user_ids = await self._encode_incremental_messages(add_messages)
            if len(response_mask) + len(user_ids) >= self.response_length:
                break
            prompt_ids += user_ids
            response_mask += [0] * len(user_ids)
            if response_logprobs:
                response_logprobs += [0.0] * len(user_ids)
            current_question = str(next_user)


        response_ids = prompt_ids[-len(response_mask) :] if response_mask else []
        prompt_only_ids = prompt_ids[: len(prompt_ids) - len(response_mask)] if response_mask else prompt_ids

        return AgentLoopOutput(
            prompt_ids=prompt_only_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            multi_modal_data={},
            # Provide dummy scalar reward_score to avoid async reward if reward_model.enable=False;
            # token-level scores are passed via extra_fields -> rm_scores in AgentLoopWorkerBase.
            reward_score=0.0,
            num_turns=assistant_turns + user_turns + 1,
            metrics=metrics,
            extra_fields={
                "turn_scores": turn_scores,
                "turn_scores_rule": turn_scores_rule,
                "turn_scores_llm": turn_scores_llm,
                "step_token_spans": step_token_spans,
                "final_reward": float(final_reward),
                "reframe_events": reframe_events,
            },
        )

    async def _encode_full_messages(self, messages: list[dict[str, Any]]) -> list[int]:
        return await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
            ),
        )

    async def _encode_incremental_messages(self, add_messages: list[dict[str, Any]]) -> list[int]:
        ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
        )
        return ids[len(self.system_prompt) :]

    async def _verify_step(
        self,
        request_id: str,
        assistant_text: str,
        step_ctx: dict[str, Any],
        sampling_params: dict[str, Any],
    ) -> dict[str, Optional[float]]:
        rule_score: Optional[float] = None
        if self.rule_verifier is not None:
            try:
                rule_score = self.rule_verifier(assistant_text, step_ctx)
            except Exception as e:
                logger.warning("rule_verifier failed: %s", e)
                rule_score = None

        llm_score: Optional[float] = None
        llm_parse_ok: Optional[bool] = None
        if bool(getattr(self.verifier_cfg, "llm_enable", False)):
            llm_score, llm_parse_ok = await self._llm_verify(request_id, assistant_text, step_ctx, sampling_params)

        # combine: prefer rule when exists else llm; if both exist take max
        vals = [v for v in [rule_score, llm_score] if v is not None]
        final = max(vals) if vals else 0.0
        final_f = float(max(0.0, min(1.0, final)))
        # NOTE: llm_parse_ok=False means judge returned non-binary output and we fell back to 0.
        return {"final": final_f, "rule": rule_score, "llm": llm_score, "llm_parse_ok": llm_parse_ok}

    async def _llm_verify(
        self,
        request_id: str,
        assistant_text: str,
        step_ctx: dict[str, Any],
        sampling_params: dict[str, Any],
    ) -> tuple[float, bool]:
        question = step_ctx.get("question") or ""
        prompt = self.verifier_cfg.llm_prompt_template.format(question=question, answer=assistant_text)
        verifier_messages = [
            {"role": "system", "content": "You are a verifier."},
            {"role": "user", "content": prompt},
        ]
        verifier_sampling = dict(sampling_params)
        # vLLM rollout server expects OpenAI-style `max_tokens` (NOT `max_new_tokens`).
        verifier_sampling.pop("max_new_tokens", None)
        verifier_sampling["max_tokens"] = int(self.verifier_cfg.llm_max_new_tokens)
        verifier_sampling["temperature"] = float(getattr(self.verifier_cfg, "llm_temperature", 0.0))
        verifier_sampling["top_p"] = float(getattr(self.verifier_cfg, "llm_top_p", 1.0))

        text = await self._completion_text(
            request_id=f"verifier_{request_id}_{step_ctx.get('step_idx', 0)}",
            messages=verifier_messages,
            sampling_params=verifier_sampling,
            backend=self.verifier_cfg.judge_backend,
            model_name_or_path=self.verifier_cfg.judge_model_name_or_path,
            base_url=self.verifier_cfg.judge_base_url,
            api_key=self.verifier_cfg.judge_api_key,
            timeout_s=float(self.verifier_cfg.judge_timeout_s),
        )
        score = _parse_binary_judge(text or "")
        if score is None:
            logger.warning("LLM step judge returned non-binary output, fallback to 0: %r", (text or "")[:200])
            return 0.0, False
        return float(score), True

    async def _decide_reframe_action(
        self,
        *args,
        **kwargs,
    ) -> Optional[dict[str, Any]]:
        # Deprecated: action decision is handled inside run() to keep token spans aligned.
        return None

    @staticmethod
    def _normalize_tool_name(name: str) -> str:
        return (name or "").strip().lower()

    def _auto_reframe_action(self, *, current_question: str) -> Optional[dict[str, Any]]:
        auto = self._normalize_tool_name(str(getattr(self.reframe_cfg, "auto_action", "reframe")))
        if auto == "split":
            return {"name": "split_subproblem", "arguments": {"subproblem": current_question}}
        return {"name": "reframe_subproblem", "arguments": {"subproblem": current_question}}

    def _select_action_from_tool_calls(self, *, tool_calls: list[Any], current_question: str) -> Optional[dict[str, Any]]:
        """
        tool_calls is a list of FunctionCall from ToolParser.
        """
        if not tool_calls:
            return self._auto_reframe_action(current_question=current_question)
        fc = tool_calls[0]
        name = self._normalize_tool_name(getattr(fc, "name", ""))
        try:
            args = json.loads(getattr(fc, "arguments", "") or "")
            if not isinstance(args, dict):
                args = {}
        except Exception:
            args = {}
        if name not in {"reframe_subproblem", "split_subproblem"}:
            return None
        if "subproblem" not in args or not str(args.get("subproblem") or "").strip():
            args["subproblem"] = current_question
        return {"name": name, "arguments": args}

    async def _execute_reframe_action(self, *, action: dict[str, Any], question: str, answer: str) -> list[str]:
        name = str(action.get("name") or "").strip().lower()
        if name not in {"reframe_subproblem", "split_subproblem"}:
            return []

        # Resolve tool backend config (defaults to verifier cfg)
        backend = (self.reframe_cfg.tool_backend or self.verifier_cfg.judge_backend or "policy").lower()
        model_name_or_path = self.reframe_cfg.tool_model_name_or_path or self.verifier_cfg.judge_model_name_or_path
        base_url = self.reframe_cfg.tool_base_url or self.verifier_cfg.judge_base_url
        api_key = self.reframe_cfg.tool_api_key or self.verifier_cfg.judge_api_key
        timeout_s = float(self.reframe_cfg.tool_timeout_s or self.verifier_cfg.judge_timeout_s or 120.0)

        if backend == "remote" and (not base_url or not model_name_or_path):
            raise ValueError("reframe_tool remote backend requires tool_base_url and tool_model_name_or_path")

        if name == "split_subproblem":
            prompt = self.reframe_cfg.split_prompt_template.format(question=question, answer=answer)
        else:
            prompt = self.reframe_cfg.reframe_prompt_template.format(question=question, answer=answer)

        messages = [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": prompt},
        ]

        tool_sampling = {
            # Use OpenAI-style max_tokens so this works for both remote OpenAI endpoints and vLLM policy backend.
            "max_tokens": int(getattr(self.reframe_cfg, "tool_max_new_tokens", 256)),
            "temperature": float(getattr(self.reframe_cfg, "tool_temperature", 0.2)),
            "top_p": float(getattr(self.reframe_cfg, "tool_top_p", 0.95)),
        }
        text = await self._completion_text(
            request_id=f"meta_tool_{uuid4().hex}",
            messages=messages,
            sampling_params=tool_sampling,
            backend=backend,
            model_name_or_path=model_name_or_path,
            base_url=base_url,
            api_key=api_key,
            timeout_s=timeout_s,
        )
        obj = _extract_json_obj(text or "")
        if not obj:
            return []

        if name == "split_subproblem":
            subs = obj.get("subproblems")
            if isinstance(subs, list):
                out = [str(s).strip() for s in subs if str(s).strip()]
                return out
            # tolerate single string
            if isinstance(subs, str) and subs.strip():
                return [subs.strip()]
            return []

        sp = obj.get("subproblem")
        if isinstance(sp, str) and sp.strip():
            return [sp.strip()]
        return []

    async def _completion_text(
        self,
        *,
        request_id: str,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
        backend: str,
        model_name_or_path: Optional[str],
        base_url: Optional[str],
        api_key: Optional[str],
        timeout_s: float,
    ) -> str:
        """
        Backend:
        - policy: use current rollout server_manager (policy model) to generate
        - local : use transformers (judge model) loaded in worker
        - remote: call OpenAI-compatible HTTP endpoint
        """
        backend = (backend or "policy").lower()

        if backend == "local":
            if self._judge_model is None or self._judge_tokenizer is None:
                raise RuntimeError("Local backend requested but local model/tokenizer not loaded.")
            import torch

            # Build prompt using judge tokenizer if possible
            def build_text():
                tok = self._judge_tokenizer
                if hasattr(tok, "apply_chat_template"):
                    try:
                        return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    except Exception:
                        pass
                parts = []
                for m in messages:
                    parts.append(f"[{m.get('role','user').upper()}] {m.get('content','')}")
                parts.append("[ASSISTANT] ")
                return "\n".join(parts)

            prompt_text = await self.loop.run_in_executor(None, build_text)
            inputs = self._judge_tokenizer(prompt_text, return_tensors="pt")
            device = str(getattr(self.verifier_cfg, "judge_device", "cpu") or "cpu").lower()
            if device != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}

            max_new = int(sampling_params.get("max_new_tokens") or sampling_params.get("max_tokens") or 128)
            with torch.no_grad():
                out_ids = self._judge_model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,
                    pad_token_id=getattr(self._judge_tokenizer, "pad_token_id", None)
                    or getattr(self._judge_tokenizer, "eos_token_id", None),
                    eos_token_id=getattr(self._judge_tokenizer, "eos_token_id", None),
                )
            gen = out_ids[0][inputs["input_ids"].shape[1] :]
            return self._judge_tokenizer.decode(gen, skip_special_tokens=True)

        if backend == "remote":
            import aiohttp

            base = str(base_url or "").rstrip("/")
            url = f"{base}/v1/chat/completions" if not base.endswith("/v1") else f"{base}/chat/completions"

            max_tokens = sampling_params.get("max_tokens")
            if max_tokens is None:
                max_tokens = sampling_params.get("max_new_tokens", 128)

            payload: dict[str, Any] = {
                "model": str(model_name_or_path),
                "messages": messages,
                "temperature": float(sampling_params.get("temperature", 0.0)),
                "top_p": float(sampling_params.get("top_p", 1.0)),
                "max_tokens": int(max_tokens),
            }
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            timeout = aiohttp.ClientTimeout(total=float(timeout_s))
            session = aiohttp.ClientSession(timeout=timeout)
            try:
                sem = getattr(self.__class__, "_remote_judge_semaphore", None)

                async def do_post():
                    async with session.post(url, headers=headers, json=payload) as resp:
                        text = await resp.text()
                        if resp.status >= 400:
                            logger.warning("Remote backend HTTP %s (fallback empty): %s", resp.status, text[:400])
                            return ""
                        data = json.loads(text)
                        choices = data.get("choices") or []
                        if not choices:
                            return ""
                        msg = (choices[0].get("message") or {})
                        return str(msg.get("content") or "")

                if sem is None:
                    return await do_post()
                async with sem:
                    return await do_post()
            except (TimeoutError, OSError, aiohttp.ClientError) as e:
                logger.warning("Remote backend request failed (fallback empty): %s", e)
                return ""
            finally:
                await session.close()

        # policy backend
        # vLLM rollout server expects OpenAI-style `max_tokens`.
        sampling_params = dict(sampling_params)
        if "max_new_tokens" in sampling_params and "max_tokens" not in sampling_params:
            sampling_params["max_tokens"] = sampling_params.pop("max_new_tokens")
        else:
            sampling_params.pop("max_new_tokens", None)
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
            ),
        )
        out = await self.server_manager.generate(
            request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, image_data=None
        )
        return await self.loop.run_in_executor(None, lambda: self.tokenizer.decode(out.token_ids, skip_special_tokens=True))


