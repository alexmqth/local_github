#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultimate Causal Inference Script (Fixed Sharding & Anti-Repetition).
1. [FIXED] Restored Data Sharding logic! Now splits dataset across GPUs correctly.
   (Previously, every GPU was processing the full dataset, causing 8x slowdown).
2. [NEW] Added robust Anti-Repetition mechanism to prevent infinite loops and save compute.
3. Keeps Robust Parsing & Real-time Acc.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional


#_CAUSAL_REASONINGMODEL_NAME = "MakimaSasha/CausalReasoningModel"
_CAUSAL_REASONINGMODEL_NAME = "/home/projects/hku-medai/larrypl/code/mq/causal-reasoning/hf_models/gsm8k_deberta_regression"


def _clean_path(p: str) -> str:
    return os.path.expanduser(str(p)).strip().rstrip("\r")


def _strip_dollar(s: str) -> str:
    t = (s or "").strip()
    if t.startswith("$$") and t.endswith("$$") and len(t) >= 4:
        return t[2:-2].strip()
    if t.startswith("$") and t.endswith("$") and len(t) >= 2:
        return t[1:-1].strip()
    return t


def _normalize_answer_text(s: str) -> str:
    t = _strip_dollar(str(s or "").strip())
    t = t.strip()
    t = t.rstrip(" .;，。；! \n\r\t")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _extract_last_boxed_content(text: str) -> Optional[str]:
    if not isinstance(text, str) or not text.strip():
        return None
    t = text
    starts = [m.start() for m in re.finditer(r"\\boxed\s*\{", t)]
    if not starts:
        return None

    def parse_from(idx: int) -> Optional[tuple[int, int]]:
        m = re.search(r"\\boxed\s*\{", t[idx:])
        if not m:
            return None
        open_brace_pos = idx + m.end() - 1
        i = open_brace_pos + 1
        depth = 1
        while i < len(t):
            ch = t[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return (open_brace_pos + 1, i)
            i += 1
        return None

    for s_idx in reversed(starts):
        span = parse_from(s_idx)
        if not span:
            continue
        a, b = span
        inner = t[a:b]
        inner = _normalize_answer_text(inner)
        return inner if inner else None
    return None


def _extract_answer_from_model_text(text: str) -> Optional[str]:
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.strip()
    blocks = re.findall(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", t, flags=re.IGNORECASE | re.DOTALL)
    if blocks:
        blk = blocks[-1].strip()
        boxed = _extract_last_boxed_content(blk)
        if boxed is not None:
            return boxed
        return _normalize_answer_text(blk) or None
    boxed = _extract_last_boxed_content(t)
    if boxed is not None:
        return boxed
    return None


def _extract_gt_boxed_from_solution(solution_text: str) -> Optional[str]:
    return _extract_last_boxed_content(solution_text)


def _extract_gsm8k_answer(answer_text: str) -> Optional[str]:
    if not isinstance(answer_text, str) or not answer_text.strip():
        return None
    match = re.search(r"####\s*(.+?)$", answer_text.strip(), flags=re.MULTILINE)
    if match:
        ans = match.group(1).strip()
        ans = ans.replace(",", "")
        return _normalize_answer_text(ans) if ans else None
    return None


def _extract_answer_gsm8k_style(text: str) -> Optional[str]:
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.strip()
    gsm_ans = _extract_gsm8k_answer(t)
    if gsm_ans is not None:
        return gsm_ans
    blocks = re.findall(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", t, flags=re.IGNORECASE | re.DOTALL)
    if blocks:
        blk = blocks[-1].strip()
        inner_gsm = _extract_gsm8k_answer(blk)
        if inner_gsm is not None:
            return inner_gsm
        boxed = _extract_last_boxed_content(blk)
        if boxed is not None:
            return boxed
        blk_clean = blk.replace(",", "")
        return _normalize_answer_text(blk_clean) or None
    boxed = _extract_last_boxed_content(t)
    if boxed is not None:
        return boxed
    match = re.search(r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*([^\n.]+)", t, flags=re.IGNORECASE)
    if match:
        ans = match.group(1).strip()
        ans = ans.replace(",", "")
        return _normalize_answer_text(ans) if ans else None
    return None


def _has_answer_marker(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.lower()
    if "<answer" in t:
        return True
    if "\\boxed" in text:
        return True
    if re.search(r"final\s+answer", text, flags=re.IGNORECASE):
        return True
    if "####" in text:
        return True
    if re.search(r"the\s+answer\s+is", text, flags=re.IGNORECASE):
        return True
    return False


def _build_causalreasoningmodel_text(*, question: str, prefix_steps: list[str], candidate_step: str) -> str:
    prefix_text = "\n".join([str(s).strip() for s in (prefix_steps or []) if str(s).strip()])
    return (
        f"Question:\n{str(question or '').strip()}\n\n"
        f"Prefix steps:\n{prefix_text}\n\n"
        f"Candidate step:\n{str(candidate_step or '').strip()}"
    )


def _build_prompt_text_fallback(messages: list[dict[str, Any]]) -> str:
    parts = []
    for m in messages:
        role = (m.get("role") or "user").upper()
        content = m.get("content") or ""
        parts.append(f"[{role}] {content}")
    parts.append("[ASSISTANT] ")
    return "\n".join(parts)


def _encode_chat(tokenizer, messages: list[dict[str, Any]]):
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt")
        except TypeError:
            ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            import torch
            return torch.tensor([ids], dtype=torch.long)
        except Exception:
            pass
    txt = _build_prompt_text_fallback(messages)
    return tokenizer(txt, return_tensors="pt").input_ids


@dataclass
class CausalLoopCfg:
    pns_mode: str = "discrete"
    pns_threshold: float = 0.6
    max_retries: int = 2
    max_steps: int = 32
    pns_feedback_template: str = (
        "This step is not acceptable: causal logic is not rigorous.\n"
        "Please rewrite ONLY this step and try again. Requirements:\n"
        "- Make the reasoning causally coherent with the prior context\n"
        "- skip unecessaryh steps\n"
        "- Do not add multiple steps at once (only generate 1-3 sentences)\n"
    )
    pns_ok_continue_msg: str = "Good. Continue with the next step (one step only)."
    pns_ok_finish_msg: str = "Good. Now provide the complete final solution and end with <ANSWER>...</ANSWER>."
    pns_proceed_after_penalty_msg: str = "Too many retries. Proceed to the next step anyway (one step only)."


@dataclass
class RunResult:
    final_text: str
    pred_boxed: Optional[str]
    steps: list[str]
    pns: list[int]
    retry_counts: list[int]
    asked_finish: bool


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_model_path", type=str, required=True)
    ap.add_argument("--policy_tokenizer_path", type=str, default="", help="Optional base tokenizer path")
    
    ap.add_argument("--judge_model_path", type=str, default=_CAUSAL_REASONINGMODEL_NAME)
    ap.add_argument("--policy_device", type=str, default="cuda")
    ap.add_argument("--judge_device", type=str, default="cpu")
    ap.add_argument("--policy_dtype", type=str, default="auto")
    ap.add_argument("--judge_dtype", type=str, default="auto")
    ap.add_argument("--trust_remote_code", action="store_true")
    
    ap.add_argument("--max_steps", type=int, default=32)
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("--step_max_new_tokens", type=int, default=256)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--force_finish_after_steps", type=int, default=0)
    ap.add_argument("--verbose_steps", action="store_true")
    
    ap.add_argument("--system_prompt", type=str, default="You are a helpful assistant.")
    ap.add_argument("--user_prefix", type=str, default="Solve the problem step by step. Provide the first step (one step only).")
    
    ap.add_argument("--dataset_name", type=str, default="HuggingFaceH4/MATH-500")
    ap.add_argument("--dataset_config", type=str, default=None)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--max_examples", type=int, default=0)
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset_type", type=str, default="auto", choices=["auto", "math", "gsm8k"])

    ap.add_argument("--profile", action="store_true", help="Enable profiling")
    ap.add_argument("--profile_every", type=int, default=25)

    args = ap.parse_args()

    args.policy_model_path = _clean_path(args.policy_model_path)
    args.policy_tokenizer_path = _clean_path(args.policy_tokenizer_path) if args.policy_tokenizer_path else ""

    import torch
    from datasets import load_dataset
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(int(args.seed))

    def resolve_device(req: str) -> str:
        r = (req or "cpu").lower()
        if r != "cpu":
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return "cpu"

    policy_device = resolve_device(args.policy_device)
    judge_device = resolve_device(args.judge_device)

    def resolve_dtype(req: str):
        r = (req or "auto").lower()
        if r == "float16":
            return torch.float16
        if r == "bfloat16":
            return torch.bfloat16
        if r == "float32":
            return torch.float32
        return "auto"

    policy_dtype = resolve_dtype(args.policy_dtype)
    judge_dtype = resolve_dtype(args.judge_dtype)

    tok_path = args.policy_tokenizer_path if args.policy_tokenizer_path else args.policy_model_path
    policy_tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=bool(args.trust_remote_code))
    
    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token

    policy_model = AutoModelForCausalLM.from_pretrained(
        args.policy_model_path,
        trust_remote_code=bool(args.trust_remote_code),
        torch_dtype=policy_dtype,
        device_map=None,
    )
    policy_model.eval()
    policy_model.to(policy_device)

    judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_model_path, use_fast=True)
    judge_model = AutoModelForSequenceClassification.from_pretrained(args.judge_model_path, torch_dtype=judge_dtype)
    judge_model.eval()
    judge_model.to(judge_device)

    if args.dataset_config:
        ds = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    else:
        ds = load_dataset(args.dataset_name, split=args.split)
    if int(args.max_examples) > 0:
        ds = ds.select(range(min(int(args.max_examples), len(ds))))

    # ✅ [CRITICAL FIX] 数据分片逻辑！
    # 读取环境变量 RANK 和 WORLD_SIZE（由 bash 脚本自动注入）
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        # 按照 GPU 数量对数据集进行切分
        ds = ds.select(range(rank, len(ds), world_size))
        print(f"[INFO] Rank {rank}/{world_size}: Processing {len(ds)} examples (Sharded)")
    else:
        print(f"[INFO] Rank {rank}: Processing {len(ds)} examples (Full Dataset)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)) or ".", exist_ok=True)
    cfg = CausalLoopCfg(max_steps=int(args.max_steps), max_retries=int(args.max_retries))

    dataset_type = str(args.dataset_type or "auto").lower()
    if dataset_type == "auto":
        ds_name_lower = str(args.dataset_name or "").lower()
        if "gsm8k" in ds_name_lower or "gsm" in ds_name_lower:
            dataset_type = "gsm8k"
        elif "math" in ds_name_lower:
            dataset_type = "math"
        else:
            cols = ds.column_names if hasattr(ds, "column_names") else []
            if "question" in cols and "answer" in cols:
                dataset_type = "gsm8k"
            else:
                dataset_type = "math"
    
    pbar = tqdm(ds, desc=f"{args.dataset_name} infer [Rank {rank}]")
    n_total = 0
    n_correct = 0

    prof_wall = 0.0
    
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for i, ex in enumerate(pbar):
            if dataset_type == "gsm8k":
                problem = str(ex.get("question", "") or "")
                answer_raw = str(ex.get("answer", "") or "")
                gt = _extract_gsm8k_answer(answer_raw)
            else:
                problem = str(ex.get("problem", "") or "")
                solution = str(ex.get("solution", "") or "")
                gt = _extract_gt_boxed_from_solution(solution)
            gt_norm = _normalize_answer_text(gt) if gt is not None else None

            user_msg = problem
            if str(args.user_prefix or "").strip():
                user_msg = f"{problem}\n\n{str(args.user_prefix).strip()}"
            messages0 = [
                {"role": "system", "content": str(args.system_prompt or "").strip()},
                {"role": "user", "content": user_msg.strip()},
            ]
            
            t0_example = time.perf_counter()

            def run_with_seed() -> RunResult:
                import torch as _torch
                messages = list(messages0)
                accepted_steps: list[str] = []
                pns_scores: list[int] = []
                retry_counts: list[int] = []
                retry_for_current_step = 0
                step_idx = 0
                asked_finish = False
                last_assistant_text = ""

                while True:
                    if cfg.max_steps > 0 and step_idx >= int(cfg.max_steps):
                        break

                    input_ids = _encode_chat(policy_tokenizer, messages).to(policy_device)

                    gen_kwargs: dict[str, Any] = {
                        "max_new_tokens": int(args.step_max_new_tokens), 
                        "do_sample": bool(args.do_sample),
                        "pad_token_id": policy_tokenizer.pad_token_id,
                    }
                    if bool(args.do_sample):
                        gen_kwargs["temperature"] = float(args.temperature)
                        gen_kwargs["top_p"] = float(args.top_p)
                    else:
                        gen_kwargs["temperature"] = 1.0
                        gen_kwargs["top_p"] = 1.0
                    
                    eos_id = getattr(policy_tokenizer, "eos_token_id", None)
                    if eos_id is not None:
                        gen_kwargs["eos_token_id"] = eos_id

                    with _torch.no_grad():
                        out_ids = policy_model.generate(
                            input_ids=input_ids, 
                            **gen_kwargs
                        )
                    gen = out_ids[0][input_ids.shape[1] :]
                    assistant_text = policy_tokenizer.decode(gen, skip_special_tokens=True)
                    assistant_text = str(assistant_text or "")
                    last_assistant_text = assistant_text
                    
                    # ===============================================
                    # [NEW] Anti-Repetition Mechanism (防复读机制)
                    # ===============================================
                    is_repeating = False
                    cur_clean = assistant_text.strip().lower()

                    if len(accepted_steps) > 0:
                        prev_clean = accepted_steps[-1].strip().lower()
                        # Condition 1: Exact match with the immediate previous step
                        if cur_clean == prev_clean:
                            is_repeating = True
                        # Condition 2: Current step is a substantial substring of the previous step
                        elif len(cur_clean) > 10 and cur_clean in prev_clean:
                            is_repeating = True
                        # Condition 3: Exact match with ANY previous step in the history
                        elif any(cur_clean == step.strip().lower() for step in accepted_steps):
                            is_repeating = True

                    if is_repeating:
                        # Force reject without appending to context or calling Judge
                        pns_discrete = 0
                        passed = False
                        finished = False
                        if bool(args.verbose_steps):
                            print(f"[step={step_idx} pns=0 (REJECTED: Repetition Detected)]")
                    else:
                        # Append to messages only if not repeating
                        messages.append({"role": "assistant", "content": assistant_text})
                        finished = bool(_has_answer_marker(assistant_text))

                        # Call Judge Model since it's a valid new step
                        judge_text = _build_causalreasoningmodel_text(
                            question=str(problem or ""),
                            prefix_steps=list(accepted_steps),
                            candidate_step=str(assistant_text or ""),
                        )
                        j_inputs = judge_tokenizer(
                            [judge_text],
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors="pt",
                        )
                        j_inputs = {k: v.to(judge_device) for k, v in j_inputs.items()}
                        with _torch.no_grad():
                            j_out = judge_model(**j_inputs)
                            logits = j_out.logits
                            if logits.dim() == 2 and logits.size(-1) == 2:
                                pred1 = _torch.argmax(logits, dim=-1).float()
                                pns_discrete = int(pred1.detach().cpu().item())
                            else:
                                prob1 = _torch.sigmoid(logits.view(-1))
                                pns_discrete = int((prob1 >= 0.5).float().detach().cpu().item())
                        passed = bool(pns_discrete == 1)

                        if bool(args.verbose_steps):
                            print(f"[step={step_idx} pns={pns_discrete}]")

                    pns_scores.append(int(pns_discrete))

                    # ===============================================
                    # Proceed logic based on pass/fail
                    # ===============================================
                    if passed:
                        accepted_steps.append(assistant_text)
                        retry_counts.append(int(retry_for_current_step))
                        retry_for_current_step = 0
                        step_idx += 1
                        if finished: break
                        if (not asked_finish) and int(args.force_finish_after_steps) > 0 and step_idx >= int(args.force_finish_after_steps):
                            asked_finish = True
                            messages.append({"role": "user", "content": str(cfg.pns_ok_finish_msg)})
                        else:
                            messages.append({"role": "user", "content": str(cfg.pns_ok_continue_msg)})
                    else:
                        retry_for_current_step += 1
                        retry_counts.append(int(retry_for_current_step))
                        if retry_for_current_step > int(cfg.max_retries):
                            accepted_steps.append(assistant_text)
                            retry_for_current_step = 0
                            step_idx += 1
                            messages.append({"role": "user", "content": str(cfg.pns_proceed_after_penalty_msg)})
                        else:
                            messages.append({"role": "user", "content": str(cfg.pns_feedback_template)})

                    if len(messages) > 2 * (cfg.max_steps * (cfg.max_retries + 2) + 8):
                        break

                pred_boxed = _extract_answer_from_model_text(last_assistant_text)
                return RunResult(str(last_assistant_text or ""), pred_boxed, accepted_steps, pns_scores, retry_counts, asked_finish)

            rr = run_with_seed()
            
            t_ex = time.perf_counter() - t0_example
            prof_wall += t_ex

            if dataset_type == "gsm8k":
                pred_ans = _extract_answer_gsm8k_style(rr.final_text)
            else:
                pred_ans = rr.pred_boxed
            pred_norm = _normalize_answer_text(pred_ans) if pred_ans is not None else None
            correct = bool(gt_norm is not None and pred_norm is not None and pred_norm == gt_norm)

            n_total += 1
            n_correct += int(correct)
            pbar.set_description(f"{args.dataset_name} acc={n_correct/n_total:.4f}")
            
            f.write(json.dumps({
                "idx": i, "problem": problem, "gt": gt_norm, "pred": pred_norm, "correct": correct, "steps": rr.steps
            }, ensure_ascii=False) + "\n")
            
            if args.profile and (i + 1) % args.profile_every == 0:
                print(f"[PROF][Rank {rank}] idx={i} avg_wall={prof_wall / (i+1):.2f}s acc={n_correct/n_total:.4f}")

    print(f"[DONE][Rank {rank}] acc={n_correct/n_total:.4f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
