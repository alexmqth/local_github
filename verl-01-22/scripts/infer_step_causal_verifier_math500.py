#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline inference + evaluation script for a policy model trained with
`verl/experimental/agent_loop/step_causal_verifier.py`.

Key behavior:
- Use a chat-style prompt (system + user) as the initial prompt.
- Generate one "step" at a time with the policy model.
- For each generated step, run a DeBERTa-style seq-classification judge
  (MakimaSasha/CausalReasoningModel) to produce a discrete PNS in {0,1}.
  If PNS==0, append feedback and retry generating ONLY this step.
  If PNS==1, append a "continue" message and proceed to the next step.
- Stop when the policy emits an answer marker (<ANSWER>...</ANSWER> or \\boxed{...} or "final answer").
- Evaluate on HuggingFaceH4/MATH-500 test split using exact match over the final boxed answer.

Usage example:
  python scripts/infer_step_causal_verifier_math500.py ^
    --policy_model_path path/to/your/checkpoint ^
    --out_jsonl outputs/math500_preds.jsonl ^
    --max_examples 50
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Optional


_CAUSAL_REASONINGMODEL_NAME = "MakimaSasha/CausalReasoningModel"


def _strip_dollar(s: str) -> str:
    t = (s or "").strip()
    if t.startswith("$$") and t.endswith("$$") and len(t) >= 4:
        return t[2:-2].strip()
    if t.startswith("$") and t.endswith("$") and len(t) >= 2:
        return t[1:-1].strip()
    return t


def _normalize_answer_text(s: str) -> str:
    # Keep this conservative: user asked for exact match, but strip superficial wrappers/punct.
    t = _strip_dollar(str(s or "").strip())
    t = t.strip()
    t = t.rstrip(" .;，。；! \n\r\t")
    # collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _extract_last_boxed_content(text: str) -> Optional[str]:
    """
    Extract the content of the LAST LaTeX \\boxed{...} in `text`.
    Supports nested braces by doing a small brace-matching parse.
    Returns the inner content (without outer braces), stripped.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    t = text
    # Find all occurrences of "\boxed{"
    starts = [m.start() for m in re.finditer(r"\\boxed\s*\{", t)]
    if not starts:
        return None

    def parse_from(idx: int) -> Optional[tuple[int, int]]:
        # idx points to the start of "\boxed{"
        m = re.search(r"\\boxed\s*\{", t[idx:])
        if not m:
            return None
        open_brace_pos = idx + m.end() - 1  # position of '{'
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

    # Try from last to first
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
    """
    Policy output may contain:
    - <ANSWER>...</ANSWER> (preferred)
    - \\boxed{...} (fallback)
    Return the boxed-inner if present, else return content in <ANSWER> tag, else None.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.strip()

    # Prefer <ANSWER>...</ANSWER> blocks
    blocks = re.findall(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", t, flags=re.IGNORECASE | re.DOTALL)
    if blocks:
        blk = blocks[-1].strip()
        boxed = _extract_last_boxed_content(blk)
        if boxed is not None:
            return boxed
        return _normalize_answer_text(blk) or None

    # Fallback: last boxed in whole text
    boxed = _extract_last_boxed_content(t)
    if boxed is not None:
        return boxed
    return None


def _extract_gt_boxed_from_solution(solution_text: str) -> Optional[str]:
    # Ground truth is specified as solution column, but we only use boxed final answer.
    return _extract_last_boxed_content(solution_text)


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
    return False


def _build_causalreasoningmodel_text(*, question: str, prefix_steps: list[str], candidate_step: str) -> str:
    # Match the template in `step_causal_verifier.py` (and run_deberta_judge_batch.py)
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


def _encode_chat_ids(tokenizer, messages: list[dict[str, Any]]) -> list[int]:
    """
    Return token ids for a chat conversation as a python list[int].
    Prefer `apply_chat_template(..., tokenize=True)` if available.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            if isinstance(ids, list):
                return [int(x) for x in ids]
        except Exception:
            pass
    txt = _build_prompt_text_fallback(messages)
    enc = tokenizer(txt, add_special_tokens=True, return_tensors=None)
    ids2 = enc.get("input_ids") or []
    if isinstance(ids2, list) and ids2 and isinstance(ids2[0], list):
        ids2 = ids2[0]
    return [int(x) for x in (ids2 or [])]


def _pad_batch(tokenizer, batch_ids: list[list[int]]):
    """
    Pad a batch of token-id lists into (input_ids, attention_mask) torch tensors.
    """
    import torch

    # Use tokenizer.pad when possible (handles left/right pad correctly per tokenizer settings).
    try:
        padded = tokenizer.pad(
            {"input_ids": batch_ids},
            padding=True,
            return_tensors="pt",
        )
        return padded["input_ids"], padded["attention_mask"]
    except Exception:
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tokenizer, "eos_token_id", 0) or 0
        max_len = max((len(x) for x in batch_ids), default=0)
        input_ids = torch.full((len(batch_ids), max_len), int(pad_id), dtype=torch.long)
        attn = torch.zeros((len(batch_ids), max_len), dtype=torch.long)
        for i, ids in enumerate(batch_ids):
            if not ids:
                continue
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[i, : len(ids)] = 1
        return input_ids, attn


@dataclass
class CausalLoopCfg:
    # These defaults are aligned to your description (discrete PNS gating).
    pns_mode: str = "discrete"  # discrete | float (we use discrete here)
    pns_threshold: float = 0.6  # only used for float mode
    max_retries: int = 2
    max_steps: int = 32

    # Messages: use your training templates by default
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


@dataclass
class _State:
    idx: int
    problem: str
    gt_boxed: Optional[str]
    messages: list[dict[str, Any]]
    accepted_steps: list[str]
    pns_scores: list[int]
    retry_counts: list[int]
    retry_for_current_step: int
    step_idx: int
    asked_finish: bool
    assistant_turns: int
    user_turns: int
    final_text: str
    done: bool


def run_step_causal_loop(
    *,
    policy_model,
    policy_tokenizer,
    policy_device: str,
    judge_model,
    judge_tokenizer,
    judge_device: str,
    problem_text: str,
    cfg: CausalLoopCfg,
    # generation knobs
    step_max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    force_finish_after_steps: int,
    verbose: bool,
) -> RunResult:
    import torch

    messages: list[dict[str, Any]] = []

    # Initial prompt: allow user to fully control via system_prompt/user_prefix in CLI.
    # Here we just assume caller already formed a good system+user message.
    # (We keep it as "messages already contains system/user".)
    # NOTE: In this function, we expect the caller to pass problem_text and we add user message only.
    messages.append({"role": "user", "content": str(problem_text or "").strip()})

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

        # Generate one step
        ids = _encode_chat_ids(policy_tokenizer, messages)
        input_ids, attn = _pad_batch(policy_tokenizer, [ids])
        input_ids = input_ids.to(policy_device)
        attn = attn.to(policy_device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(step_max_new_tokens),
            "do_sample": bool(do_sample),
        }
        if bool(do_sample):
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(top_p)
        else:
            # ensure deterministic
            gen_kwargs["temperature"] = 1.0
            gen_kwargs["top_p"] = 1.0

        pad_id = getattr(policy_tokenizer, "pad_token_id", None) or getattr(policy_tokenizer, "eos_token_id", None)
        eos_id = getattr(policy_tokenizer, "eos_token_id", None)
        if pad_id is not None:
            gen_kwargs["pad_token_id"] = pad_id
        if eos_id is not None:
            gen_kwargs["eos_token_id"] = eos_id

        with torch.no_grad():
            out_ids = policy_model.generate(input_ids=input_ids, attention_mask=attn, **gen_kwargs)

        in_len = int(attn[0].sum().item())
        gen = out_ids[0][in_len:]
        assistant_text = policy_tokenizer.decode(gen, skip_special_tokens=True)
        assistant_text = str(assistant_text or "")
        last_assistant_text = assistant_text
        messages.append({"role": "assistant", "content": assistant_text})

        finished = bool(_has_answer_marker(assistant_text))

        # Judge PNS (seqcls)
        judge_text = _build_causalreasoningmodel_text(
            question=str(problem_text or ""),
            prefix_steps=list(accepted_steps),
            candidate_step=str(assistant_text or ""),
        )
        inputs = judge_tokenizer(
            [judge_text],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(judge_device) for k, v in inputs.items()}

        with torch.no_grad():
            j_out = judge_model(**inputs)
            logits = j_out.logits
            if logits.dim() == 2 and logits.size(-1) == 2:
                pred1 = torch.argmax(logits, dim=-1).float()
                pns_discrete = int(pred1.detach().cpu().item())
            else:
                prob1 = torch.sigmoid(logits.view(-1))
                pns_discrete = int((prob1 >= 0.5).float().detach().cpu().item())

        pns_scores.append(int(pns_discrete))

        passed = bool(pns_discrete == 1) if str(cfg.pns_mode).lower() == "discrete" else bool(
            float(pns_discrete) >= float(cfg.pns_threshold)
        )

        if verbose:
            print(
                f"[step={step_idx} attempt={retry_for_current_step} pns={pns_discrete} passed={passed} finished={finished}]"
            )

        if passed:
            accepted_steps.append(assistant_text)
            retry_counts.append(int(retry_for_current_step))
            retry_for_current_step = 0
            step_idx += 1

            if finished:
                break

            # Optionally ask for final solution once
            if (not asked_finish) and int(force_finish_after_steps) > 0 and step_idx >= int(force_finish_after_steps):
                asked_finish = True
                messages.append({"role": "user", "content": str(cfg.pns_ok_finish_msg)})
            else:
                messages.append({"role": "user", "content": str(cfg.pns_ok_continue_msg)})

        else:
            retry_for_current_step += 1
            retry_counts.append(int(retry_for_current_step))

            if retry_for_current_step > int(cfg.max_retries):
                # proceed anyway
                accepted_steps.append(assistant_text)
                retry_for_current_step = 0
                step_idx += 1
                messages.append({"role": "user", "content": str(cfg.pns_proceed_after_penalty_msg)})
            else:
                messages.append({"role": "user", "content": str(cfg.pns_feedback_template)})

        # Safety: avoid infinite loops on degenerate models
        if len(messages) > 2 * (cfg.max_steps * (cfg.max_retries + 2) + 8):
            break

    pred_boxed = _extract_answer_from_model_text(last_assistant_text)
    return RunResult(
        final_text=str(last_assistant_text or ""),
        pred_boxed=pred_boxed,
        steps=accepted_steps,
        pns=pns_scores,
        retry_counts=retry_counts,
        asked_finish=asked_finish,
    )


def _should_stop_by_turn_limits(
    *,
    assistant_turns: int,
    user_turns: int,
    total_messages: int,
    max_assistant_turns: int,
    max_user_turns: int,
    max_total_messages: int,
) -> bool:
    if int(max_assistant_turns) > 0 and int(assistant_turns) >= int(max_assistant_turns):
        return True
    if int(max_user_turns) > 0 and int(user_turns) >= int(max_user_turns):
        return True
    if int(max_total_messages) > 0 and int(total_messages) >= int(max_total_messages):
        return True
    return False


def run_step_causal_loop_batch(
    *,
    states: list[_State],
    policy_model,
    policy_tokenizer,
    policy_device: str,
    judge_model,
    judge_tokenizer,
    judge_device: str,
    cfg: CausalLoopCfg,
    step_max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    force_finish_after_steps: int,
    max_assistant_turns: int,
    max_user_turns: int,
    max_total_messages: int,
    verbose: bool,
) -> None:
    """
    Step-synchronous batched inference:
    - For all unfinished states, generate one assistant step in a single `model.generate` batch.
    - Judge all candidate steps in a single seqcls forward batch.
    - Update each state (retry/accept/feedback) independently.
    """
    import torch

    # Precompute generation kwargs
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": int(step_max_new_tokens),
        "do_sample": bool(do_sample),
    }
    if bool(do_sample):
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)
    else:
        gen_kwargs["temperature"] = 1.0
        gen_kwargs["top_p"] = 1.0

    pad_id = getattr(policy_tokenizer, "pad_token_id", None) or getattr(policy_tokenizer, "eos_token_id", None)
    eos_id = getattr(policy_tokenizer, "eos_token_id", None)
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id
    if eos_id is not None:
        gen_kwargs["eos_token_id"] = eos_id

    while True:
        alive = [s for s in states if not s.done]
        if not alive:
            return

        # Stop samples that hit step limits / turn limits before generating.
        for s in alive:
            if cfg.max_steps > 0 and int(s.step_idx) >= int(cfg.max_steps):
                s.done = True
                continue
            if _should_stop_by_turn_limits(
                assistant_turns=s.assistant_turns,
                user_turns=s.user_turns,
                total_messages=len(s.messages),
                max_assistant_turns=max_assistant_turns,
                max_user_turns=max_user_turns,
                max_total_messages=max_total_messages,
            ):
                s.done = True
                continue

        alive = [s for s in states if not s.done]
        if not alive:
            return

        # 1) Policy generate one step for each alive state
        batch_ids = [_encode_chat_ids(policy_tokenizer, s.messages) for s in alive]
        input_ids, attn = _pad_batch(policy_tokenizer, batch_ids)
        input_ids = input_ids.to(policy_device)
        attn = attn.to(policy_device)
        in_lens = attn.sum(dim=1).tolist()

        with torch.no_grad():
            out_ids = policy_model.generate(input_ids=input_ids, attention_mask=attn, **gen_kwargs)

        assistant_texts: list[str] = []
        for row_i, s in enumerate(alive):
            s.assistant_turns += 1
            gen = out_ids[row_i][int(in_lens[row_i]) :]
            txt = policy_tokenizer.decode(gen, skip_special_tokens=True)
            txt = str(txt or "")
            s.final_text = txt
            s.messages.append({"role": "assistant", "content": txt})
            assistant_texts.append(txt)

        # 2) Judge in batch
        judge_texts = [
            _build_causalreasoningmodel_text(
                question=str(s.problem or ""),
                prefix_steps=list(s.accepted_steps),
                candidate_step=str(assistant_texts[i] or ""),
            )
            for i, s in enumerate(alive)
        ]
        j_inputs = judge_tokenizer(
            judge_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        j_inputs = {k: v.to(judge_device) for k, v in j_inputs.items()}

        with torch.no_grad():
            j_out = judge_model(**j_inputs)
            logits = j_out.logits
            if logits.dim() == 2 and logits.size(-1) == 2:
                pred1 = torch.argmax(logits, dim=-1).tolist()
                pns_batch = [int(x) for x in pred1]
            else:
                prob1 = torch.sigmoid(logits.view(-1))
                pns_batch = [int(x) for x in (prob1 >= 0.5).float().tolist()]

        # 3) Update states
        for i, s in enumerate(alive):
            pns = int(pns_batch[i])
            s.pns_scores.append(pns)
            finished = bool(_has_answer_marker(s.final_text))
            passed = bool(pns == 1) if str(cfg.pns_mode).lower() == "discrete" else bool(
                float(pns) >= float(cfg.pns_threshold)
            )

            if verbose:
                print(
                    f"[idx={s.idx} step={s.step_idx} attempt={s.retry_for_current_step} pns={pns} passed={passed} finished={finished}]"
                )

            if passed:
                s.accepted_steps.append(s.final_text)
                s.retry_counts.append(int(s.retry_for_current_step))
                s.retry_for_current_step = 0
                s.step_idx += 1
                if finished:
                    s.done = True
                    continue

                # user follow-up
                if (not s.asked_finish) and int(force_finish_after_steps) > 0 and int(s.step_idx) >= int(
                    force_finish_after_steps
                ):
                    s.asked_finish = True
                    s.messages.append({"role": "user", "content": str(cfg.pns_ok_finish_msg)})
                else:
                    s.messages.append({"role": "user", "content": str(cfg.pns_ok_continue_msg)})
                s.user_turns += 1
            else:
                s.retry_for_current_step += 1
                s.retry_counts.append(int(s.retry_for_current_step))
                if int(s.retry_for_current_step) > int(cfg.max_retries):
                    s.accepted_steps.append(s.final_text)
                    s.retry_for_current_step = 0
                    s.step_idx += 1
                    s.messages.append({"role": "user", "content": str(cfg.pns_proceed_after_penalty_msg)})
                    s.user_turns += 1
                else:
                    s.messages.append({"role": "user", "content": str(cfg.pns_feedback_template)})
                    s.user_turns += 1

            # Safety net (message growth)
            if len(s.messages) > 2 * (cfg.max_steps * (cfg.max_retries + 2) + 16):
                s.done = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_model_path", type=str, required=True, help="HF path to your trained policy model")
    ap.add_argument(
        "--judge_model_path",
        type=str,
        default=_CAUSAL_REASONINGMODEL_NAME,
        help="Seq-classification judge model (default: MakimaSasha/CausalReasoningModel)",
    )
    ap.add_argument("--policy_device", type=str, default="cuda", help="cuda|cpu (auto-fallback if no cuda)")
    ap.add_argument("--judge_device", type=str, default="cpu", help="cuda|cpu (cpu recommended)")
    ap.add_argument("--policy_dtype", type=str, default="auto", help="auto|float16|bfloat16|float32")
    ap.add_argument("--judge_dtype", type=str, default="auto", help="auto|float16|bfloat16|float32")
    ap.add_argument("--trust_remote_code", action="store_true", help="Pass trust_remote_code=True to transformers")

    # Loop / generation
    ap.add_argument("--max_steps", type=int, default=32)
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("--step_max_new_tokens", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=1, help="Batch size for dataset inference (>=1).")
    ap.add_argument(
        "--max_assistant_turns",
        type=int,
        default=0,
        help="Max assistant generations per sample (0 means no limit).",
    )
    ap.add_argument(
        "--max_user_turns",
        type=int,
        default=0,
        help="Max injected user messages per sample (0 means no limit).",
    )
    ap.add_argument(
        "--max_total_messages",
        type=int,
        default=0,
        help="Max total chat messages per sample (0 means no limit).",
    )
    ap.add_argument("--do_sample", action="store_true", help="Enable sampling for policy generation")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument(
        "--force_finish_after_steps",
        type=int,
        default=0,
        help="If >0, after N accepted steps ask model to output full final solution ending with <ANSWER>...</ANSWER>.",
    )
    ap.add_argument("--verbose_steps", action="store_true")

    # Prompt control (so you can exactly match your training turn0 prompt)
    ap.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt used in the initial messages",
    )
    ap.add_argument(
        "--user_prefix",
        type=str,
        default="Solve the problem step by step. Provide the first step (one step only).",
        help="Extra instruction appended after the problem in the initial user message",
    )

    # Data / output
    ap.add_argument("--dataset_name", type=str, default="HuggingFaceH4/MATH-500")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--max_examples", type=int, default=0, help="If >0, only run first N examples")
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    # Lazy imports (keep module import cheap)
    import torch
    from datasets import load_dataset
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(int(args.seed))

    # Device resolution
    def resolve_device(req: str) -> str:
        r = (req or "cpu").lower()
        if r != "cpu":
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return "cpu"

    policy_device = resolve_device(args.policy_device)
    judge_device = resolve_device(args.judge_device)

    # dtype resolution
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

    # Load models
    policy_tokenizer = AutoTokenizer.from_pretrained(args.policy_model_path, trust_remote_code=bool(args.trust_remote_code))
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

    # Load dataset
    ds = load_dataset(args.dataset_name, split=args.split)
    if int(args.max_examples) > 0:
        ds = ds.select(range(min(int(args.max_examples), len(ds))))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)) or ".", exist_ok=True)

    cfg = CausalLoopCfg(max_steps=int(args.max_steps), max_retries=int(args.max_retries))

    n_total = 0
    n_correct = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        bs = max(1, int(args.batch_size))
        for base in tqdm(range(0, len(ds), bs), desc="MATH-500 inference"):
            chunk = ds.select(range(base, min(base + bs, len(ds))))

            states: list[_State] = []
            for j in range(len(chunk)):
                i = base + j
                ex = chunk[j]
                problem = str(ex.get("problem", "") or "")
                solution = str(ex.get("solution", "") or "")
                gt = _extract_gt_boxed_from_solution(solution)

                # Build initial prompt (system + user)
                user_msg = problem
                if str(args.user_prefix or "").strip():
                    user_msg = f"{problem}\n\n{str(args.user_prefix).strip()}"
                messages0 = [
                    {"role": "system", "content": str(args.system_prompt or "").strip()},
                    {"role": "user", "content": user_msg.strip()},
                ]

                states.append(
                    _State(
                        idx=int(i),
                        problem=problem,
                        gt_boxed=gt,
                        messages=messages0,
                        accepted_steps=[],
                        pns_scores=[],
                        retry_counts=[],
                        retry_for_current_step=0,
                        step_idx=0,
                        asked_finish=False,
                        assistant_turns=0,
                        user_turns=0,  # initial user is already in messages0; we count only injected follow-ups
                        final_text="",
                        done=False,
                    )
                )

            run_step_causal_loop_batch(
                states=states,
                policy_model=policy_model,
                policy_tokenizer=policy_tokenizer,
                policy_device=policy_device,
                judge_model=judge_model,
                judge_tokenizer=judge_tokenizer,
                judge_device=judge_device,
                cfg=cfg,
                step_max_new_tokens=int(args.step_max_new_tokens),
                do_sample=bool(args.do_sample),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                force_finish_after_steps=int(args.force_finish_after_steps),
                max_assistant_turns=int(args.max_assistant_turns),
                max_user_turns=int(args.max_user_turns),
                max_total_messages=int(args.max_total_messages),
                verbose=bool(args.verbose_steps),
            )

            for s in states:
                gt_norm = _normalize_answer_text(s.gt_boxed) if s.gt_boxed is not None else None
                pred_boxed = _extract_answer_from_model_text(s.final_text)
                pred_norm = _normalize_answer_text(pred_boxed) if pred_boxed is not None else None
                correct = bool(gt_norm is not None and pred_norm is not None and pred_norm == gt_norm)

                rec = {
                    "idx": int(s.idx),
                    "dataset": str(args.dataset_name),
                    "split": str(args.split),
                    "problem": s.problem,
                    "gt_boxed": s.gt_boxed,
                    "pred_boxed": pred_boxed,
                    "correct": bool(correct),
                    "final_text": s.final_text,
                    "steps": s.accepted_steps,
                    "pns_scores": s.pns_scores,
                    "retry_counts": s.retry_counts,
                    "asked_finish": bool(s.asked_finish),
                    "assistant_turns": int(s.assistant_turns),
                    "user_turns": int(s.user_turns),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                n_total += 1
                n_correct += int(correct)

    acc = (float(n_correct) / float(n_total)) if n_total else 0.0
    print(f"[DONE] total={n_total} correct={n_correct} acc={acc:.4f} out={args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


