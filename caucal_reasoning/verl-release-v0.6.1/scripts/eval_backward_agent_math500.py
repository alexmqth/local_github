"""
Evaluate a trained backward agent on HuggingFaceH4/MATH-500 using the SAME multi-turn protocol as training.

What this script does:
1) Downloads HuggingFaceH4/MATH-500 via `datasets`.
2) Generates an evaluation JSONL dataset that is "almost the same" as training samples:
   - prompt: [{"role":"user","content": <problem>}]
   - extra_info.problem_text
   - reward_model.ground_truth
3) Runs multi-turn backward<->forward interaction:
   - backward (policy) is a REMOTE OpenAI-compatible endpoint (e.g., vLLM OpenAI server hosting your trained model).
   - forward (frozen solver) is another REMOTE OpenAI-compatible endpoint.
   - Maintains KnownClues list, pending FIFO subproblem queue, and uses the same prompt templates as training
     (imported from `verl.experimental.agent_loop.backward_agent_loop.BackwardAgentLoop` methods via a dummy instance).
4) Reports objective correctness accuracy (exact match on extracted final answer vs ground_truth).
   We still follow the SAME protocol constraints during rollout:
   - final answer: at most ONE retry opportunity (restate or split) if the first final attempt is wrong.

Usage (example):
  python scripts/eval_backward_agent_math500.py ^
    --out_jsonl outputs/math500_backward_eval.jsonl ^
    --backward_base_url http://127.0.0.1:9000/v1 --backward_model my_backward ^
    --forward_base_url http://127.0.0.1:8000/v1 --forward_model my_forward ^
    --max_samples 100 --max_rounds 24
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse


def _json_dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _extract_final_answer(text: str) -> Optional[str]:
    """Same heuristic as agent loop: <Answer>, ####, \\boxed."""
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.strip()

    m = re.findall(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", t, flags=re.IGNORECASE | re.DOTALL)
    t2 = m[-1].strip() if m else t

    m2 = re.findall(r"####\s*([^\n\r]+)", t2)
    if m2:
        cand = m2[-1].strip()
        if cand:
            return cand

    m3 = re.findall(r"\\boxed\s*\{(.*?)\}", t2, flags=re.DOTALL)
    if m3:
        cand = m3[-1].strip()
        if cand:
            return cand.rstrip(" .;，。；!")

    if m:
        cand = t2.strip()
        return cand if cand else None
    return None


def _extract_ground_truth_from_math_solution(sol: str) -> Optional[str]:
    """Best-effort: take last \\boxed{...}, else None."""
    if not isinstance(sol, str):
        return None
    m = re.findall(r"\\boxed\s*\{(.*?)\}", sol, flags=re.DOTALL)
    if m:
        cand = (m[-1] or "").strip()
        return cand.rstrip(" .;，。；!") if cand else None
    return None


async def _remote_chat_text(
    *,
    base_url: str,
    api_key: Optional[str],
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout_s: float,
) -> str:
    """OpenAI-compatible chat.completions -> assistant text."""
    import aiohttp

    class ContextOverflowError(RuntimeError):
        pass

    base_in = str(base_url or "").strip()
    if not base_in:
        raise ValueError("base_url is empty")
    # Make it robust: allow passing raw host/IP (e.g., '83.27.154.209:8000/v1') without scheme.
    parsed = urlparse(base_in)
    if not parsed.scheme:
        base_in = "http://" + base_in.lstrip("/")
    base = base_in.rstrip("/")
    url = f"{base}/v1/chat/completions" if not base.endswith("/v1") else f"{base}/chat/completions"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def make_payload(mt: int) -> dict[str, Any]:
        return {
            "model": str(model),
            "messages": messages,
            "max_tokens": int(mt),
            "temperature": float(temperature),
            "top_p": float(top_p),
        }

    def parse_context_overflow_limit(err_text: str) -> Optional[int]:
        """
        Parse common OpenAI-style context overflow message, e.g.:
        "'max_tokens' ... maximum context length is 4096 tokens and your request has 4041 input tokens
         (256 > 4096 - 4041)."
        Return the safe max_tokens upper bound, or None if cannot parse.
        """
        if not isinstance(err_text, str):
            return None
        # Try to extract context_len and input_tokens
        m = re.search(r"maximum context length is\s+(\d+)\s+tokens.*request has\s+(\d+)\s+input tokens", err_text)
        if not m:
            return None
        try:
            ctx_len = int(m.group(1))
            in_tok = int(m.group(2))
        except Exception:
            return None
        # Leave 1 token slack
        return max(0, ctx_len - in_tok - 1)

    payload = make_payload(int(max_tokens))
    timeout = aiohttp.ClientTimeout(total=float(timeout_s))
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=headers, json=payload) as r:
            raw = await r.text()
            if int(r.status) >= 400:
                # Best-effort retry once on context overflow by reducing max_tokens.
                try:
                    data = json.loads(raw)
                except Exception:
                    data = None
                msg = None
                if isinstance(data, dict) and isinstance(data.get("error"), dict):
                    msg = data["error"].get("message")
                msg_s = str(msg or raw)
                cap = parse_context_overflow_limit(msg_s)
                if int(r.status) == 400 and cap is not None:
                    if cap <= 0:
                        raise ContextOverflowError(msg_s[:500])
                    payload2 = make_payload(min(int(max_tokens), int(cap)))
                    async with session.post(url, headers=headers, json=payload2) as r2:
                        raw2 = await r2.text()
                        if int(r2.status) >= 400:
                            # If still overflow, surface as ContextOverflowError for caller to skip.
                            try:
                                data2 = json.loads(raw2)
                            except Exception:
                                data2 = None
                            msg2 = None
                            if isinstance(data2, dict) and isinstance(data2.get("error"), dict):
                                msg2 = data2["error"].get("message")
                            raise ContextOverflowError(str(msg2 or raw2)[:500])
                        raw = raw2
                else:
                    raise RuntimeError(f"HTTP {r.status}: {raw[:500]}")
    data = json.loads(raw)
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = (choices[0] or {}).get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and "text" in p:
                parts.append(str(p.get("text") or ""))
        return "".join(parts)
    return ""


def _safe_json_first_object(text: str) -> Optional[dict[str, Any]]:
    if not isinstance(text, str):
        return None
    t = text.strip()
    if not t:
        return None
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _parse_strict_json_object(text: str) -> Optional[dict[str, Any]]:
    if not isinstance(text, str):
        return None
    t = text.strip()
    if not t:
        return None
    try:
        obj = json.loads(t)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


@dataclass
class EvalConfig:
    # Backward remote (trained)
    backward_base_url: str
    backward_model: str
    backward_api_key: Optional[str] = None
    backward_max_tokens: int = 256
    backward_temperature: float = 0.0
    backward_top_p: float = 1.0
    backward_timeout_s: float = 120.0
    backward_max_concurrency: int = 8

    # Forward remote (frozen)
    forward_base_url: str = ""
    forward_model: str = ""
    forward_api_key: Optional[str] = None
    forward_max_tokens: int = 1024
    forward_temperature: float = 0.0
    forward_top_p: float = 1.0
    forward_timeout_s: float = 120.0
    forward_max_concurrency: int = 8

    # Protocol
    max_rounds: int = 24
    max_queue_items: int = 32
    max_clues: int = 10
    max_clues_chars_total: int = 2200


def _clamp_clues(clues: list[str], *, max_items: int, max_chars_total: int) -> list[str]:
    norm = [str(c).strip() for c in clues if str(c or "").strip()]
    # stable dedupe
    seen = set()
    dedup: list[str] = []
    for s in norm:
        if s in seen:
            continue
        seen.add(s)
        dedup.append(s)
    if max_items and len(dedup) > max_items:
        dedup = dedup[-max_items:]
    if max_chars_total and max_chars_total > 0:
        out: list[str] = []
        total = 0
        for s in reversed(dedup):
            add = len(s) + 2
            if out and total + add > max_chars_total:
                break
            if not out and add > max_chars_total:
                out = [s[: max_chars_total - 1]]
                break
            out.append(s)
            total += add
        dedup = list(reversed(out))
    return dedup


def _make_prompt_helpers() -> tuple[str, str, Any]:
    """
    Import prompt builders from the training agent loop to ensure test uses the same prompt templates.
    We instantiate a dummy BackwardAgentLoop object without running AgentLoopBase.__init__.
    """
    from verl.experimental.agent_loop.backward_agent_loop import BackwardAgentLoop, _BackwardAgentConfig  # type: ignore

    dummy = BackwardAgentLoop.__new__(BackwardAgentLoop)
    dummy.loop_cfg = _BackwardAgentConfig()  # default prompts
    backward_system = str(dummy.loop_cfg.backward_system_prompt)
    forward_system = str(dummy.loop_cfg.forward_system_prompt)
    return backward_system, forward_system, dummy


async def eval_one(
    *,
    cfg: EvalConfig,
    problem_text: str,
    ground_truth: Optional[str],
    sem_backward: Optional[asyncio.Semaphore],
    sem_forward: Optional[asyncio.Semaphore],
    rng: random.Random,
    return_trace: bool = False,
    trace_max_chars_per_msg: int = 8000,
    trace_max_rounds: int = 10,
) -> dict[str, Any]:
    backward_system, forward_system, dummy = _make_prompt_helpers()

    # Stateful histories (same structure as training)
    backward_messages: list[dict[str, Any]] = [{"role": "system", "content": backward_system}]
    forward_messages: list[dict[str, Any]] = [{"role": "system", "content": forward_system}]

    known_clues: list[str] = []
    asked_subproblems: list[str] = []
    forward_step_solutions: list[str] = []
    pending_queue: deque[str] = deque()

    last_forward_solution = ""
    forward_last_was_final = False
    final_checked_failed_once = False

    rounds = 0
    final_attempts = 0
    objective_correct = False
    backward_is_correct_on_final: Optional[bool] = None
    skipped_context_overflow = False

    # Initial user_state (same as training)
    user_state = dummy._build_backward_user_state(  # type: ignore[attr-defined]
        problem_text=problem_text,
        known_clues=[],
        asked_subproblems=[],
        forward_step_solutions=[],
        last_forward_solution="",
        pending_queue=deque(),
        forward_last_was_final=False,
        final_checked_failed_once=False,
    )
    backward_messages.append({"role": "user", "content": str(user_state)})

    def trunc_text(s: Any) -> str:
        maxc = int(trace_max_chars_per_msg or 8000)
        t = str(s or "")
        if maxc > 0 and len(t) > maxc:
            return t[:maxc]
        return t

    # Compact trace steps (do NOT dump full message histories; they can blow up quickly).
    trace_steps: list[dict[str, Any]] = []

    for round_idx in range(int(cfg.max_rounds)):
        rounds = round_idx + 1

        # backward decide
        async def call_backward():
            return await _remote_chat_text(
                base_url=cfg.backward_base_url,
                api_key=cfg.backward_api_key,
                model=cfg.backward_model,
                messages=backward_messages,
                max_tokens=cfg.backward_max_tokens,
                temperature=cfg.backward_temperature,
                top_p=cfg.backward_top_p,
                timeout_s=cfg.backward_timeout_s,
            )

        if sem_backward is None:
            backward_text = await call_backward()
        else:
            async with sem_backward:
                backward_text = await call_backward()

        backward_messages.append({"role": "assistant", "content": backward_text})

        strict_obj = _parse_strict_json_object(backward_text)
        action = strict_obj if strict_obj is not None else (_safe_json_first_object(backward_text) or {})

        # Apply updates
        hint_updates = action.get("hint_updates") if isinstance(action, dict) else None
        if isinstance(hint_updates, list) and hint_updates:
            known_clues.extend([str(x) for x in hint_updates if str(x or "").strip()])
            known_clues = _clamp_clues(known_clues, max_items=cfg.max_clues, max_chars_total=cfg.max_clues_chars_total)

        queue_push = action.get("queue_push") if isinstance(action, dict) else None
        if isinstance(queue_push, list) and queue_push:
            for item in queue_push:
                s = str(item or "").strip()
                if s:
                    pending_queue.append(s)
            while cfg.max_queue_items > 0 and len(pending_queue) > cfg.max_queue_items:
                pending_queue.popleft()

        is_correct = bool(action.get("is_correct", True)) if isinstance(action, dict) else True
        request_final = bool(action.get("request_final", False)) if isinstance(action, dict) else False
        restate = str(action.get("restate_subproblem") or "").strip() if isinstance(action, dict) else ""
        next_sub = str(action.get("next_subproblem") or "").strip() if isinstance(action, dict) else ""

        # Record compact per-round trace BEFORE we possibly terminate.
        if return_trace and int(round_idx) < int(trace_max_rounds):
            trace_steps.append(
                {
                    "round": int(round_idx),
                    "backward_output": trunc_text(backward_text),
                    "backward_action_strict_json": bool(strict_obj is not None),
                    # Save parsed action (may be empty if parse failed). Keep it small.
                    "backward_action": action if isinstance(action, dict) else {},
                    "known_clues": list(known_clues),
                    "pending_queue": list(pending_queue),
                    # For debugging planning quality (what controller will do next), but keep minimal.
                    "request_final": bool(request_final),
                    "next_subproblem": trunc_text(next_sub),
                    "restate_subproblem": trunc_text(restate),
                    "queue_push_n": int(len(queue_push)) if isinstance(queue_push, list) else 0,
                }
            )

        # FINAL_VERIFICATION termination rules for evaluation:
        # - We do NOT compute "final_reward" here; we only care about objective correctness.
        # - If the final answer is objectively correct, we terminate immediately.
        # - If objectively wrong:
        #     - if backward says correct: terminate immediately (counts as wrong)
        #     - if backward says wrong: allow at most ONE retry (restate or split); second wrong terminates.
        if forward_last_was_final:
            extracted = _extract_final_answer(last_forward_solution) or ""
            objective_ok = (
                (str(extracted).strip() == str(ground_truth).strip())
                if (ground_truth is not None)
                else bool(str(extracted).strip())
            )
            final_attempts += 1
            backward_is_correct_on_final = bool(is_correct)

            if objective_ok:
                objective_correct = True
                break

            # objectively wrong
            if bool(is_correct):
                # backward mistakenly accepts a wrong final answer => stop (no retry)
                break

            # backward also says wrong => allow at most one retry
            if final_checked_failed_once:
                break
            final_checked_failed_once = True

        # Decide forward request (queue priority)
        if restate:
            forward_request = restate
            forward_last_was_final = False
        elif pending_queue:
            forward_request = pending_queue.popleft()
            forward_last_was_final = False
        elif request_final:
            forward_request = (
                "Now provide the COMPLETE final solution to the original problem and state the final answer.\n"
                "Output the final answer STRICTLY as <Answer>...</Answer>."
            )
            forward_last_was_final = True
        elif next_sub:
            forward_request = next_sub
            forward_last_was_final = False
        else:
            forward_request = "Propose the next smallest helpful subproblem to solve."
            forward_last_was_final = False

        asked_subproblems.append(forward_request)

        # forward call
        forward_user = dummy._build_forward_user_prompt(  # type: ignore[attr-defined]
            problem_text=problem_text,
            known_clues=known_clues,
            request=forward_request,
        )
        forward_messages.append({"role": "user", "content": str(forward_user)})

        async def call_forward():
            return await _remote_chat_text(
                base_url=cfg.forward_base_url,
                api_key=cfg.forward_api_key,
                model=cfg.forward_model,
                messages=forward_messages,
                max_tokens=cfg.forward_max_tokens,
                temperature=cfg.forward_temperature,
                top_p=cfg.forward_top_p,
                timeout_s=cfg.forward_timeout_s,
            )

        try:
            if sem_forward is None:
                forward_text = await call_forward()
            else:
                async with sem_forward:
                    forward_text = await call_forward()
        except Exception as e:
            # If forward context overflows even after auto-retry, skip this sample to keep evaluation running.
            if e.__class__.__name__ == "ContextOverflowError" or "maximum context length" in str(e):
                skipped_context_overflow = True
                break
            raise

        last_forward_solution = str(forward_text or "")
        forward_messages.append({"role": "assistant", "content": last_forward_solution})
        forward_step_solutions.append(last_forward_solution)

        # Attach forward I/O to the most recent trace step if we recorded it.
        if return_trace and trace_steps and int(trace_steps[-1].get("round", -1)) == int(round_idx):
            trace_steps[-1]["forward_request_used"] = trunc_text(forward_request)
            trace_steps[-1]["forward_step_solution"] = trunc_text(last_forward_solution)

        # append next backward user_state (like training)
        user_state = dummy._build_backward_user_state(  # type: ignore[attr-defined]
            problem_text=problem_text,
            known_clues=known_clues,
            asked_subproblems=asked_subproblems,
            forward_step_solutions=forward_step_solutions,
            last_forward_solution=last_forward_solution,
            pending_queue=pending_queue,
            forward_last_was_final=forward_last_was_final,
            final_checked_failed_once=final_checked_failed_once,
        )
        backward_messages.append({"role": "user", "content": str(user_state)})

    return {
        "objective_correct": bool(objective_correct),
        "rounds": int(rounds),
        "ground_truth": ground_truth,
        "final_extracted": _extract_final_answer(last_forward_solution),
        "forward_last_was_final": bool(forward_last_was_final),
        "final_attempts": int(final_attempts),
        "backward_is_correct_on_final": backward_is_correct_on_final,
        "skipped_context_overflow": bool(skipped_context_overflow),
        **(
            {
                "trace": {
                    "problem_text": problem_text,
                    "ground_truth": ground_truth,
                    "steps": trace_steps,
                }
            }
            if return_trace
            else {}
        ),
    }


def build_eval_jsonl(*, out_jsonl: str, split: str = "test", max_samples: Optional[int] = None) -> int:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split=split)
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    n = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            if max_samples is not None and n >= int(max_samples):
                break
            # Common keys in MATH datasets: problem/solution
            problem = ex.get("problem") or ex.get("question") or ex.get("prompt") or ""
            solution = ex.get("solution") or ex.get("answer") or ""
            gt = ex.get("final_answer") or ex.get("ground_truth") or _extract_ground_truth_from_math_solution(str(solution))
            rec = {
                "prompt": [{"role": "user", "content": str(problem)}],
                "extra_info": {
                    "problem_text": str(problem),
                    "ground_truth": str(gt) if gt is not None else None,
                    "source_dataset": "HuggingFaceH4/MATH-500",
                    "source_split": str(split),
                    "source_index": int(i),
                },
                "reward_model": {"ground_truth": str(gt) if gt is not None else None},
            }
            f.write(_json_dump(rec) + "\n")
            n += 1
    return n


async def run_eval_from_jsonl(
    *,
    jsonl_path: str,
    cfg: EvalConfig,
    max_samples: Optional[int] = None,
    seed: int = 1,
    trace_jsonl: Optional[str] = None,
    trace_max_chars_per_msg: int = 8000,
    trace_max_rounds: int = 10,
) -> dict[str, Any]:
    rng = random.Random(int(seed))
    sem_b = asyncio.Semaphore(int(cfg.backward_max_concurrency)) if cfg.backward_max_concurrency > 0 else None
    sem_f = asyncio.Semaphore(int(cfg.forward_max_concurrency)) if cfg.forward_max_concurrency > 0 else None

    results: list[dict[str, Any]] = []
    trace_f = None
    if trace_jsonl:
        os.makedirs(os.path.dirname(trace_jsonl) or ".", exist_ok=True)
        trace_f = open(trace_jsonl, "w", encoding="utf-8")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        try:
            for line_idx, line in enumerate(f):
                if max_samples is not None and len(results) >= int(max_samples):
                    break
                obj = json.loads(line)
                problem_text = (
                    ((obj.get("extra_info") or {}).get("problem_text"))
                    or ((obj.get("prompt") or [{}])[-1].get("content"))
                    or ""
                )
                gt = ((obj.get("reward_model") or {}).get("ground_truth")) or ((obj.get("extra_info") or {}).get("ground_truth"))
                r = await eval_one(
                    cfg=cfg,
                    problem_text=str(problem_text),
                    ground_truth=str(gt) if gt is not None else None,
                    sem_backward=sem_b,
                    sem_forward=sem_f,
                    rng=rng,
                    return_trace=bool(trace_f is not None),
                    trace_max_chars_per_msg=int(trace_max_chars_per_msg),
                    trace_max_rounds=int(trace_max_rounds),
                )
                r["index"] = int(line_idx)
                results.append(r)
                if trace_f is not None:
                    trace_f.write(_json_dump({"index": int(line_idx), **r.get("trace", {})}) + "\n")
        finally:
            if trace_f is not None:
                trace_f.close()

    # Metrics (exclude skipped samples from accuracy denominator)
    skipped = sum(1 for r in results if bool(r.get("skipped_context_overflow", False)))
    valid = max(1, len(results) - skipped)
    acc = sum(1 for r in results if (not bool(r.get("skipped_context_overflow", False))) and bool(r.get("objective_correct", False))) / float(valid)
    # Agreement stats (optional diagnostics)
    correct_and_backward_correct = sum(
        1
        for r in results
        if bool(r.get("objective_correct", False)) and (r.get("backward_is_correct_on_final") is True)
    )
    correct_but_backward_incorrect = sum(
        1
        for r in results
        if bool(r.get("objective_correct", False)) and (r.get("backward_is_correct_on_final") is False)
    )
    wrong_but_backward_correct = sum(
        1
        for r in results
        if (not bool(r.get("objective_correct", False))) and (r.get("backward_is_correct_on_final") is True)
    )
    avg_rounds = sum(int(r["rounds"]) for r in results) / max(1, len(results))
    avg_final_attempts = sum(int(r.get("final_attempts", 0) or 0) for r in results) / max(1, len(results))
    return {
        "n": len(results),
        "skipped_context_overflow": int(skipped),
        "valid_n": int(len(results) - skipped),
        "accuracy": acc,
        "correct_and_backward_correct": int(correct_and_backward_correct),
        "correct_but_backward_incorrect": int(correct_but_backward_incorrect),
        "wrong_but_backward_correct": int(wrong_but_backward_correct),
        "avg_rounds": avg_rounds,
        "avg_final_attempts": avg_final_attempts,
        "results": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_jsonl", type=str, default="outputs/math500_backward_eval.jsonl")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--build_only", action="store_true")
    ap.add_argument("--max_samples", type=int, default=200)

    ap.add_argument("--backward_base_url", type=str, required=True)
    ap.add_argument("--backward_model", type=str, required=True)
    ap.add_argument("--backward_api_key", type=str, default=None)
    ap.add_argument("--backward_max_tokens", type=int, default=256)
    ap.add_argument("--backward_temperature", type=float, default=0.0)
    ap.add_argument("--backward_top_p", type=float, default=1.0)
    ap.add_argument("--backward_timeout_s", type=float, default=120.0)
    ap.add_argument("--backward_max_concurrency", type=int, default=8)

    ap.add_argument("--forward_base_url", type=str, required=True)
    ap.add_argument("--forward_model", type=str, required=True)
    ap.add_argument("--forward_api_key", type=str, default=None)
    ap.add_argument("--forward_max_tokens", type=int, default=1024)
    ap.add_argument("--forward_temperature", type=float, default=0.0)
    ap.add_argument("--forward_top_p", type=float, default=1.0)
    ap.add_argument("--forward_timeout_s", type=float, default=120.0)
    ap.add_argument("--forward_max_concurrency", type=int, default=8)

    ap.add_argument("--max_rounds", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--trace_jsonl", type=str, default="")
    ap.add_argument("--trace_max_chars_per_msg", type=int, default=8000)
    ap.add_argument("--trace_max_rounds", type=int, default=10)
    args = ap.parse_args()

    n = build_eval_jsonl(out_jsonl=args.out_jsonl, split=args.split, max_samples=args.max_samples)
    print(f"[eval] wrote {n} samples to {args.out_jsonl}")
    if args.build_only:
        return

    cfg = EvalConfig(
        backward_base_url=args.backward_base_url,
        backward_model=args.backward_model,
        backward_api_key=args.backward_api_key,
        backward_max_tokens=args.backward_max_tokens,
        backward_temperature=args.backward_temperature,
        backward_top_p=args.backward_top_p,
        backward_timeout_s=args.backward_timeout_s,
        backward_max_concurrency=args.backward_max_concurrency,
        forward_base_url=args.forward_base_url,
        forward_model=args.forward_model,
        forward_api_key=args.forward_api_key,
        forward_max_tokens=args.forward_max_tokens,
        forward_temperature=args.forward_temperature,
        forward_top_p=args.forward_top_p,
        forward_timeout_s=args.forward_timeout_s,
        forward_max_concurrency=args.forward_max_concurrency,
        max_rounds=args.max_rounds,
    )

    summary = asyncio.run(
        run_eval_from_jsonl(
            jsonl_path=args.out_jsonl,
            cfg=cfg,
            max_samples=args.max_samples,
            seed=args.seed,
            trace_jsonl=(args.trace_jsonl or None),
            trace_max_chars_per_msg=int(args.trace_max_chars_per_msg),
            trace_max_rounds=int(args.trace_max_rounds),
        )
    )
    print("[eval] summary:", _json_dump({k: v for k, v in summary.items() if k != "results"}))


if __name__ == "__main__":
    main()


