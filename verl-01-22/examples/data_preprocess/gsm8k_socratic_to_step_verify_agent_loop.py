#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert `openai/gsm8k` (socratic config) into multi-turn parquet for `step_verify_agent`.

Key idea
--------
GSM8K Socratic stores step-by-step solution in the `answer` field (with intermediate lines like:
  ... = $<<...>>... .
and final answer like:
  #### 5

We:
1) Parse the `answer` into (a) step blocks and (b) final numeric answer
2) Call a local OpenAI-compatible LLM (vLLM) to rewrite step blocks into sequential "subproblems"
3) Emit parquet rows that match StepVerifyAgentLoop expectation:
   - agent_name: "step_verify_agent"
   - prompt: system + first user subproblem
   - extra_info.subproblems: all user subproblems (including a final synthesis request)
   - reward_model.ground_truth: final numeric answer (string)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import datasets
import requests
from datasets import Dataset


def _extract_final_numeric_answer(answer: str) -> Optional[str]:
    """
    GSM8K uses '#### <answer>' at the end. Return stripped answer with commas removed.
    """
    if not isinstance(answer, str):
        return None
    m = re.search(r"####\s*([^\n\r]+)", answer)
    if not m:
        return None
    a = m.group(1).strip()
    a = a.replace(",", "")
    return a if a else None


def _split_socratic_steps(answer: str) -> List[str]:
    """
    Split the reasoning part of GSM8K Socratic answer into step blocks.
    Heuristic: blocks separated by blank lines.
    """
    if not isinstance(answer, str):
        return []
    # Remove final answer section starting at the last '####' if present
    parts = answer.split("####")
    reasoning = parts[0] if parts else answer
    reasoning = reasoning.strip()
    if not reasoning:
        return []
    # Normalize newlines
    reasoning = reasoning.replace("\r\n", "\n").replace("\r", "\n")
    # Primary heuristic: blocks separated by blank lines
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", reasoning) if b.strip()]

    # Socratic often has no blank lines; in that case, treat each non-empty line as a step.
    # Also split on the common " ** " marker used in some GSM8K explanations.
    if len(blocks) <= 1:
        lines: List[str] = []
        for ln in reasoning.split("\n"):
            ln = ln.strip()
            if not ln:
                continue
            # split "Question ** Answer" style into separate segments
            parts = [p.strip() for p in ln.split("**") if p.strip()]
            if parts:
                lines.extend(parts)
        blocks = lines if lines else blocks

    # Filter out extremely short artifacts
    blocks = [b for b in blocks if isinstance(b, str) and b.strip() and len(b.strip()) >= 3]
    return blocks


def _truncate(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if max_chars <= 0:
        return s
    return s if len(s) <= max_chars else (s[: max_chars - 12] + " ...[truncated]")


def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None
    t = text.strip()
    # Remove common fences
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    # Try raw JSON first
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # Try to find the first {...} block
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _openai_chat_url(base_url: str) -> str:
    """
    Accept base_url like:
    - http://127.0.0.1:8000/v1
    - http://127.0.0.1:8000
    Return the /chat/completions URL.
    """
    b = (base_url or "").rstrip("/")
    if b.endswith("/v1"):
        return b + "/chat/completions"
    return b + "/v1/chat/completions"


def call_openai_compat_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_s: float,
) -> str:
    url = _openai_chat_url(base_url)
    headers = {"Content-Type": "application/json"}
    if api_key and api_key.upper() != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code} from {url}: {_truncate(r.text, 800)}")
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Unexpected response schema from {url}: {e}; body={_truncate(json.dumps(data), 800)}")


@dataclass
class ConvertCfg:
    base_url: str
    api_key: str
    model: str
    timeout_s: float = 240.0
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 768
    max_steps: int = 12
    max_chars_each_step: int = 500
    max_retries: int = 2


def llm_steps_to_subproblems(*, question: str, steps: List[str], cfg: ConvertCfg) -> Tuple[List[str], str]:
    """
    Convert step blocks into a list of user-facing subproblems (strings).
    Returns: (subproblems, raw_model_text)
    """
    steps = steps[: int(cfg.max_steps)]
    numbered_steps = "\n\n".join([f"STEP {i+1}:\n{_truncate(s, cfg.max_chars_each_step)}" for i, s in enumerate(steps)])

    system = (
        "You convert solution steps into sequential subproblems for a multi-turn math tutor.\n"
        "You must output ONLY valid JSON.\n"
    )
    user = (
        "Given the original problem and the step-by-step solution STEPS, produce an equivalent list of USER subproblems.\n"
        "Each subproblem should be short, actionable, and guide the student to derive the next step.\n"
        "Do NOT include the final numeric answer.\n"
        "Do NOT include tool calls.\n"
        "Return ONLY JSON in exactly this schema:\n"
        '{ "subproblems": ["Subproblem 1: ...", "Subproblem 2: ...", "..."] }\n\n'
        f"Original problem:\n{question}\n\n"
        f"STEPS (verbatim):\n{numbered_steps}\n"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    last_text = ""
    for attempt in range(cfg.max_retries + 1):
        try:
            text = call_openai_compat_chat(
                base_url=cfg.base_url,
                api_key=cfg.api_key,
                model=cfg.model,
                messages=messages,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_tokens=cfg.max_tokens,
                timeout_s=cfg.timeout_s,
            )
            last_text = text or ""
            obj = _extract_json_obj(last_text)
            subs = (obj or {}).get("subproblems")
            if isinstance(subs, list) and all(isinstance(x, str) and x.strip() for x in subs):
                subs = [x.strip() for x in subs][: int(cfg.max_steps)]
                return subs, last_text
        except Exception:
            if attempt >= cfg.max_retries:
                break
            time.sleep(0.5 * (attempt + 1))

    # Fallback: trivial conversion (keeps pipeline unblocked; judge will still handle correctness)
    fallback = [f"Subproblem {i+1}:\nExplain/derive the following step:\n{_truncate(s, 400)}" for i, s in enumerate(steps)]
    return fallback, last_text


def build_row(
    *,
    idx: int,
    split: str,
    question: str,
    answer: str,
    cfg: ConvertCfg,
) -> Dict[str, Any]:
    steps = _split_socratic_steps(answer)
    final_ans = _extract_final_numeric_answer(answer)

    # Convert steps -> subproblems (LLM)
    subproblems_llm, raw = llm_steps_to_subproblems(question=question, steps=steps, cfg=cfg)

    # Final synthesis step: enforce <ANSWER>...</ANSWER> for downstream answer extractor.
    final_prompt = (
        "Now, using the answers to the previous subproblems, provide the complete final solution to the original problem.\n\n"
        f"Problem:\n{question}\n\n"
        "Requirements:\n"
        "- Be concise.\n"
        "- End with the final numeric answer wrapped EXACTLY as:\n"
        "  <ANSWER>...</ANSWER>\n"
    )
    all_subproblems = list(subproblems_llm) + [final_prompt]

    system_text = (
        "You are solving a math problem via multiple subproblems.\n"
        "For each user turn, answer that subproblem correctly and briefly.\n"
        "Each subproblem solution should help progress toward the final solution.\n"
        "On the final user prompt, produce a complete final solution and clearly state the final answer.\n"
        "Be concise. No extra chit-chat.\n"
        "IMPORTANT: Answer in English only.\n"
    )

    prompt = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": all_subproblems[0] if all_subproblems else question},
    ]

    # Optional: keep expected_answers aligned with subproblems for rule-verifier usage.
    # Here we don't have per-subproblem gold answers (GSM8K gives reasoning steps),
    # so we leave them empty but keep length aligned.
    expected_answers: List[str] = [""] * len(all_subproblems)

    row = {
        "data_source": "openai/gsm8k:socratic",
        "agent_name": "step_verify_agent",
        "prompt": prompt,
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": final_ans},
        "extra_info": {
            "split": split,
            "index": idx,
            "problem_text": question,
            "source_solution": answer,
            "ground_truth": final_ans,
            "subproblems": all_subproblems,
            "expected_answers": expected_answers,
            "convert_debug": {
                "raw_subproblem_llm_output": raw,
                "num_steps_from_answer": len(steps),
                "num_subproblems_llm": len(subproblems_llm),
            },
        },
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--out_jsonl", default=None, help="Optional: write the same rows to JSONL incrementally (flushed).")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--max_samples", type=int, default=0, help="0 means all")

    # vLLM(OpenAI compat) for conversion
    ap.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--api_key", default="EMPTY")
    ap.add_argument("--timeout_s", type=float, default=240.0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=768)
    ap.add_argument("--max_steps", type=int, default=12)
    ap.add_argument("--max_chars_each_step", type=int, default=500)
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)

    args = ap.parse_args()

    cfg = ConvertCfg(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout_s=float(args.timeout_s),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
        max_steps=int(args.max_steps),
        max_chars_each_step=int(args.max_chars_each_step),
        max_retries=int(args.max_retries),
    )

    ds = datasets.load_dataset("openai/gsm8k", "socratic", split=args.split)
    if args.max_samples and args.max_samples > 0:
        ds = ds.select(range(min(len(ds), int(args.max_samples))))

    rows: List[Dict[str, Any]] = []

    def _one(i: int) -> Dict[str, Any]:
        ex = ds[int(i)]
        q = str(ex.get("question") or "")
        a = str(ex.get("answer") or "")
        return build_row(idx=int(i), split=args.split, question=q, answer=a, cfg=cfg)

    # Parallelize conversion calls (requests in threads)
    workers = max(1, int(args.workers))
    jsonl_f = None
    if args.out_jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)), exist_ok=True)
        # line-buffered for better "immediate" visibility; we also flush() after each write.
        jsonl_f = open(args.out_jsonl, "w", encoding="utf-8", buffering=1, newline="\n")
        print(f"[INFO] Writing JSONL incrementally -> {args.out_jsonl}")

    try:
        with ThreadPoolExecutor(max_workers=workers) as tp:
            futs = {tp.submit(_one, i): i for i in range(len(ds))}
            for fut in as_completed(futs):
                row = fut.result()
                rows.append(row)
                if jsonl_f is not None:
                    jsonl_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    jsonl_f.flush()
    finally:
        if jsonl_f is not None:
            jsonl_f.close()

    # Keep deterministic order by index
    rows.sort(key=lambda r: int((r.get("extra_info") or {}).get("index", 0)))

    out_ds = Dataset.from_list(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_parquet)), exist_ok=True)
    out_ds.to_parquet(args.out_parquet)
    print(f"[OK] split={args.split}, rows={len(rows)}, saved -> {args.out_parquet}")


if __name__ == "__main__":
    main()


