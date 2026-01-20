#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM-assisted near-duplicate removal for `extra_info.subproblems` in an AgentLoop parquet.

What it does
------------
- Reads an input parquet (HF datasets parquet).
- For each row:
  - Looks for `extra_info.subproblems` (list[str]).
  - Optionally aligns and prunes `extra_info.expected_answers` (list[str]) with the same indices.
  - Calls an OpenAI-compatible chat completion endpoint (vLLM / SGLang / OpenAI) to decide which indices to KEEP.
  - Rewrites only those lists; all other fields are preserved as-is.
- Writes a new parquet.

Why LLM
-------
Some "near duplicates" are paraphrases, so simple string dedupe is insufficient.

Safety / invariants
-------------------
- Preserves order.
- Always keeps index 0 (the entry prompt) and the last item (often the final synthesis step) if present.
- If the model output cannot be parsed, falls back to a conservative heuristic (keep all).
- Does NOT add new keys to rows (only rewrites existing subproblems/expected_answers lists).

Example
-------
python examples/data_preprocess/dedupe_subproblems_with_llm.py ^
  --in_parquet in.parquet ^
  --out_parquet out_dedup.parquet ^
  --base_url http://127.0.0.1:8000/v1 ^
  --model Qwen/Qwen2.5-7B-Instruct ^
  --api_key EMPTY
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from datasets import Dataset, load_dataset


def _truncate(s: str, max_chars: int) -> str:
    s = s or ""
    s = s.strip()
    return s if len(s) <= max_chars else s[: max_chars - 12] + " ...[truncated]"


def _format_numbered(items: List[str], max_chars_each: int) -> str:
    lines = []
    for i, it in enumerate(items):
        lines.append(f"{i}: {_truncate(it, max_chars_each)}")
    return "\n".join(lines)


def _extract_json_array(text: str) -> Optional[List[int]]:
    """
    Try to parse a JSON array of ints from the model output.
    Accepts either:
      - a full JSON array
      - a JSON object containing {"keep_indices":[...]} or {"keep":[...]}
    """
    if not isinstance(text, str):
        return None
    t = text.strip()

    # Try raw JSON
    for candidate in [t]:
        try:
            obj = json.loads(candidate)
        except Exception:
            obj = None
        if isinstance(obj, list) and all(isinstance(x, int) for x in obj):
            return obj
        if isinstance(obj, dict):
            for key in ("keep_indices", "keep", "indices"):
                v = obj.get(key)
                if isinstance(v, list) and all(isinstance(x, int) for x in v):
                    return v

    # Try to find the first [...] block
    m = re.search(r"\[[\s\S]*?\]", t)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, list) and all(isinstance(x, int) for x in obj):
                return obj
        except Exception:
            pass

    # Last resort: extract integers
    nums = re.findall(r"\b\d+\b", t)
    if nums:
        try:
            return [int(x) for x in nums]
        except Exception:
            return None
    return None


def _conservative_keep_all(n: int) -> List[int]:
    return list(range(n))


def call_openai_compat_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout_s: int = 120,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key and api_key.upper() != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    if not r.ok:
        # include response body for easier debugging
        raise RuntimeError(f"HTTP {r.status_code} from {url}: {_truncate(r.text, 800)}")
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Unexpected response schema from {url}: {e}; body={_truncate(json.dumps(data), 800)}")


def llm_dedupe_indices(
    *,
    subproblems: List[str],
    base_url: str,
    api_key: str,
    model: str,
    max_chars_each: int,
    max_tokens: int,
    temperature: float,
) -> Tuple[List[int], str, Optional[str], Optional[str]]:
    n = len(subproblems)
    if n <= 2:
        return _conservative_keep_all(n), "skip_short", None, None

    system = (
        "You are a data-cleaning assistant.\n"
        "Task: remove near-duplicate steps from a numbered list of 'subproblems' while preserving order.\n"
        "Rules:\n"
        "- KEEP index 0.\n"
        "- KEEP the last index.\n"
        "- Remove items that are essentially the same step (paraphrase / repeated substitution / same algebra).\n"
        "- If two items overlap heavily, keep the earlier one.\n"
        "- Be proactive: if there is ANY near-duplicate pair, you MUST remove at least one item.\n"
        "- Do NOT keep all indices unless every item is clearly distinct and adds new information.\n"
        "- Output ONLY JSON, either:\n"
        "  [0, 2, 3, ...]\n"
        "or:\n"
        '  {"keep_indices":[0,2,3,...]}\n'
    )
    user = (
        "Here are the subproblems:\n\n"
        f"{_format_numbered(subproblems, max_chars_each)}\n\n"
        "Return keep indices."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    try:
        out = call_openai_compat_chat(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        # If request fails, be conservative
        return _conservative_keep_all(n), "request_error", None, str(e)

    keep = _extract_json_array(out)
    if keep is None:
        return _conservative_keep_all(n), "parse_error", out, None

    # sanitize
    keep = [int(i) for i in keep if 0 <= int(i) < n]
    keep = sorted(set(keep))
    if 0 not in keep:
        keep = [0] + keep
    if (n - 1) not in keep:
        keep = keep + [n - 1]
    keep = sorted(set(keep))
    return keep, "ok", out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_parquet", required=True)
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--base_url", required=True, help="OpenAI-compatible base url, e.g. http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api_key", default="EMPTY")
    ap.add_argument("--max_chars_each", type=int, default=600)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--sleep_s", type=float, default=0.0, help="Optional sleep between calls to avoid rate limits")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only process first N rows (debug)")
    ap.add_argument(
        "--debug_first_n",
        type=int,
        default=0,
        help="If >0, print LLM outputs/parsing for the first N processed rows (for debugging).",
    )
    ap.add_argument(
        "--fail_on_error",
        action="store_true",
        help="If set, abort on the first LLM request/parse error instead of silently keeping all.",
    )
    args = ap.parse_args()

    in_path = os.path.abspath(os.path.expanduser(args.in_parquet))
    out_path = os.path.abspath(os.path.expanduser(args.out_parquet))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ds = load_dataset("parquet", data_files=in_path, split="train")
    rows = [ds[i] for i in range(len(ds))]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    out_rows: List[Dict[str, Any]] = []
    total_removed = 0
    total_before = 0
    llm_calls = 0
    llm_ok = 0
    llm_request_errors = 0
    llm_parse_errors = 0
    kept_all = 0
    for idx, row in enumerate(rows):
        extra = row.get("extra_info") or {}
        subproblems = extra.get("subproblems")
        if not isinstance(subproblems, list) or not all(isinstance(x, str) for x in subproblems):
            out_rows.append(row)
            continue

        keep, status, raw_out, err = llm_dedupe_indices(
            subproblems=subproblems,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            max_chars_each=args.max_chars_each,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        llm_calls += 1
        if status == "ok":
            llm_ok += 1
        elif status == "request_error":
            llm_request_errors += 1
            if args.fail_on_error:
                raise RuntimeError(f"LLM request error on row {idx}: {err}")
        elif status == "parse_error":
            llm_parse_errors += 1
            if args.fail_on_error:
                raise RuntimeError(f"LLM parse error on row {idx}: raw_out={raw_out}")
        if len(keep) == len(subproblems):
            kept_all += 1

        if args.debug_first_n and idx < args.debug_first_n:
            print("========== DEBUG ==========")
            print(f"row={idx} status={status} n_before={len(subproblems)} n_keep={len(keep)}")
            if err:
                print(f"error: {err}")
            if raw_out:
                print(f"raw_llm_output:\n{raw_out}")
            print(f"keep_indices={keep}")

        new_sub = [subproblems[i] for i in keep]

        # Align expected_answers if present
        expected = extra.get("expected_answers")
        if isinstance(expected, list):
            if len(expected) == len(subproblems):
                new_expected = [expected[i] for i in keep]
            else:
                # if not aligned, keep as-is (do not risk breaking downstream)
                new_expected = expected
        else:
            new_expected = None

        # rewrite only the two lists; keep everything else identical
        new_row = dict(row)
        new_extra = dict(extra)
        new_extra["subproblems"] = new_sub
        if new_expected is not None:
            new_extra["expected_answers"] = new_expected
        new_row["extra_info"] = new_extra

        out_rows.append(new_row)

        total_before += len(subproblems)
        total_removed += len(subproblems) - len(new_sub)

        if args.sleep_s and args.sleep_s > 0:
            time.sleep(args.sleep_s)

        if (idx + 1) % 50 == 0:
            print(
                f"[progress] {idx+1}/{len(rows)} rows, removed_steps={total_removed} | "
                f"llm_calls={llm_calls} ok={llm_ok} req_err={llm_request_errors} parse_err={llm_parse_errors} "
                f"kept_all={kept_all}"
            )

    print(
        f"[done] rows={len(out_rows)}, steps_before={total_before}, steps_removed={total_removed} | "
        f"llm_calls={llm_calls} ok={llm_ok} req_err={llm_request_errors} parse_err={llm_parse_errors} "
        f"kept_all={kept_all}"
    )
    out_ds = Dataset.from_list(out_rows)
    out_ds.to_parquet(out_path)
    print(f"[OK] saved -> {out_path}")


if __name__ == "__main__":
    main()


