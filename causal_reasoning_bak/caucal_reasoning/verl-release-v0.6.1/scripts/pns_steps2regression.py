#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a single-turn regression dataset from PNS-style multi-step JSONL.

Goal
----
For each original sample (question + multi-step CoT + per-step PNS),
select only the steps that participated in PNS evaluation (pns_eval_mask==1),
and emit one training row per selected step.

Each output row contains:
- question
- prev_steps (all steps before current step)
- current_step
- text (question + prev_steps + current_step, for DeBERTa-style encoders)
- label (the per_step_pns value for the current step)

This script does NOT call any LLM.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from typing import Any, Iterable, Optional


_HASH_ANSWER_RE = re.compile(r"^\s*####\s*", flags=re.IGNORECASE)


def _iter_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _as_list(x: Any) -> list[Any]:
    if isinstance(x, list):
        return x
    return []


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _is_eval(mask_val: Any) -> bool:
    # pns_eval_mask may be 0/1 ints, bools, or strings.
    if mask_val is True:
        return True
    if mask_val is False or mask_val is None:
        return False
    try:
        return int(mask_val) == 1
    except Exception:
        return False


def _normalize_steps(steps: list[Any]) -> list[str]:
    out: list[str] = []
    for s in steps:
        ss = str(s).strip()
        if ss:
            out.append(ss)
    return out


def _drop_hash_answer_steps(steps: list[str]) -> list[str]:
    return [s for s in steps if s and not _HASH_ANSWER_RE.match(s)]


def _build_text(
    *,
    question: str,
    prev_steps: list[str],
    current_step: str,
    prev_steps_joiner: str,
    include_prev_steps_header: bool,
) -> str:
    q = (question or "").strip()
    cur = (current_step or "").strip()
    prev = [s.strip() for s in prev_steps if s and str(s).strip()]
    prev_txt = prev_steps_joiner.join(prev).strip()

    if include_prev_steps_header:
        prev_block = prev_txt if prev_txt else "(none)"
        return (
            f"Problem:\n{q}\n\n"
            f"Previous steps:\n{prev_block}\n\n"
            f"Current step:\n{cur}"
        ).strip()

    # No header: just concatenate.
    parts: list[str] = [f"Problem:\n{q}"]
    if prev_txt:
        parts.append(prev_txt)
    parts.append(cur)
    return "\n\n".join(parts).strip()


def build_rows_for_obj(
    obj: dict[str, Any],
    *,
    text_field: str,
    label_field: str,
    drop_hash_answer_step: bool,
    prev_steps_joiner: str,
    include_prev_steps_header: bool,
    passthrough_unknown_fields: bool,
) -> list[dict[str, Any]]:
    question = str(obj.get("question") or "").strip()
    if not question:
        return []

    steps = _normalize_steps(_as_list(obj.get("orig_steps")))
    per_step_pns = _as_list(obj.get("per_step_pns"))
    pns_eval_mask = _as_list(obj.get("pns_eval_mask"))
    pns_eval_indices = _as_list(obj.get("pns_eval_indices"))

    if not steps or not per_step_pns or not pns_eval_mask:
        return []

    n = min(len(steps), len(per_step_pns), len(pns_eval_mask))
    if n <= 0:
        return []

    steps = steps[:n]
    per_step_pns = per_step_pns[:n]
    pns_eval_mask = pns_eval_mask[:n]

    rows: list[dict[str, Any]] = []

    # Known fields we already map into fixed columns.
    known_top = {
        "question",
        "gold_answer",
        "type",
        "level",
        "cot_original",
        "cot_final",
        "orig_steps",
        "kept_steps",
        "ps",
        "per_step_pns",
        "pns_eval_indices",
        "pns_eval_mask",
        "rollout_backend",
        "k",
        "alpha",
        "pns_eval_max_steps",
        "pns_eval_strategy",
        "pns_exclude_last_n",
        "pns_allow_answer_steps",
    }

    eval_set: Optional[set[int]] = None
    if pns_eval_indices:
        try:
            eval_set = {int(i) for i in pns_eval_indices}
        except Exception:
            eval_set = None

    for i in range(n):
        if not _is_eval(pns_eval_mask[i]):
            continue
        y = _safe_float(per_step_pns[i])
        if y is None:
            continue

        cur = steps[i]
        prev = steps[:i]

        if drop_hash_answer_step:
            if _HASH_ANSWER_RE.match(cur):
                continue
            prev = _drop_hash_answer_steps(prev)

        text = _build_text(
            question=question,
            prev_steps=prev,
            current_step=cur,
            prev_steps_joiner=prev_steps_joiner,
            include_prev_steps_header=include_prev_steps_header,
        )

        # Binary label from pns_eval_indices (membership). If indices are missing/broken, fall back to mask.
        binary_label = int((i in eval_set) if eval_set is not None else _is_eval(pns_eval_mask[i]))

        row: dict[str, Any] = {
            text_field: text,
            label_field: y,
            "binary_label": binary_label,
            "question": question,
            "prev_steps": prev,
            "current_step": cur,
            "step_index": i,
            "num_steps": n,
            # useful metadata passthrough (common)
            "gold_answer": obj.get("gold_answer", ""),
            "type": obj.get("type", ""),
            "level": obj.get("level", ""),
            "ps": obj.get("ps", None),
            "rollout_backend": obj.get("rollout_backend", None),
            "k": obj.get("k", None),
            "alpha": obj.get("alpha", None),
        }

        if passthrough_unknown_fields:
            extra: dict[str, Any] = {}
            for k, v in obj.items():
                if k in known_top:
                    continue
                if k in row:
                    continue
                extra[k] = v
            if extra:
                row["meta"] = extra

        rows.append(row)

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, help="Input PNS JSONL (multi-step).")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL (single-step regression).")
    ap.add_argument("--out_parquet", default="", help="Optional parquet output path (built from JSONL at the end).")
    ap.add_argument("--max_samples", type=int, default=0, help="0 means all input lines.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--text_field",
        default="text",
        help="Field name for the model input text (HF datasets often use 'text').",
    )
    ap.add_argument(
        "--label_field",
        default="label",
        help="Field name for regression label (HF Trainer defaults: 'labels', but many use 'label').",
    )
    ap.add_argument(
        "--drop_hash_answer_step",
        action="store_true",
        help="Drop GSM8K-style '#### <answer>' steps (both as current step and from prev_steps context).",
    )
    ap.add_argument(
        "--prev_steps_joiner",
        default="\n",
        help="Joiner used to merge prev_steps into a text block (default: newline).",
    )
    ap.add_argument(
        "--include_prev_steps_header",
        action="store_true",
        help="If set, text becomes a 3-block template with headers: Problem/Previous steps/Current step.",
    )
    ap.add_argument(
        "--passthrough_unknown_fields",
        action="store_true",
        help="If set, unknown top-level fields are copied into row['meta'] (e.g., ds_index/shard_id/...).",
    )
    args = ap.parse_args()

    random.seed(int(args.seed))

    in_path = str(args.in_jsonl)
    out_path = str(args.out_jsonl)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    n_in = 0
    n_obj_used = 0
    n_rows = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for obj in _iter_jsonl(in_path):
            n_in += 1
            if int(args.max_samples) > 0 and n_in > int(args.max_samples):
                break

            rows = build_rows_for_obj(
                obj,
                text_field=str(args.text_field),
                label_field=str(args.label_field),
                drop_hash_answer_step=bool(args.drop_hash_answer_step),
                prev_steps_joiner=str(args.prev_steps_joiner),
                include_prev_steps_header=bool(args.include_prev_steps_header),
                passthrough_unknown_fields=bool(args.passthrough_unknown_fields),
            )
            if not rows:
                continue

            n_obj_used += 1
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_rows += 1

    if args.out_parquet:
        try:
            import pandas as pd  # type: ignore[import-not-found]
        except Exception as e:
            raise SystemExit(
                "Failed to import pandas for --out_parquet. Install it first (e.g., `pip install pandas pyarrow`)."
            ) from e

        out_pq = str(args.out_parquet)
        os.makedirs(os.path.dirname(out_pq) or ".", exist_ok=True)
        df = pd.read_json(out_path, lines=True)
        df.to_parquet(out_pq, index=False)

    print(
        json.dumps(
            {
                "in_jsonl": in_path,
                "out_jsonl": out_path,
                "out_parquet": (str(args.out_parquet) if args.out_parquet else None),
                "read_lines": n_in,
                "used_objects": n_obj_used,
                "written_rows": n_rows,
                "text_field": str(args.text_field),
                "label_field": str(args.label_field),
                "drop_hash_answer_step": bool(args.drop_hash_answer_step),
                "include_prev_steps_header": bool(args.include_prev_steps_header),
                "passthrough_unknown_fields": bool(args.passthrough_unknown_fields),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


