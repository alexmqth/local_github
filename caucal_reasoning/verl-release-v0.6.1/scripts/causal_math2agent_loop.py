#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert `causual_math_gen.py` output JSONL (PNS-style) into AgentLoop multi-turn JSONL
compatible with `StepVerifyAgentLoop`.

Input (per line): output of `examples/data_preprocess/causual_math_gen.py`, e.g.:
  {
    "question": ...,
    "gold_answer": ...,
    "cot_original": ...,
    "orig_steps": [...],
    "kept_steps": [...],
    "cot_final": ...,
    ...
  }

Output (per line): AgentLoop row:
  {
    "agent_name": "step_verify_agent",
    "prompt": [{"role":"system","content":...}, {"role":"user","content": subproblems[0]}],
    "reward_model": {"style":"rule","ground_truth": gold_answer},
    "extra_info": {
      "problem_text": question,
      "subproblems": [...],          # multi-turn user inputs
      "expected_answers": [...],     # optional; aligned with subproblems
      ... (metadata passthrough)
    }
  }

Notes
-----
- This converter does NOT call any LLM. It uses `kept_steps`/`orig_steps` as reference
  step answers and wraps them into "subproblem -> brief answer" turns.
- It appends a final synthesis user turn to force a final answer, and sets the final
  expected answer to end with `<ANSWER>...</ANSWER>` so final reward extraction works.
- For GSM8K-style data, `orig_steps/kept_steps` sometimes contain a trailing `#### <ans>`
  line. Use `--drop_hash_answer_step` to drop those from subproblems.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from typing import Any, Dict, Iterable, List, Optional


FINAL_SYNTHESIS_USER_PROMPT_EN = (
    "Now, using the answers to the previous subproblems, provide the complete final solution.\n"
    "Requirement: the last line MUST be <ANSWER>...</ANSWER> containing the final answer."
)


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


_ANSWER_BLOCK_RE = re.compile(r"<\s*ANSWER\s*>(.*?)<\s*/\s*ANSWER\s*>", flags=re.IGNORECASE | re.DOTALL)


def _ensure_answer_tag(text: str, answer: str) -> str:
    t = str(text or "").strip()
    # Strip existing answer tags and re-append one clean final line.
    t = _ANSWER_BLOCK_RE.sub("", t).strip()
    a = str(answer or "").strip()
    if t:
        return t + "\n\n" + f"<ANSWER>{a}</ANSWER>"
    return f"<ANSWER>{a}</ANSWER>"


def _pick_steps(obj: dict[str, Any], *, use_kept_steps: bool) -> list[str]:
    if use_kept_steps:
        steps = obj.get("kept_steps")
    else:
        steps = obj.get("orig_steps")
    if isinstance(steps, list) and steps:
        return [str(x).strip() for x in steps if str(x).strip()]
    # fallback: split cot_final by lines
    cot_final = str(obj.get("cot_final") or "").strip()
    if cot_final:
        return [ln.strip() for ln in cot_final.splitlines() if ln.strip()]
    # last resort: empty
    return []


_GSM8K_HASH_ANSWER_RE = re.compile(r"^\s*####\s*", flags=re.IGNORECASE)


def _maybe_drop_hash_answer_steps(steps: list[str], *, drop_hash_answer_step: bool) -> list[str]:
    """Drop GSM8K '#### <answer>' marker steps (usually the last line)."""
    if not drop_hash_answer_step:
        return steps
    out: list[str] = []
    for s in steps:
        ss = str(s).strip()
        if not ss:
            continue
        if _GSM8K_HASH_ANSWER_RE.match(ss):
            continue
        out.append(ss)
    return out


# Known top-level fields in the PNS-style JSON.
# Used by passthrough_unknown_fields to avoid duplicating canonical fields.
_KNOWN_TOP_FIELDS = {
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
    "pns_records",
    "kept_mask",
    "eval_orig_idxs",
    "pns_eval_indices",
    "pns_eval_mask",
    "pns_eval_max_steps",
    "pns_eval_strategy",
    "pns_exclude_last_n",
    "pns_allow_answer_steps",
    "rollout_backend",
    "k",
    "alpha",
    "gpu_id",
}


def build_agentloop_row(
    obj: dict[str, Any],
    *,
    system_prompt: str,
    final_synthesis_user_prompt: str,
    use_kept_steps: bool,
    include_problem_each_turn: bool,
    data_source: str,
    mode: str,
    drop_hash_answer_step: bool,
    passthrough_unknown_fields: bool,
) -> Optional[dict[str, Any]]:
    question = str(obj.get("question") or "").strip()
    gold = str(obj.get("gold_answer") or "").strip()
    if not question or not gold:
        return None

    mode = str(mode or "interactive").lower()
    if mode not in {"interactive", "subproblems"}:
        mode = "interactive"

    if mode == "interactive":
        # Minimal entry-only prompt. All intermediate user turns will be generated by the agent loop dynamically.
        user0 = (
            "Solve the following problem step by step.\n"
            "Important constraints:\n"
            "- In each assistant turn, write ONLY one coherent reasoning step (1-3 sentences).\n"
            "- Do NOT jump ahead or write multiple steps at once.\n"
            "- Do NOT provide the final answer until you are ready to finish.\n"
            "- When you finish, provide a complete final solution and end with a final line exactly:\n"
            "  <ANSWER>...</ANSWER>\n"
            f"\nProblem:\n{question}"
        )
        prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user0}]
        row = {
            "data_source": data_source,
            "agent_name": "step_verify_agent",  # caller may override later
            "prompt": prompt,
            "reward_model": {"style": "rule", "ground_truth": gold},
            "extra_info": {
                "problem_text": question,
                "gold_answer": gold,
                "type": obj.get("type", ""),
                "level": obj.get("level", ""),
                # passthrough
                "cot_original": obj.get("cot_original"),
                "cot_final": obj.get("cot_final"),
                "orig_steps": obj.get("orig_steps"),
                "kept_steps": obj.get("kept_steps"),
                "per_step_pns": obj.get("per_step_pns"),
                "ps": obj.get("ps"),
                "rollout_backend": obj.get("rollout_backend"),
                "k": obj.get("k"),
                "alpha": obj.get("alpha"),
            },
        }
        if passthrough_unknown_fields:
            for k, v in obj.items():
                if k in _KNOWN_TOP_FIELDS:
                    continue
                if k in row["extra_info"]:
                    continue
                row["extra_info"][k] = v
        return row

    # mode == "subproblems" (legacy-ish): pack steps into user turns.
    steps = _pick_steps(obj, use_kept_steps=use_kept_steps)
    steps = _maybe_drop_hash_answer_steps(steps, drop_hash_answer_step=drop_hash_answer_step)
    if not steps:
        return None

    subproblems: list[str] = []
    expected_answers: list[str] = []
    for i, step in enumerate(steps, start=1):
        instruction = (
            "Solve this subproblem correctly and briefly. "
            "You may state key equations/conclusions directly; avoid long derivations."
        )
        if include_problem_each_turn or i == 1:
            sp = f"Problem:\n{question}\n\nSubproblem {i}:\n{instruction}\n\nTarget step:\n{step}"
        else:
            sp = f"Subproblem {i}:\n{instruction}\n\nTarget step:\n{step}"
        subproblems.append(sp)
        expected_answers.append(step)

    final_user = f"{final_synthesis_user_prompt}\n\nProblem:\n{question}"
    subproblems.append(final_user)

    cot_final = str(obj.get("cot_final") or "\n".join(steps)).strip()
    final_expected = _ensure_answer_tag(cot_final, gold)
    expected_answers.append(final_expected)

    prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": subproblems[0]}]

    row = {
        "data_source": data_source,
        "agent_name": "step_verify_agent",
        "prompt": prompt,
        "reward_model": {"style": "rule", "ground_truth": gold},
        "extra_info": {
            "problem_text": question,
            "gold_answer": gold,
            "type": obj.get("type", ""),
            "level": obj.get("level", ""),
            "subproblems": subproblems,
            "expected_answers": expected_answers,
            # passthrough for analysis/debugging
            "cot_original": obj.get("cot_original"),
            "cot_final": obj.get("cot_final"),
            "orig_steps": obj.get("orig_steps"),
            "kept_steps": obj.get("kept_steps"),
            "per_step_pns": obj.get("per_step_pns"),
            "ps": obj.get("ps"),
            "rollout_backend": obj.get("rollout_backend"),
            "k": obj.get("k"),
            "alpha": obj.get("alpha"),
        },
    }
    if passthrough_unknown_fields:
        for k, v in obj.items():
            if k in _KNOWN_TOP_FIELDS:
                continue
            if k in row["extra_info"]:
                continue
            row["extra_info"][k] = v
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, help="Input JSONL generated by causual_math_gen.py")
    ap.add_argument("--out_jsonl", required=True, help="Output AgentLoop JSONL (append-only overwrite)")
    ap.add_argument("--out_parquet", default="", help="Optional parquet output path (built from JSONL at the end).")
    ap.add_argument("--max_samples", type=int, default=0, help="0 means all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--use_kept_steps",
        action="store_true",
        help="Use kept_steps/cot_final as step targets (recommended). If unset, uses orig_steps.",
    )
    ap.add_argument(
        "--include_problem_each_turn",
        action="store_true",
        help="If set, include full Problem text in every subproblem (more context, longer prompts).",
    )
    ap.add_argument(
        "--mode",
        choices=["interactive", "subproblems"],
        default="interactive",
        help="interactive: only output initial prompt; loop generates intermediate user feedback dynamically. "
        "subproblems: pack steps into extra_info.subproblems.",
    )
    ap.add_argument(
        "--drop_hash_answer_step",
        action="store_true",
        help="Drop GSM8K-style trailing '#### <answer>' steps from orig_steps/kept_steps (useful for mode=subproblems).",
    )
    ap.add_argument(
        "--passthrough_unknown_fields",
        action="store_true",
        help="Copy unknown top-level fields from input obj into output extra_info (e.g., ds_index/shard_id/...).",
    )
    ap.add_argument("--data_source", default="pns_step_verify_agent", help="data_source field for routing/metadata.")
    ap.add_argument(
        "--agent_name",
        default="step_verify_agent",
        help="Agent loop name to put into each row (e.g., step_verify_agent / step_causal_verifier_agent).",
    )
    ap.add_argument(
        "--lang",
        choices=["en"],
        default="en",
        help="Language of system/final prompts. (Only English is supported to avoid mixing languages.)",
    )
    args = ap.parse_args()

    random.seed(int(args.seed))

    system_prompt_en = (
        "You solve a math problem via multiple subproblems.\n"
        "For each user turn, answer that subproblem correctly and briefly.\n"
        "On the final user prompt, provide a complete solution and end with a final line: <ANSWER>...</ANSWER>.\n"
        "No chit-chat."
    )
    system_prompt = system_prompt_en
    final_prompt = FINAL_SYNTHESIS_USER_PROMPT_EN

    in_path = str(args.in_jsonl)
    out_path = str(args.out_jsonl)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    rows: list[dict[str, Any]] = []
    n_in = 0
    for obj in _iter_jsonl(in_path):
        n_in += 1
        if int(args.max_samples) > 0 and len(rows) >= int(args.max_samples):
            break
        row = build_agentloop_row(
            obj,
            system_prompt=system_prompt,
            final_synthesis_user_prompt=final_prompt,
            use_kept_steps=bool(args.use_kept_steps),
            include_problem_each_turn=bool(args.include_problem_each_turn),
            data_source=str(args.data_source),
            mode=str(args.mode),
            drop_hash_answer_step=bool(args.drop_hash_answer_step),
            passthrough_unknown_fields=bool(args.passthrough_unknown_fields),
        )
        if row is None:
            continue
        # override agent name
        row["agent_name"] = str(args.agent_name)
        rows.append(row)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.out_parquet:
        import pandas as pd

        out_pq = str(args.out_parquet)
        os.makedirs(os.path.dirname(out_pq) or ".", exist_ok=True)
        pd.DataFrame(rows).to_parquet(out_pq, index=False)

    print(
        json.dumps(
            {
                "in_jsonl": in_path,
                "out_jsonl": out_path,
                "out_parquet": (str(args.out_parquet) if args.out_parquet else None),
                "read_lines": n_in,
                "written": len(rows),
                "use_kept_steps": bool(args.use_kept_steps),
                "include_problem_each_turn": bool(args.include_problem_each_turn),
                "lang": args.lang,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


