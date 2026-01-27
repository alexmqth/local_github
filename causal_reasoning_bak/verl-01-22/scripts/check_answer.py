"""
Evaluate (pred vs gold) equivalence with a verifier model served by vLLM (OpenAI-compatible API).

Supports your JSON format like:
  {"gold": "...", "pred": "...", "trace_user": ["Problem: ..."], ...}

It will:
  - extract the math problem text from trace_user[0] (strip "Problem:" and tail instruction)
  - call vLLM /v1/chat/completions (or /v1/completions)
  - parse "Final Decision: Yes/No" (or 1/0 if you set a strict prompt)
  - write an output jsonl with added fields:
      verifier_raw, verifier_decision, verifier_correct

Example (chat API):
  python3 verl/experimental/agent_loop/general_verifier_vllm_eval.py \
    --input_jsonl input.jsonl \
    --output_jsonl output.judged.jsonl \
    --base_url http://127.0.0.1:8000/v1 \
    --model general-verifier \
    --api_key EMPTY \
    --api_mode chat

Example vLLM server:
  python -m vllm.entrypoints.openai.api_server \
    --model TIGER-Lab/general-verifier \
    --served-model-name general-verifier \
    --host 0.0.0.0 --port 8000 \
    --dtype float16 --max-model-len 4096
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_TAIL_MARKERS = [
    "I will prompt you with backward thinking",
    "I will prompt you with backward",
]


def _strip_tail(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    t = s
    for m in _TAIL_MARKERS:
        if m in t:
            t = t.split(m, 1)[0]
    return t.strip()


def _strip_problem_prefix(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    t = s.strip()
    for prefix in ("Problem:", "Problem:\n", "Question:", "Question:\n"):
        if t.lower().startswith(prefix.lower()):
            t = t[len(prefix) :].lstrip()
            break
    return t


def _extract_problem_from_trace_user(trace_user: Any) -> str:
    """
    Best-effort: use trace_user[0] as the problem prompt.
    """
    if isinstance(trace_user, list) and trace_user:
        s = str(trace_user[0] or "")
    else:
        s = ""
    s = _strip_tail(s)
    s = _strip_problem_prefix(s)
    return s.strip()


def build_general_verifier_prompt(*, question: str, ground_truth: str, student_answer: str) -> str:
    # Keep close to TIGER-Lab example; this is not chat-format dependent.
    return (
        f"User: ### Question: {question}\n\n"
        f"### Ground Truth Answer: {ground_truth}\n\n"
        f"### Student Answer: {student_answer}\n\n"
        "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
        "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
        'If the student\'s answer is correct, output "Final Decision: Yes". '
        'If the student\'s answer is incorrect, output "Final Decision: No".\n'
        'Output EXACTLY one line and nothing else:\n'
        'Final Decision: Yes\n'
        'or\n'
        'Final Decision: No\n'
        "Assistant:"
    )


def parse_final_decision(text: str) -> Optional[bool]:
    """
    Parse verifier output.
    Accepts:
      - "Final Decision: Yes" / "Final Decision: No"
      - or a strict 0/1 single token (if you changed the prompt)
    """
    if not isinstance(text, str):
        return None
    t = text.strip()
    if not t:
        return None

    # strict binary
    m = re.search(r"(?<!\d)([01])(?!\d)", t)
    if m:
        return True if m.group(1) == "1" else False

    m2 = re.search(r"final\s*decision\s*:\s*(yes|no)", t, flags=re.IGNORECASE)
    if not m2:
        # Fallback: sometimes the model only outputs "Yes"/"No" without the prefix.
        # Be conservative: look at the LAST short line.
        last_line = t.splitlines()[-1].strip().lower()
        if last_line in {"yes", "no"}:
            return last_line == "yes"
        return None
    return m2.group(1).lower() == "yes"


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={**headers, "Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def call_vllm(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    api_mode: str,
    max_tokens: int,
    timeout_s: float,
) -> str:
    base = base_url.rstrip("/")
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if api_mode == "completion":
        url = f"{base}/completions" if base.endswith("/v1") else f"{base}/v1/completions"
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": int(max_tokens),
        }
        data = _post_json(url, payload, headers, timeout_s)
        choices = data.get("choices") or []
        if not choices:
            return ""
        return str(choices[0].get("text") or "")

    # chat mode
    url = f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": int(max_tokens),
    }
    data = _post_json(url, payload, headers, timeout_s)
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    return str(msg.get("content") or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--base_url", required=True, help='e.g. "http://127.0.0.1:8000/v1"')
    ap.add_argument("--model", required=True, help='served model name, e.g. "general-verifier"')
    ap.add_argument("--api_key", default="", help="Bearer token. Use EMPTY/blank if your server doesn't require it.")
    ap.add_argument("--api_mode", default="chat", choices=["chat", "completion"])
    ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument("--timeout_s", type=float, default=120.0)
    ap.add_argument("--sleep_s", type=float, default=0.0, help="Optional sleep between requests.")
    ap.add_argument(
        "--debug_fail_k",
        type=int,
        default=5,
        help="Print first K parse-fail raw outputs to stderr (default: 5).",
    )
    args = ap.parse_args()

    api_key = "" if args.api_key.strip().upper() in {"", "EMPTY", "NONE"} else args.api_key.strip()

    n = 0
    n_yes = 0
    n_no = 0
    n_parse_fail = 0
    fail_examples: list[str] = []
    t0 = time.time()

    with open(args.input_jsonl, "r", encoding="utf-8") as fin, open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            n += 1

            gold = str(obj.get("gold") or "")
            pred = str(obj.get("pred") or "")
            question = _extract_problem_from_trace_user(obj.get("trace_user"))

            prompt = build_general_verifier_prompt(question=question, ground_truth=gold, student_answer=pred)
            try:
                raw = call_vllm(
                    base_url=args.base_url,
                    api_key=api_key,
                    model=args.model,
                    prompt=prompt,
                    api_mode=args.api_mode,
                    max_tokens=args.max_tokens,
                    timeout_s=args.timeout_s,
                )
            except (HTTPError, URLError, TimeoutError) as e:
                raw = f"[ERROR] {type(e).__name__}: {e}"

            decision = parse_final_decision(raw)
            if decision is None:
                n_parse_fail += 1
                if len(fail_examples) < max(0, int(args.debug_fail_k)):
                    fail_examples.append(raw[:300])
            else:
                if decision:
                    n_yes += 1
                else:
                    n_no += 1

            obj["verifier_raw"] = raw
            obj["verifier_decision"] = "yes" if decision is True else ("no" if decision is False else None)
            # match == verifier says "Yes" (equivalent)
            obj["verifier_match"] = bool(decision) if decision is not None else None

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            if args.sleep_s > 0:
                time.sleep(float(args.sleep_s))

    dt = time.time() - t0
    if fail_examples:
        print("[debug] parse_fail examples (first few):", file=sys.stderr)
        for i, ex in enumerate(fail_examples):
            print(f"  [{i}] {ex!r}", file=sys.stderr)
    print(
        json.dumps(
            {
                "input": args.input_jsonl,
                "output": args.output_jsonl,
                "n": n,
                "match_yes": n_yes,
                "match_no": n_no,
                "parse_fail": n_parse_fail,
                "match_rate_yes": (n_yes / n) if n else None,
                "seconds": dt,
                "rps": (n / dt) if dt > 0 else None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()


