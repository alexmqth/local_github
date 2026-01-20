# pns_prompt_based_math.py
# -*- coding: utf-8 -*-
"""
Prompt-based PNS on MATH (qwedsacf/competition_math).
- Filters out Geometry problems
- Keeps original step text (not used in computation)
- Extracts final boxed answer from 'solution'
- vLLM or Transformers backend for rollout
"""

from __future__ import annotations
import os, re, math, json, random
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Dict, Any, Optional, Tuple, Protocol, Union, overload
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from datasets import load_dataset

# -----------------------------
# Dataset helpers
# -----------------------------

_ANSWER_TAG_RE = re.compile(r"<\s*ANSWER\s*>(.*?)<\s*/\s*ANSWER\s*>", flags=re.IGNORECASE | re.DOTALL)
_GSM8K_HASH_ANSWER_RE = re.compile(r"####\s*([^\n\r]+)")


def _extract_last_boxed_content(text: str) -> Optional[str]:
    """
    Extract the content of the LAST LaTeX \\boxed{...} in `text`.
    Supports nested braces by brace matching (e.g. \\boxed{\\frac{1}{2}}).
    Returns the inner content without outer braces.
    """
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

    for s_idx in reversed(starts):
        span = parse_from(s_idx)
        if not span:
            continue
        a, b = span
        inner = t[a:b].strip()
        # Strip surrounding $...$ / $$...$$
        if inner.startswith("$$") and inner.endswith("$$") and len(inner) >= 4:
            inner = inner[2:-2].strip()
        if inner.startswith("$") and inner.endswith("$") and len(inner) >= 2:
            inner = inner[1:-1].strip()
        # Strip trailing punctuation
        inner = inner.rstrip(" .;，。；! \n\r\t")
        return inner if inner else None
    return None

def latex_frac_to_str(s: str) -> str:
    """Rudimentary canonicalizer: \frac{a}{b} -> a/b ; remove \! and spaces."""
    s = s.replace(r"\!", "").strip()
    s = re.sub(r"\\dfrac", r"\\frac", s)
    def _rep(m):
        a, b = m.group(1), m.group(2)
        return f"{a}/{b}"
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", _rep, s)
    return s

def canonical_answer(ans: str) -> str:
    """Turn LaTeX-ish boxed content into a simple canonical string."""
    s = latex_frac_to_str(ans)
    s = s.strip()
    # Strip surrounding $...$
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    # collapse spaces
    s = re.sub(r"\s+", "", s)
    # Common aliases
    s = s.replace(r"\sqrt{1}", "1")  # trivial cleanup
    return s


def extract_gsm8k_hash_answer(text: str) -> Optional[str]:
    """Extract GSM8K final answer after '####' and canonicalize it."""
    if not isinstance(text, str) or not text.strip():
        return None
    m = _GSM8K_HASH_ANSWER_RE.search(text)
    if not m:
        return None
    ans = (m.group(1) or "").strip().replace(",", "")
    return canonical_answer(ans) if ans else None

def try_numeric(ans: str) -> Optional[float]:
    """Try to convert canonical 'a/b' or number to float; else None."""
    # allow things like 3, -2.5, a/b
    if re.fullmatch(r"-?\d+(\.\d+)?", ans):
        return float(ans)
    if re.fullmatch(r"-?\d+/-?\d+", ans):
        try:
            a, b = ans.split("/")
            return float(Fraction(int(a), int(b)))
        except Exception:
            return None
    return None

def extract_boxed_answer(solution_text: str) -> Optional[str]:
    """Extract content inside the last \\boxed{...} in the text (supports nested braces)."""
    inner = _extract_last_boxed_content(str(solution_text or ""))
    if not inner:
        return None
    return canonical_answer(inner)

def split_solution_into_steps(solution_text: str) -> List[str]:
    """
    Preserve original wording as much as possible:
    - split by line breaks first
    - further split by sentence boundary '. ' ONLY if the boundary is OUTSIDE LaTeX math environments
    """

    def _split_sentences_outside_math(s: str) -> List[str]:
        """
        Split on ".<spaces>" boundaries only when we are NOT inside common LaTeX math environments.
        This is a heuristic, but avoids common failure cases where LaTeX symbols/commands contain periods.

        Supported math delimiters (best-effort):
        - \\[ ... \\]
        - \\( ... \\)
        - $$ ... $$
        - $ ... $   (very heuristic; assumes $ is not escaped)
        """
        if not s:
            return []
        out: List[str] = []
        buf: List[str] = []

        in_dollar = False       # $...$
        in_dollars2 = False     # $$...$$
        in_bracket = False      # \[...\]
        in_paren = False        # \(...\)

        i = 0
        n = len(s)
        while i < n:
            ch = s[i]

            # Detect \[ \] \( \)
            if ch == "\\" and i + 1 < n:
                nxt = s[i + 1]
                if not (in_dollar or in_dollars2):
                    if nxt == "[":
                        in_bracket = True
                    elif nxt == "]":
                        in_bracket = False
                    elif nxt == "(":
                        in_paren = True
                    elif nxt == ")":
                        in_paren = False
                buf.append(ch)
                buf.append(nxt)
                i += 2
                continue

            # Detect $$ ... $$
            if ch == "$":
                if i + 1 < n and s[i + 1] == "$":
                    in_dollars2 = not in_dollars2
                    buf.append("$$")
                    i += 2
                    continue
                # Detect $ ... $
                if not in_dollars2:
                    in_dollar = not in_dollar
                buf.append(ch)
                i += 1
                continue

            # Split on ".<spaces>" only outside math
            if (
                ch == "."
                and (i == 0 or s[i - 1] != "\\")
                and not (in_dollar or in_dollars2 or in_bracket or in_paren)
            ):
                j = i + 1
                # require at least one whitespace to be a sentence boundary (match old regex intent)
                if j < n and s[j].isspace():
                    # consume all whitespace after dot
                    while j < n and s[j].isspace():
                        j += 1
                    piece = "".join(buf).strip()
                    if piece:
                        out.append(piece)
                    buf = []
                    i = j
                    continue

            buf.append(ch)
            i += 1

        last = "".join(buf).strip()
        if last:
            out.append(last)
        return out

    # Split by newlines first; then only split sentences OUTSIDE math.
    lines = [ln.strip() for ln in solution_text.splitlines() if ln.strip()]
    steps: List[str] = []
    for ln in lines:
        parts = _split_sentences_outside_math(ln)
        steps.extend(parts)
    return steps

# -----------------------------
# PNS plumbing
# -----------------------------

def semantic_dissimilar(generated_tail: str, original_tail: str,
                        method: str = "ngram", thr: float = 0.8) -> bool:
    """
    Return True if 'generated_tail' is sufficiently different from 'original_tail'.
    For now, a light heuristic: Jaccard over lowercased word unigrams < (1 - thr).
    You can swap to SBERT/BERTScore here.
    """
    def grams(s: str) -> set:
        toks = re.findall(r"[A-Za-z0-9\\]+", s.lower())
        return set(toks)
    g, o = grams(generated_tail), grams(original_tail)
    if not g or not o:
        return True
    inter = len(g & o)
    union = len(g | o)
    jacc = inter / union if union else 0.0
    return jacc < (1.0 - thr)  # lower overlap => more dissimilar

class Validator(Protocol):
    def __call__(self, full_cot: str, gold_ans: str) -> float: ...


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={**headers, "Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=float(timeout_s)) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def _parse_final_decision(text: str) -> Optional[bool]:
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
        # fallback: sometimes only outputs "Yes"/"No" on last line
        last = t.splitlines()[-1].strip().lower()
        if last in {"yes", "no"}:
            return last == "yes"
        return None
    return m2.group(1).lower() == "yes"


def _build_general_verifier_prompt(*, question: str, ground_truth: str, student_answer: str) -> str:
    # Keep close to TIGER-Lab/general-verifier style (same as scripts/check_answer.py).
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


def _call_openai_compatible(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    api_mode: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout_s: float,
) -> str:
    """
    Call an OpenAI-compatible endpoint (vLLM/sglang/etc.).
    api_mode:
      - "chat": /v1/chat/completions
      - "completion": /v1/completions (uses concatenated prompt)
    """
    base = str(base_url or "").rstrip("/")
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    api_mode_l = str(api_mode or "chat").lower()
    if api_mode_l == "completion":
        url = f"{base}/completions" if base.endswith("/v1") else f"{base}/v1/completions"
        # naive concat
        prompt = "\n".join([f"[{m.get('role','user').upper()}] {m.get('content','')}" for m in messages])
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        }
        data = _post_json(url, payload, headers, timeout_s)
        choices = data.get("choices") or []
        if not choices:
            return ""
        return str(choices[0].get("text") or "")

    url = f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
    }
    data = _post_json(url, payload, headers, timeout_s)
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    return str(msg.get("content") or "")


class ExternalAnswerVerifier:
    """
    Optional expensive verifier that checks (pred vs gold) equivalence via an OpenAI-compatible endpoint.
    Intended as a safety net when string extraction/EM is unreliable.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        api_mode: str = "chat",
        max_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
        timeout_s: float = 120.0,
    ):
        self.base_url = str(base_url or "")
        self.model = str(model or "")
        self.api_key = "" if str(api_key).strip().upper() in {"", "EMPTY", "NONE"} else str(api_key).strip()
        self.api_mode = str(api_mode or "chat")
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.timeout_s = float(timeout_s)

    def verify(self, *, question: str, gold: str, pred: str) -> Optional[bool]:
        prompt = _build_general_verifier_prompt(question=question, ground_truth=gold, student_answer=pred)
        try:
            raw = _call_openai_compatible(
                base_url=self.base_url,
                api_key=self.api_key,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                api_mode=self.api_mode,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                timeout_s=self.timeout_s,
            )
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as e:
            # Fail-soft: if verifier fails, return None (caller decides fallback)
            return None
        return _parse_final_decision(raw or "")

class SimpleMathValidator:
    """
    V(S) in [0,1].
    - Extract final boxed (preferred) or "Answer: ..." number from model output
    - EM against gold (canonical); if both numeric, allow small tolerance
    """
    ANSWER_RE = re.compile(r"(?:^|\n)\s*(?:Answer|Final\s*Answer)\s*:\s*([^\n]+)", re.IGNORECASE)

    def __init__(
        self,
        *,
        verifier: Optional[ExternalAnswerVerifier] = None,
        verifier_question: str = "",
        verifier_on_mismatch: bool = True,
        verifier_on_no_extract: bool = False,
    ):
        self._verifier = verifier
        self._verifier_question = str(verifier_question or "")
        self._verifier_on_mismatch = bool(verifier_on_mismatch)
        self._verifier_on_no_extract = bool(verifier_on_no_extract)

    def __call__(self, full_cot: str, gold_ans: str) -> float:
        # Prefer <ANSWER>...</ANSWER> blocks if present, then fallback to last boxed, then "Answer:" line, then GSM8K '####'.
        cand = None
        blocks = _ANSWER_TAG_RE.findall(str(full_cot or ""))
        if blocks:
            blk = blocks[-1].strip()
            cand = extract_boxed_answer(blk) or canonical_answer(blk)
        if not cand:
            cand = extract_boxed_answer(full_cot)
        if not cand:
            m = self.ANSWER_RE.search(full_cot)
            if m:
                cand = canonical_answer(m.group(1))
        if not cand:
            cand = extract_gsm8k_hash_answer(full_cot)
        if not cand:
            # Optional expensive fallback: ask an external verifier model to judge equivalence.
            if self._verifier is not None and self._verifier_on_no_extract and self._verifier_question:
                ok = self._verifier.verify(question=self._verifier_question, gold=str(gold_ans), pred=str(full_cot or ""))
                if ok is True:
                    return 1.0
            return 0.0
        # canonical compare
        c_gold, c_cand = canonical_answer(gold_ans), canonical_answer(cand)
        # numeric tolerant compare
        ng, nc = try_numeric(c_gold), try_numeric(c_cand)
        if ng is not None and nc is not None:
            return 1.0 if math.isclose(ng, nc, rel_tol=1e-9, abs_tol=1e-9) else 0.0
        if c_gold == c_cand:
            return 1.0

        # Optional expensive fallback when mismatch (helps if extraction/canonicalization still fails).
        if self._verifier is not None and self._verifier_on_mismatch and self._verifier_question:
            ok = self._verifier.verify(question=self._verifier_question, gold=str(gold_ans), pred=str(cand))
            if ok is True:
                return 1.0
        return 0.0

# -----------------------------
# Generation backends
# -----------------------------

class RolloutModel(Protocol):
    @overload
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        n: int = 1,
    ) -> List[str]: ...

    @overload
    def generate(
        self,
        prompt: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        n: int = 1,
    ) -> List[List[str]]: ...

    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        n: int = 1,
    ) -> Union[List[str], List[List[str]]]: ...

@dataclass
class TransformersConfig:
    model_name: str
    device: str = "auto"
    dtype: Optional[str] = None

class TransformersRollout:
    def __init__(self, cfg: TransformersConfig):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=getattr(__import__("torch"), cfg.dtype) if cfg.dtype else None,
            device_map=cfg.device, trust_remote_code=True
        )
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        n: int = 1,
    ) -> Union[List[str], List[List[str]]]:
        import torch
        prompts: List[str] = [prompt] if isinstance(prompt, str) else list(prompt)
        # For HF generate, num_return_sequences applies per input row.
        inputs = self.tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=int(n),
            pad_token_id=self.tok.eos_token_id,
        )
        texts = self.tok.batch_decode(outputs, skip_special_tokens=True)

        # HF returns (batch_size * n) sequences in row-major order.
        grouped: List[List[str]] = []
        for i in range(len(prompts)):
            chunk = texts[i * int(n) : (i + 1) * int(n)]
            p = prompts[i]
            # cut to only completion part
            grouped.append([t[len(p) :] if t.startswith(p) else t for t in chunk])
        return grouped[0] if isinstance(prompt, str) else grouped

@dataclass
class VLLMConfig:
    model_name: str
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9

class VLLMRollout:
    def __init__(self, cfg: VLLMConfig):
        from vllm import LLM, SamplingParams
        self.SamplingParams = SamplingParams
        self.llm = LLM(model=cfg.model_name,
                       tensor_parallel_size=cfg.tensor_parallel_size,
                       gpu_memory_utilization=cfg.gpu_memory_utilization,
                       trust_remote_code=True)

    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        n: int = 1,
    ) -> Union[List[str], List[List[str]]]:
        prompts: List[str] = [prompt] if isinstance(prompt, str) else list(prompt)
        sp = self.SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, n=int(n))
        outs = self.llm.generate(prompts, sp)
        # Each RequestOutput has `.outputs` as a list of n candidates.
        grouped = [[o.text for o in req.outputs] for req in outs]
        return grouped[0] if isinstance(prompt, str) else grouped

# -----------------------------
# Prompt-based intervention
# -----------------------------

PROMPT_ALTERNATIVE = """You are revising a reasoning trace to explore an alternative path.
Instructions:
- Do NOT repeat the original step verbatim.
- Provide ONE short alternative intermediate step that could still lead to the correct result.
- Keep it mathematically valid.

Problem:
{q}

Reasoning so far (prefix):
{prefix}

Original step to replace:
{s_t}

Write the alternative intermediate step:
"""

PROMPT_CONTINUE = """You are solving a math problem step by step.

Problem:
{q}

Reasoning so far:
{prefix}

Continue the reasoning from the prefix concisely and give the final answer at the end in the form:
Answer: \\boxed{{...}}"""

PROMPT_CONTINUE_GSM8K = """You are solving a grade-school math word problem step by step.

Problem:
{q}

Reasoning so far:
{prefix}

Continue the reasoning from the prefix concisely and give the final answer at the end in the form:
#### <answer>"""

# This indirection lets us switch the continuation format per dataset (keep rollout/strategy identical).
CONTINUE_PROMPT_TEMPLATE = PROMPT_CONTINUE

def prompt_based_alternative(gen: RolloutModel, q: str, s_prev: List[str], s_t: str) -> str:
    """Generate a semantically different alternative step."""
    prompt = PROMPT_ALTERNATIVE.format(q=q, prefix="\n".join(s_prev), s_t=s_t)
    alt = gen.generate(prompt, max_new_tokens=96, temperature=0.7, top_p=0.9, n=1)[0].strip()
    # post-trim to single line
    alt = alt.split("\n")[0].strip()
    return alt

def rollout_after_intervention(gen: RolloutModel, q: str, prefix_steps: List[str], k: int) -> List[str]:
    prompt = CONTINUE_PROMPT_TEMPLATE.format(q=q, prefix="\n".join([s for s in prefix_steps if s]))
    return gen.generate(prompt, max_new_tokens=384, temperature=0.7, top_p=0.9, n=k)

# -----------------------------
# PNS per step
# -----------------------------

def pns_for_step(gen: RolloutModel, validator: Validator, q: str, gold_ans: str,
                 steps: List[str], t: int, k: int, dissim_thr: float = 0.8, retry: int = 3) -> float:
    s_prev = steps[:t]
    s_t = steps[t]
    alt = "" if random.random() < 0.5 else prompt_based_alternative(gen, q, s_prev, s_t)
    new_prefix = s_prev + ([alt] if alt else [])

    # Generate k samples (with dissimilarity filter vs original tail).
    # Optimization: generate in batches (n>1) instead of one-by-one.
    samples, tries = [], 0
    original_tail = " ".join(steps[t:])
    budget = int(retry) * int(k)
    while len(samples) < k and tries < budget:
        need = k - len(samples)
        remaining = budget - tries
        # generate a small batch to amortize overhead; cap by remaining budget
        batch_n = min(remaining, max(need, 4))
        cands = rollout_after_intervention(gen, q, new_prefix, int(batch_n))
        tries += int(batch_n)
        for cand in cands:
            if semantic_dissimilar(cand, original_tail, thr=dissim_thr):
                samples.append(cand)
                if len(samples) >= k:
                    break

    if not samples:
        return 1.0  # conservative: mark necessary

    vs = [validator(s, gold_ans) for s in samples]
    return 1.0 - sum(vs) / len(vs)

def optimize_cot_by_pns(gen: RolloutModel, validator: Validator, q: str, gold_ans: str,
                        S_init: str, k: int = 5, alpha: float = 0.6,
                        eval_indices: Optional[List[int]] = None) -> Dict[str, Any]:
    steps = split_solution_into_steps(S_init)
    # PS check: does original solution contain a final answer matching gold?
    # (Works for both MATH boxed and GSM8K #### formats.)
    ps = 1 if float(validator(S_init, gold_ans)) >= 1.0 else 0
    if ps == 0:
        return {
            "S_final": S_init,
            "ps": 0,
            "per_step_pns": [],
            "kept_steps": [],
            "orig_steps": steps,
            "pns_eval_indices": [],
            "pns_eval_mask": [0 for _ in steps],
        }

    # Choose which step indices to evaluate PNS on.
    # If eval_indices is None: evaluate all steps.
    if eval_indices is None:
        eval_set = set(range(len(steps)))
        eval_indices_out = list(range(len(steps)))
    else:
        eval_set = set(int(i) for i in eval_indices if 0 <= int(i) < len(steps))
        eval_indices_out = sorted(list(eval_set))

    kept: List[str] = []
    # Align per_step_pns to orig_steps length; None means "not evaluated".
    per_step_pns: List[Optional[float]] = [None for _ in steps]
    pns_eval_mask: List[int] = [1 if i in eval_set else 0 for i in range(len(steps))]
    t = 0
    while t < len(steps):
        if t in eval_set:
            # Build working sequence as kept + remaining to pass into pns_for_step
            working = kept + steps[t:]
            pns = pns_for_step(gen, validator, q, gold_ans, working, len(kept), k)
            per_step_pns[t] = float(pns)
            if float(pns) > float(alpha):
                kept.append(steps[t])  # necessary
            # else: drop this step
        else:
            # Not evaluated => keep by default (no intervention), so downstream kept_steps is not misleading.
            kept.append(steps[t])
        t += 1

    S_final = "\n".join(kept)
    return {
        "S_final": S_final,
        "ps": 1,
        "per_step_pns": per_step_pns,
        "kept_steps": kept,
        "orig_steps": steps,
        "pns_eval_indices": eval_indices_out,
        "pns_eval_mask": pns_eval_mask,
    }

# -----------------------------
# External rollout hook (placeholder)
# -----------------------------

class ExternalRolloutProxy:
    """
    Placeholder for external (stronger) model rollout.
    Implement .generate(...) with your preferred stronger model.
    Then pass this instance wherever a RolloutModel is expected.
    """
    def generate(self, prompt: str, max_new_tokens: int = 256,
                 temperature: float = 0.7, top_p: float = 0.9, n: int = 1) -> List[str]:
        raise NotImplementedError("Plug in your external model here.")

# -----------------------------
# Batch runner
# -----------------------------

def load_math_dataset(split: str = "train"):
    """
    qwedsacf/competition_math format:
      fields: problem, solution, level ('Level 1'..'Level 5'), type (Algebra, ... Geometry, ...).
      solution contains final answer as \\boxed{...}
    """
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")  # this dataset has a single split as 'train'
    # filter out Geometry
    ds = ds.filter(lambda ex: ex.get("type", "") != "Geometry")
    return ds


def load_gsm8k_dataset(*, config_name: str = "socratic", split: str = "train"):
    """
    openai/gsm8k:
      - question: str
      - answer: str (socratic contains multi-step reasoning + final '#### <ans>')
    """
    return load_dataset("openai/gsm8k", config_name, split=split)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["vllm", "hf"], default="hf")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", type=str, default="pns_out.jsonl")
    parser.add_argument("--dataset", choices=["math_lighteval", "gsm8k_socratic"], default="math_lighteval")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--gsm8k_config", type=str, default="socratic", help="HF config name for openai/gsm8k")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pns_eval_max_steps",
        type=int,
        default=0,
        help="If >0, only evaluate PNS on at most N steps per solution; other steps are kept by default.",
    )
    parser.add_argument(
        "--pns_eval_strategy",
        choices=["random", "first", "last", "uniform", "middle"],
        default="middle",
        help="How to choose which steps to evaluate when pns_eval_max_steps>0.",
    )
    parser.add_argument(
        "--pns_exclude_last_n",
        type=int,
        default=1,
        help="Exclude the last N steps from PNS evaluation (default: 1, usually the final-answer step).",
    )
    parser.add_argument(
        "--pns_allow_answer_steps",
        action="store_true",
        help="If set, allow evaluating steps that look like they contain the final answer (\\boxed, <ANSWER>, 'Answer:').",
    )
    # Optional expensive final-answer verifier (OpenAI-compatible, e.g. vLLM served TIGER-Lab/general-verifier)
    parser.add_argument("--answer_verifier_enable", action="store_true", help="Enable external verifier safety net.")
    parser.add_argument("--answer_verifier_base_url", type=str, default="", help='e.g. "http://127.0.0.1:8000/v1"')
    parser.add_argument("--answer_verifier_model", type=str, default="", help='served model name, e.g. "general-verifier"')
    parser.add_argument("--answer_verifier_api_key", type=str, default="", help="Bearer token (or EMPTY).")
    parser.add_argument("--answer_verifier_api_mode", type=str, default="chat", choices=["chat", "completion"])
    parser.add_argument("--answer_verifier_max_tokens", type=int, default=64)
    parser.add_argument("--answer_verifier_timeout_s", type=float, default=120.0)
    parser.add_argument(
        "--answer_verifier_on_mismatch",
        action="store_true",
        help="If set, call verifier when extracted answers do not match (recommended).",
    )
    parser.add_argument(
        "--answer_verifier_on_no_extract",
        action="store_true",
        help="If set, call verifier when no answer can be extracted (very expensive; default off).",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Init generator
    if args.backend == "vllm":
        gen: RolloutModel = VLLMRollout(VLLMConfig(model_name=args.model))
    else:
        gen = TransformersRollout(TransformersConfig(model_name=args.model, device="auto", dtype=None))

    verifier: Optional[ExternalAnswerVerifier] = None
    if bool(getattr(args, "answer_verifier_enable", False)):
        if not str(args.answer_verifier_base_url or "").strip() or not str(args.answer_verifier_model or "").strip():
            raise ValueError("--answer_verifier_enable requires --answer_verifier_base_url and --answer_verifier_model")
        verifier = ExternalAnswerVerifier(
            base_url=str(args.answer_verifier_base_url),
            model=str(args.answer_verifier_model),
            api_key=str(args.answer_verifier_api_key or ""),
            api_mode=str(args.answer_verifier_api_mode or "chat"),
            max_tokens=int(args.answer_verifier_max_tokens),
            temperature=0.0,
            top_p=1.0,
            timeout_s=float(args.answer_verifier_timeout_s),
        )

    # Dataset selection (keep rollout/strategy the same)
    if str(args.dataset) == "gsm8k_socratic":
        ds = load_gsm8k_dataset(config_name=str(args.gsm8k_config), split=str(args.split))
        print(f"[info] dataset={args.dataset}({args.gsm8k_config}/{args.split}) size: {len(ds)}")
        # Ensure rollout continuations follow GSM8K-style final answer format.
        global CONTINUE_PROMPT_TEMPLATE
        CONTINUE_PROMPT_TEMPLATE = PROMPT_CONTINUE_GSM8K
    else:
        ds = load_math_dataset()
        print(f"[info] dataset={args.dataset} size (no-Geometry): {len(ds)}")
        global CONTINUE_PROMPT_TEMPLATE
        CONTINUE_PROMPT_TEMPLATE = PROMPT_CONTINUE

    seen = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for ex in ds:
            if seen >= args.max_samples:
                break
            if str(args.dataset) == "gsm8k_socratic":
                q = str(ex.get("question") or "")
                S_init = str(ex.get("answer") or "")
                gold = extract_gsm8k_hash_answer(S_init)
                if not gold:
                    continue
                ex_type = "gsm8k"
                ex_level = ""
            else:
                q = ex["problem"]
                S_init = ex["solution"]  # keep original full solution (used as initial CoT text)
                gold = extract_boxed_answer(S_init)  # use solution's boxed as gold
                if not gold:
                    continue
                ex_type = ex.get("type", "")
                ex_level = ex.get("level", "")

            # Re-bind validator with the current question for optional verifier calls.
            validator = SimpleMathValidator(
                verifier=verifier,
                verifier_question=str(q),
                verifier_on_mismatch=bool(getattr(args, "answer_verifier_on_mismatch", False)),
                verifier_on_no_extract=bool(getattr(args, "answer_verifier_on_no_extract", False)),
            )

            # also keep original step list (not used in computation)
            orig_steps = split_solution_into_steps(S_init)
            # If the solution is already very short, skip (saves lots of model calls).
            if len(orig_steps) <= 2:
                continue

            # Choose indices to evaluate PNS on (optional).
            eval_indices: Optional[List[int]] = None
            max_eval = int(args.pns_eval_max_steps)
            if max_eval > 0:
                # Prefer choosing "middle" steps and avoid final-answer steps.
                n_steps = len(orig_steps)
                exclude = set()
                last_n = max(0, int(getattr(args, "pns_exclude_last_n", 1)))
                if last_n > 0:
                    for i in range(max(0, n_steps - last_n), n_steps):
                        exclude.add(i)

                # Exclude obvious "answer step" unless user explicitly allows it.
                if not bool(getattr(args, "pns_allow_answer_steps", False)):
                    for i, s in enumerate(orig_steps):
                        ss = str(s or "")
                        if "\\boxed" in ss or "<ANSWER" in ss.upper() or re.search(
                            r"(?:^|\n)\s*(?:Answer|Final\s*Answer)\s*:", ss, re.IGNORECASE
                        ):
                            exclude.add(i)

                allowed = [i for i in range(n_steps) if i not in exclude]
                if not allowed:
                    allowed = list(range(n_steps))  # fallback: don't exclude everything

                if max_eval >= len(allowed):
                    eval_indices = list(allowed)
                else:
                    st = str(args.pns_eval_strategy or "middle").lower()
                    if st == "first":
                        eval_indices = list(allowed[:max_eval])
                    elif st == "last":
                        eval_indices = list(allowed[-max_eval:])
                    elif st == "uniform":
                        if max_eval <= 1:
                            eval_indices = [allowed[len(allowed) // 2]]
                        else:
                            step = (len(allowed) - 1) / float(max_eval - 1)
                            eval_indices = [allowed[int(round(i * step))] for i in range(max_eval)]
                            eval_indices = sorted(set(eval_indices))
                    elif st == "middle":
                        mid = len(allowed) // 2
                        half = max_eval // 2
                        start = max(0, mid - half)
                        end = min(len(allowed), start + max_eval)
                        start = max(0, end - max_eval)
                        eval_indices = list(allowed[start:end])
                    else:
                        eval_indices = random.sample(list(allowed), k=max_eval)

            result = optimize_cot_by_pns(
                gen, validator, q, gold, S_init, k=args.k, alpha=args.alpha, eval_indices=eval_indices
            )
            out = {
                "dataset": str(args.dataset),
                "split": str(args.split),
                "question": q,
                "gold_answer": gold,
                "type": ex_type,
                "level": ex_level,
                "cot_original": S_init,
                "orig_steps": orig_steps,    # 保留原文步骤（暂不使用）
                "ps": result["ps"],
                "per_step_pns": result.get("per_step_pns", []),
                "cot_final": result.get("S_final", S_init),
                "kept_steps": result.get("kept_steps", []),
                "pns_eval_indices": result.get("pns_eval_indices", []),
                "pns_eval_mask": result.get("pns_eval_mask", []),
                "rollout_backend": args.backend,
                "k": args.k,
                "alpha": args.alpha,
                "pns_eval_max_steps": int(args.pns_eval_max_steps),
                "pns_eval_strategy": str(args.pns_eval_strategy),
                "pns_exclude_last_n": int(getattr(args, "pns_exclude_last_n", 1)),
                "pns_allow_answer_steps": bool(getattr(args, "pns_allow_answer_steps", False)),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            seen += 1
            if seen % 10 == 0:
                print(f"[info] processed {seen}")

    print(f"[done] wrote {seen} items to {args.output}")

if __name__ == "__main__":
    main()
