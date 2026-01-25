#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
causal_inf.py
- Step-wise generation + causal judge gating (PNS)
- Pure bash multi-process sharding (rank::world_size)
- Per-example profiling: wall/policy/judge/tokens/tok-s/attempts
- Robust local-path handling (strip CRLF, preflight checks)
- Tokenizer fallback: if ckpt lacks tokenizer files, try infer from config.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


_CAUSAL_REASONINGMODEL_NAME = "/home/projects/hku-medai/larrypl/code/mq/causal-reasoning/hf_models/CausalReasoningModel"


def _clean_path(p: str) -> str:
    # 防 Windows CRLF/尾部空格导致路径不存在
    return os.path.expanduser(str(p)).strip().rstrip("\r")


def _normalize_answer_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip()
    s = s.replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_gsm8k_answer(ans: str) -> Optional[str]:
    if ans is None:
        return None
    ans = str(ans)
    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", ans)
    if m:
        return m.group(1).strip()
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", ans)
    return nums[-1].strip() if nums else None


def _extract_gt_boxed_from_solution(sol: str) -> Optional[str]:
    if sol is None:
        return None
    sol = str(sol)
    m = re.search(r"\\boxed\{([^}]*)\}", sol)
    if m:
        return m.group(1).strip()
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", sol)
    return nums[-1].strip() if nums else None


def _has_answer_marker(text: str) -> bool:
    if not text:
        return False
    t = str(text)
    if "<ANSWER>" in t and "</ANSWER>" in t:
        return True
    if "\\boxed" in t:
        return True
    if re.search(r"####\s*[-+]?\d", t):
        return True
    return False


def _extract_pred_answer(text: str) -> Optional[str]:
    if not text:
        return None
    t = str(text)

    m = re.search(r"<ANSWER>\s*(.*?)\s*</ANSWER>", t, flags=re.DOTALL)
    if m:
        return _normalize_answer_text(m.group(1))

    m = re.search(r"\\boxed\{([^}]*)\}", t)
    if m:
        return _normalize_answer_text(m.group(1))

    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", t)
    if m:
        return _normalize_answer_text(m.group(1))

    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", t)
    return _normalize_answer_text(nums[-1]) if nums else None


def _build_judge_text(question: str, accepted_steps: List[str], candidate_step: str) -> str:
    parts = []
    parts.append("Question:\n" + (question or "").strip())
    if accepted_steps:
        parts.append("\nAccepted steps so far:\n" + "\n".join([f"{i+1}. {s.strip()}" for i, s in enumerate(accepted_steps)]))
    parts.append("\nCandidate step:\n" + (candidate_step or "").strip())
    parts.append("\nIs the candidate step logically valid given the question and accepted steps? Output 1 for valid else 0.")
    return "\n".join(parts)


def _encode_chat(tokenizer, messages: List[Dict[str, str]], device: str):
    if hasattr(tokenizer, "apply_chat_template"):
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    else:
        s = ""
        for m in messages:
            s += f"{m.get('role','user').upper()}: {m.get('content','')}\n"
        s += "ASSISTANT: "
        ids = tokenizer([s], return_tensors="pt")["input_ids"]
    return ids.to(device)


@dataclass
class LoopCfg:
    max_steps: int = 32
    max_retries: int = 2
    ok_continue_msg: str = "Continue."
    ok_finish_msg: str = "Now provide the final solution and end with <ANSWER>...</ANSWER>."
    fail_msg: str = "The step is invalid. Rewrite this step correctly. Only provide ONE step."


def _resolve_device(req: str, torch) -> str:
    r = (req or "cpu").lower()
    if r != "cpu":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def _resolve_dtype(s: str, torch):
    s = (s or "auto").lower()
    if s == "auto":
        return None
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    if s == "float32":
        return torch.float32
    return None


def _load_tokenizer_with_fallback(AutoTokenizer, tok_path: str, model_path: str, trust_remote_code: bool):
    # 1) 优先 tok_path（用户可传 base tokenizer）
    # 2) 失败则尝试从 model_path/config.json 推断 base_model_name_or_path 或 _name_or_path（只用本地缓存，不联网）
    tok_path = _clean_path(tok_path) if tok_path else ""
    model_path = _clean_path(model_path)

    tried = []

    def _try(p: str, local_only: bool):
        tried.append((p, local_only))
        return AutoTokenizer.from_pretrained(
            p,
            trust_remote_code=trust_remote_code,
            local_files_only=local_only,
        )

    if tok_path:
        local_only = os.path.isdir(tok_path)
        try:
            return _try(tok_path, local_only)
        except Exception:
            pass

    # try from model dir
    if os.path.isdir(model_path):
        try:
            return _try(model_path, True)
        except Exception:
            pass

        # infer from config.json
        cfg_path = os.path.join(model_path, "config.json")
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                guess = cfg.get("base_model_name_or_path") or cfg.get("_name_or_path") or ""
                if isinstance(guess, str) and guess.strip():
                    guess = guess.strip()
                    # 只用本地缓存，不联网（避免卡住）
                    try:
                        return _try(guess, True)
                    except Exception:
                        pass
            except Exception:
                pass

    # 最后：抛出更友好的错误
    msg = [
        "Failed to load tokenizer.",
        f"tok_path={tok_path!r}",
        f"model_path={model_path!r}",
        "Tried (path, local_files_only):",
        *[f"  - {p!r}, local_only={lo}" for p, lo in tried],
        "",
        "Fix:",
        "1) 确认 ckpt 目录里有 tokenizer.json / vocab / merges / spiece 等文件；",
        "2) 或者在命令行传 --policy_tokenizer_path 指向 base 模型 tokenizer 目录；",
        "3) 并确保脚本 dos2unix，避免路径末尾带 \\r。",
    ]
    raise RuntimeError("\n".join(msg))


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--policy_model_path", type=str, required=True)
    ap.add_argument("--policy_tokenizer_path", type=str, default="", help="Optional: base tokenizer path if ckpt lacks tokenizer files.")
    ap.add_argument("--judge_model_path", type=str, default=_CAUSAL_REASONINGMODEL_NAME)

    ap.add_argument("--policy_device", type=str, default="cuda")
    ap.add_argument("--judge_device", type=str, default="cpu")
    ap.add_argument("--policy_dtype", type=str, default="auto")
    ap.add_argument("--judge_dtype", type=str, default="auto")
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--max_steps", type=int, default=32)
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("--step_max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--force_finish_after_steps", type=int, default=0)
    ap.add_argument("--verbose_steps", action="store_true")

    ap.add_argument("--system_prompt", type=str, default="You are a helpful assistant.")
    ap.add_argument("--user_prefix", type=str, default="Solve the problem step by step. Provide the first step (one step only).")

    ap.add_argument("--dataset_name", type=str, required=True)
    ap.add_argument("--dataset_config", type=str, default=None)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--max_examples", type=int, default=0)
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--rank", type=int, default=-1)
    ap.add_argument("--world_size", type=int, default=-1)

    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--profile_every", type=int, default=25)

    ap.add_argument("--dataset_type", type=str, default="auto", choices=["auto", "math", "gsm8k"])

    args = ap.parse_args()

    # ---- sanitize paths ----
    args.policy_model_path = _clean_path(args.policy_model_path)
    args.policy_tokenizer_path = _clean_path(args.policy_tokenizer_path) if args.policy_tokenizer_path else ""
    args.judge_model_path = _clean_path(args.judge_model_path)
    args.dataset_name = _clean_path(args.dataset_name)
    args.out_jsonl = _clean_path(args.out_jsonl)

    # ---- preflight checks (avoid HF repo-id validation issues) ----
    if not os.path.isdir(args.policy_model_path):
        raise FileNotFoundError(
            f"policy_model_path is NOT an existing local directory:\n  {args.policy_model_path}\n"
            f"Common cause: path ends with CRLF '\\r'. Run:  sed -i 's/\\r$//' causal_inf.sh causal_inf.py"
        )
    if not os.path.exists(args.dataset_name):
        # dataset_name could be HF name; allow that
        pass

    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)) or ".", exist_ok=True)

    import torch
    from datasets import load_dataset
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(int(args.seed))

    policy_device = _resolve_device(args.policy_device, torch)
    judge_device = _resolve_device(args.judge_device, torch)

    policy_dtype = _resolve_dtype(args.policy_dtype, torch)
    judge_dtype = _resolve_dtype(args.judge_dtype, torch)

    # ---- sharding ----
    rank = int(args.rank) if int(args.rank) >= 0 else int(os.getenv("RANK", os.getenv("LOCAL_RANK", 0)))
    world_size = int(args.world_size) if int(args.world_size) >= 0 else int(os.getenv("WORLD_SIZE", 1))
    world_size = max(int(world_size), 1)
    rank = 0 if (rank < 0 or rank >= world_size) else rank

    # per-rank output file
    out_path = args.out_jsonl
    if world_size > 1 and ".rank" not in os.path.basename(out_path):
        if out_path.endswith(".jsonl"):
            out_path = out_path[:-6] + f".rank{rank}.jsonl"
        else:
            out_path = out_path + f".rank{rank}.jsonl"

    print(f"[INFO] rank={rank}/{world_size} policy_device={policy_device} judge_device={judge_device}")
    print(f"[INFO] policy_model_path={args.policy_model_path}")
    if args.policy_tokenizer_path:
        print(f"[INFO] policy_tokenizer_path={args.policy_tokenizer_path}")
    print(f"[INFO] out_path={out_path}")

    # ---- load tokenizer (robust) ----
    policy_tokenizer = _load_tokenizer_with_fallback(
        AutoTokenizer,
        tok_path=args.policy_tokenizer_path,
        model_path=args.policy_model_path,
        trust_remote_code=bool(args.trust_remote_code),
    )

    # ---- load models (force local dir) ----
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.policy_model_path,
        torch_dtype=policy_dtype,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=True,
        device_map=None,
    ).to(policy_device)
    policy_model.eval()

    judge_tokenizer = AutoTokenizer.from_pretrained(
        args.judge_model_path,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=os.path.isdir(args.judge_model_path),
    )
    judge_model = AutoModelForSequenceClassification.from_pretrained(
        args.judge_model_path,
        torch_dtype=judge_dtype,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=os.path.isdir(args.judge_model_path),
        device_map=None,
    ).to(judge_device)
    judge_model.eval()

    # ---- dataset ----
    if args.dataset_config:
        ds = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    else:
        ds = load_dataset(args.dataset_name, split=args.split)
    if int(args.max_examples) > 0:
        ds = ds.select(range(min(int(args.max_examples), len(ds))))

    dataset_type = (args.dataset_type or "auto").lower()
    if dataset_type == "auto":
        name_lower = str(args.dataset_name).lower()
        if "gsm8k" in name_lower or "gsm" in name_lower:
            dataset_type = "gsm8k"
        elif "math" in name_lower:
            dataset_type = "math"
        else:
            cols = getattr(ds, "column_names", [])
            dataset_type = "gsm8k" if ("question" in cols and "answer" in cols) else "math"
    print(f"[INFO] dataset_type={dataset_type}")

    cfg = LoopCfg(max_steps=int(args.max_steps), max_retries=int(args.max_retries))

    indices = list(range(rank, len(ds), world_size))
    desc = f"infer (rank {rank}/{world_size})"

    # profiling accumulators
    prof_wall = prof_policy = prof_judge = 0.0
    prof_tokens = prof_attempts = 0

    n_total = 0
    n_correct = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for local_i, i in enumerate(tqdm(indices, desc=desc)):
            ex = ds[i]

            if dataset_type == "gsm8k":
                problem = str(ex.get("question", "") or "")
                gt = _extract_gsm8k_answer(str(ex.get("answer", "") or ""))
            else:
                problem = str(ex.get("problem", "") or "")
                gt = _extract_gt_boxed_from_solution(str(ex.get("solution", "") or ""))

            gt_norm = _normalize_answer_text(gt)

            # -------- per-example profiling --------
            t_wall0 = time.perf_counter() if args.profile else 0.0
            t_policy = 0.0
            t_judge = 0.0
            gen_tokens_total = 0
            attempts_total = 0
            max_prompt_tokens = 0
            # --------------------------------------

            accepted_steps: List[str] = []
            pns_scores: List[int] = []
            retry_counts: List[int] = []
            asked_finish = False

            user_msg = problem
            if str(args.user_prefix or "").strip():
                user_msg = f"{problem}\n\n{args.user_prefix}".strip()

            messages: List[Dict[str, str]] = [
                {"role": "system", "content": str(args.system_prompt or "You are a helpful assistant.")},
                {"role": "user", "content": user_msg},
            ]

            step_idx = 0
            retry_for_current_step = 0

            while step_idx < int(cfg.max_steps):
                input_ids = _encode_chat(policy_tokenizer, messages, device=policy_device)
                max_prompt_tokens = max(max_prompt_tokens, int(input_ids.shape[1]))

                gen_kwargs = {
                    "max_new_tokens": int(args.step_max_new_tokens),
                    "do_sample": bool(float(args.temperature) > 0.0),
                    "temperature": float(args.temperature),
                    "top_p": float(args.top_p),
                }
                eos_id = getattr(policy_tokenizer, "eos_token_id", None)
                if eos_id is not None:
                    gen_kwargs["eos_token_id"] = eos_id

                with torch.no_grad():
                    t0 = time.perf_counter() if args.profile else 0.0
                    out_ids = policy_model.generate(input_ids=input_ids, **gen_kwargs)
                    if args.profile:
                        t_policy += (time.perf_counter() - t0)

                gen = out_ids[0][input_ids.shape[1]:]
                gen_tokens_total += int(gen.numel())
                attempts_total += 1

                assistant_text = policy_tokenizer.decode(gen, skip_special_tokens=True).strip()
                finished = _has_answer_marker(assistant_text)

                judge_text = _build_judge_text(problem, accepted_steps, assistant_text)
                j_inputs = judge_tokenizer(
                    [judge_text],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                j_inputs = {k: v.to(judge_device) for k, v in j_inputs.items()}

                with torch.no_grad():
                    t0j = time.perf_counter() if args.profile else 0.0
                    j_out = judge_model(**j_inputs)
                    if args.profile:
                        t_judge += (time.perf_counter() - t0j)

                    logits = j_out.logits
                    if logits.dim() == 2 and logits.size(-1) == 2:
                        pns = int(torch.argmax(logits, dim=-1).item())
                    else:
                        pns = int((torch.sigmoid(logits.view(-1)) >= 0.5).item())

                pns_scores.append(pns)
                passed = (pns == 1)

                if args.verbose_steps:
                    print(f"[idx={i} step={step_idx} retry={retry_for_current_step} pns={pns} passed={passed} finished={finished}]")

                if passed:
                    accepted_steps.append(assistant_text)
                    retry_counts.append(retry_for_current_step)
                    retry_for_current_step = 0
                    step_idx += 1

                    if finished:
                        break

                    if (not asked_finish) and int(args.force_finish_after_steps) > 0 and step_idx >= int(args.force_finish_after_steps):
                        asked_finish = True
                        messages.append({"role": "user", "content": cfg.ok_finish_msg})
                    else:
                        messages.append({"role": "user", "content": cfg.ok_continue_msg})
                else:
                    retry_for_current_step += 1
                    if retry_for_current_step > int(cfg.max_retries):
                        # give up this step, accept anyway
                        accepted_steps.append(assistant_text)
                        retry_counts.append(retry_for_current_step)
                        retry_for_current_step = 0
                        step_idx += 1
                        if finished:
                            break
                        messages.append({"role": "user", "content": cfg.ok_continue_msg})
                    else:
                        messages.append({"role": "user", "content": cfg.fail_msg})

            final_text = "\n".join(accepted_steps) if accepted_steps else ""
            pred = _extract_pred_answer(final_text)
            pred_norm = _normalize_answer_text(pred)

            correct = (pred_norm is not None) and (gt_norm is not None) and (pred_norm == gt_norm)

            rec = {
                "idx": int(i),
                "question": problem,
                "gt": gt,
                "gt_norm": gt_norm,
                "pred": pred,
                "pred_norm": pred_norm,
                "correct": bool(correct),
                "steps": accepted_steps,
                "pns_scores": pns_scores,
                "retry_counts": retry_counts,
                "asked_finish": bool(asked_finish),
                "profile": None,
            }

            if args.profile:
                wall_s = time.perf_counter() - t_wall0
                rec["profile"] = {
                    "rank": rank,
                    "world_size": world_size,
                    "wall_s": float(wall_s),
                    "policy_s": float(t_policy),
                    "judge_s": float(t_judge),
                    "attempts": int(attempts_total),
                    "gen_tokens": int(gen_tokens_total),
                    "max_prompt_tokens": int(max_prompt_tokens),
                    "tok_per_s": (float(gen_tokens_total) / float(t_policy)) if t_policy > 1e-9 else None,
                }

                prof_wall += wall_s
                prof_policy += t_policy
                prof_judge += t_judge
                prof_tokens += gen_tokens_total
                prof_attempts += attempts_total

                if int(args.profile_every) > 0 and ((local_i + 1) % int(args.profile_every) == 0):
                    avg_wall = prof_wall / (local_i + 1)
                    avg_tok_s = (prof_tokens / prof_policy) if prof_policy > 1e-9 else 0.0
                    print(
                        f"[PROF][rank {rank}] done={local_i+1}/{len(indices)} "
                        f"avg_wall={avg_wall:.3f}s tok/s={avg_tok_s:.1f} "
                        f"policy_share={(prof_policy/prof_wall if prof_wall>1e-9 else 0.0):.2f} "
                        f"judge_share={(prof_judge/prof_wall if prof_wall>1e-9 else 0.0):.2f} "
                        f"avg_attempts={(prof_attempts/(local_i+1)):.2f}"
                    )

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            n_total += 1
            n_correct += int(correct)

    acc = (n_correct / n_total) if n_total else 0.0
    print(f"[DONE][rank {rank}] n_total={n_total} n_correct={n_correct} acc={acc:.4f} out={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

