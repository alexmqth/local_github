"""
Step-wise verifier agent loop with token-level credit assignment metadata.

This agent loop is designed for:
- multi-turn "subproblem" conversations (the environment provides the next user subproblem)
- step-wise verification after each assistant response (hybrid verifier: rule + optional LLM judge)
- recording per-step token spans so we can assign rewards precisely at token level later

Token-level credit assignment
-----------------------------
This loop outputs:
  - extra_fields["turn_scores"]: list[float]        # step rewards
  - extra_fields["step_token_spans"]: list[tuple[int,int]]  # spans in *response token space* (start, end)

Then `AgentLoopWorkerBase._postprocess()` can convert these into per-token `rm_scores` by evenly distributing
each step reward across tokens in its span.
"""

from __future__ import annotations

import importlib
import logging
import os
import re
import time
import socket
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


RuleVerifierFn = Callable[[str, dict[str, Any]], Optional[float]]


def _import_by_path(dotted_path: str) -> Any:
    if ":" in dotted_path:
        module_name, attr = dotted_path.split(":", 1)
    else:
        module_name, attr = dotted_path.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, attr)


def _extract_last_assistant_text(messages: list[dict[str, Any]]) -> str:
    for item in reversed(messages):
        if item.get("role") == "assistant":
            return str(item.get("content", ""))
    return ""


def _has_answer_marker(text: str) -> bool:
    """
    Heuristic: detect whether a model output already contains a "final answer" marker.
    This is useful for debugging "why does the model answer too early?".
    """
    if not isinstance(text, str):
        return False
    t = text
    # common patterns we care about
    if "<answer" in t.lower() or "</answer" in t.lower():
        return True
    if "\\boxed" in t:
        return True
    if re.search(r"final\s+answer", t, flags=re.IGNORECASE):
        return True
    return False


def _extract_final_answer(text: str) -> Optional[str]:
    """
    Best-effort final answer extraction.
    Supports common patterns:
    - <Answer>...</Answer> / <ANSWER>...</ANSWER>
    - '#### <answer>' (GSM8K style)
    - last \\boxed{...}
    Fallback: None
    """
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.strip()

    # <Answer>...</Answer>
    import re

    m = re.findall(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", t, flags=re.IGNORECASE | re.DOTALL)
    if m:
        ans = m[-1].strip()
        if ans:
            t2 = ans
        else:
            t2 = ""
    else:
        t2 = t

    # #### answer
    m2 = re.findall(r"####\s*([^\n\r]+)", t2)
    if m2:
        cand = m2[-1].strip()
        if cand:
            return cand

    # boxed
    m3 = re.findall(r"\\boxed\s*\{(.*?)\}", t2, flags=re.DOTALL)
    if m3:
        cand = m3[-1].strip()
        if cand:
            return cand.rstrip(" .;，。；!")

    # If <Answer> existed, use its content as last resort
    if m:
        cand = t2.strip()
        return cand if cand else None
    return None


def _parse_binary_judge(text: str) -> Optional[float]:
    """
    Parse judge output into 0/1.
    Accepts outputs like:
    - "1" / "0"
    - "答案是 1" / "score:0"
    Rejects ambiguous numeric strings like "10", "0.5", "100".
    Returns:
      1.0 / 0.0 when confidently parsed; otherwise None.
    """
    if not isinstance(text, str):
        return None
    t = text.strip()
    if not t:
        return None
    import re

    m = re.search(r"(?<!\d)([01])(?!\d)", t)
    if not m:
        return None
    return 1.0 if m.group(1) == "1" else 0.0


@dataclass
class _VerifierConfig:
    rule_verifier_fn: Optional[str] = None
    # LLM verifier settings
    llm_enable: bool = False
    llm_always: bool = False
    llm_on_incorrect: bool = True
    llm_on_none: bool = True
    llm_max_new_tokens: int = 16
    # Step LLM judge decoding params (for llm_enable step verification).
    # NOTE:
    # - For local/policy backends, these map to generation sampling params.
    # - For remote backends (OpenAI-compatible), these are sent as `temperature`/`top_p`.
    llm_temperature: float = 0.0
    llm_top_p: float = 1.0
    # Step LLM judge system message (applies to policy/local/remote backends).
    # This is the system role content used when calling the step judge.
    llm_system_prompt: str = "You are a verifier."
    # If True, STEP LLM judge will use the original problem text as `{question}` in llm_prompt_template.
    # If False (default), it uses the current subproblem text.
    #
    # This lets you keep llm_prompt_template unchanged while still exposing the full problem to the judge.
    # IMPORTANT: we define "original problem" as extra_info.problem_text/question/question_hint only
    # (no fallback to the first subproblem), so you won't accidentally feed subproblem[0] as the problem.
    llm_use_problem_text: bool = False
    llm_prompt_template: str = (
        "You are a strict verifier. Judge whether the assistant answer correctly solves the given subproblem.\n"
        "Return ONLY one token: 1 (correct) or 0 (incorrect).\n\n"
        "Subproblem:\n{question}\n\n"
        "Assistant answer:\n{answer}\n"
    )
    # combine strategy when both rule and llm exist
    combine: str = "max"  # max | mean | rule_only | llm_only | prm_only

    # Step reward scaling (process reward)
    # If enabled, scale each per-turn step reward by:
    #   scaled = raw / N * step_total_weight
    # so the total process reward across N turns is <= step_total_weight (when raw in [0,1]).
    step_scale_by_num_turns: bool = False
    step_total_weight: float = 1.0

    # Judge backend (for llm-based step/final verification)
    # - "policy": use the same rollout server_manager (policy model) to judge (current default)
    # - "local": load a separate judge model locally via transformers (no extra vLLM server needed)
    # - "remote": call an external OpenAI-compatible HTTP server (e.g., vLLM OpenAI server) for judging
    judge_backend: str = "policy"  # policy | local | remote
    judge_model_name_or_path: Optional[str] = None
    judge_device: str = "cpu"  # cpu | cuda
    judge_dtype: str = "auto"  # auto | float16 | bfloat16 | float32
    # Remote judge settings (OpenAI-compatible)
    # Example: "http://10.0.0.12:8000/v1" (recommended) or "http://10.0.0.12:8000"
    judge_base_url: Optional[str] = None
    judge_api_key: Optional[str] = None
    judge_timeout_s: float = 120.0
    # Limit in-flight remote judge requests per worker process to avoid flooding the vLLM server.
    # 0 or negative means "no limit".
    judge_max_concurrency: int = 8

    # PRM step reward model (process supervision model) settings (optional)
    # If enabled, compute step rewards in [0,1] from a PRM model by extracting the positive-class probability
    # at the step separator token positions (e.g., "<extra_0>").
    prm_enable: bool = False
    prm_model_name_or_path: Optional[str] = None
    prm_device: str = "cuda"  # cpu | cuda
    prm_dtype: str = "bfloat16"  # auto | float16 | bfloat16 | float32
    prm_step_sep_token: str = "<extra_0>"
    # How to split assistant_text into PRM "steps" when it doesn't already contain prm_step_sep_token:
    # - "auto": if contains sep token => use as-is; else treat as single step
    # - "single": whole assistant_text is one step
    # - "newline": split by non-empty lines as steps
    prm_split_strategy: str = "auto"  # auto | single | newline
    # PRM evaluation mode:
    # - "per_turn": run PRM once per assistant turn (more compute, aligns with subproblem-level judging)
    # - "final_once": run PRM once at the end over the whole problem + all assistant turns, then map sep scores back
    #                to per-turn rewards (cheaper, more "problem-level" judging)
    prm_eval_mode: str = "per_turn"  # per_turn | final_once

    # final reward settings (independent from step reward)
    final_enable: bool = True
    final_weight: float = 1.0
    # final_mode:
    # - "answer": extract/normalize final answer (<Answer>, ####, \\boxed{}) and exact-match to ground_truth;
    #            optional LLM judge can be enabled for robustness.
    # - "proof":  judge the full proof/solution correctness via LLM (expects ground_truth to be a reference proof
    #            or a formal statement; no answer extraction).
    # - "auto":   heuristic: if ground_truth looks like a short answer => "answer", else "proof"
    final_mode: str = "answer"  # answer | proof | auto
    final_llm_enable: bool = False
    final_llm_max_new_tokens: int = 128
    # Final judge decoding params
    # NOTE: Some OpenAI models (e.g., gpt-5-mini) are typically run with non-zero temperature.
    # Keep configurable so users can match their judge model recommendations.
    final_llm_temperature: float = 0.0
    final_llm_top_p: float = 1.0
    # Optional overrides for FINAL judge backend/model (useful when step judge != final judge).
    # If any of these are set, final judge will use them instead of the shared judge_* config.
    final_judge_backend: Optional[str] = None  # policy | local | remote
    final_judge_model_name_or_path: Optional[str] = None
    final_judge_base_url: Optional[str] = None
    final_judge_api_key: Optional[str] = None
    final_judge_timeout_s: Optional[float] = None
    final_llm_prompt_template_answer: str = (
        "You are a strict verifier. Determine whether the FINAL answer is correct.\n"
        "Return ONLY one token: 1 (correct) or 0 (incorrect).\n\n"
        "Problem:\n{problem}\n\n"
        "Assistant final response:\n{answer}\n\n"
        "Ground truth answer:\n{ground_truth}\n"
    )
    final_llm_prompt_template_proof: str = (
        "You are a strict proof verifier.\n"
        "Determine whether the assistant's proof/solution is correct and fully proves the goal.\n"
        "Return ONLY one token: 1 (correct/complete) or 0 (incorrect/incomplete).\n\n"
        "Goal/Problem:\n{problem}\n\n"
        "Assistant proof/solution:\n{answer}\n\n"
        "Reference solution/proof (may be empty):\n{ground_truth}\n"
    )

    # Optional: dump per-trajectory judge traces to JSONL for debugging / analysis.
    # One line per trajectory, includes multi-turn step verification details.
    judge_trace_enable: bool = False
    judge_trace_output_dir: str = "judge_traces"
    judge_trace_filename_prefix: str = "judge_trace"
    judge_trace_max_raw_chars: int = 4000
    # Optional: explicitly set an output JSONL file path (overrides output_dir + prefix).
    # Supports format placeholders: {pid}, {host}
    judge_trace_output_path: Optional[str] = None
    # Real-time persistence knobs:
    # - flush: call f.flush() after each line
    # - fsync: call os.fsync(f.fileno()) after each line (safer, slower)
    judge_trace_flush: bool = True
    judge_trace_fsync: bool = False
    # Output format:
    # - "trajectory": one JSONL line per trajectory (debug-oriented)
    # - "judge_train_steps": one JSONL line per STEP (good for training a judge)
    judge_trace_format: str = "trajectory"
    # For judge_train_steps, which label to emit:
    # - "final": use per-step final score (after combine) as label
    # - "llm": use step LLM judge parsed score as label (falls back to 0 if not used / non-binary)
    # - "rule": use rule score as label (falls back to 0 if None)
    judge_train_label: str = "final"


@register("step_verify_agent")
class StepVerifyAgentLoop(AgentLoopBase):
    """
    Multi-turn subproblem solver with step-wise verification.

    Input contract (per sample)
    ---------------------------
    - raw_prompt: list[dict] : initial messages, usually includes the first user subproblem
    - extra_info.subproblems: list[str] (optional) : full list of subproblems; loop will feed next ones as user turns
      - If provided, we assume raw_prompt already includes subproblems[0] as the last user message.
    - extra_info.expected_answers: list[Any] (optional) : aligned list of expected answers (for rule verifier)

    Config contract (trainer config)
    -------------------------------
    - actor_rollout_ref.rollout.multi_turn.max_user_turns/max_assistant_turns : limits
    - actor_rollout_ref.rollout.response_length/prompt_length : token limits
    - data.apply_chat_template_kwargs : passed to tokenizer/processor chat template
    - rollout.agent.step_verify.* : verifier settings (optional, see below)
    """

    # Class-level counter and lock for periodic sample logging
    _sample_counter: int = 0
    _counter_lock = threading.Lock()
    _log_sample_interval: int = 50  # Default: log every 50 samples

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True

        cls.tokenizer = tokenizer
        cls.processor = processor
        rollout = config.actor_rollout_ref.rollout
        cls.prompt_length = rollout.prompt_length
        cls.response_length = rollout.response_length
        cls.max_user_turns = rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = rollout.multi_turn.max_assistant_turns
        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})

        # used to strip system prefix when encoding incremental turns (consistent with ToolAgentLoop)
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )

        # Verifier config
        # IMPORTANT:
        # `actor_rollout_ref.rollout.agent` is a strict dataclass (AgentLoopConfig) and cannot accept arbitrary fields.
        # Custom per-agent-loop config must be passed via `agent_loop_config_path`, and will arrive here as **kwargs.
        # So we read `step_verify` from kwargs (preferred) instead of from rollout.agent.
        step_cfg = kwargs.get("step_verify") or {}
        if step_cfg and not isinstance(step_cfg, dict):
            # be robust to OmegaConf containers
            try:
                from omegaconf import OmegaConf

                step_cfg = OmegaConf.to_container(step_cfg, resolve=True)  # type: ignore[assignment]
            except Exception:
                step_cfg = {}
        cls.verifier_cfg = _VerifierConfig(
            rule_verifier_fn=step_cfg.get("rule_verifier_fn"),
            llm_enable=bool(step_cfg.get("llm_enable", False)),
            llm_always=bool(step_cfg.get("llm_always", False)),
            llm_on_incorrect=bool(step_cfg.get("llm_on_incorrect", True)),
            llm_on_none=bool(step_cfg.get("llm_on_none", True)),
            llm_max_new_tokens=int(step_cfg.get("llm_max_new_tokens", 256)),
            llm_temperature=float(step_cfg.get("llm_temperature", 0.0)),
            llm_top_p=float(step_cfg.get("llm_top_p", 1.0)),
            llm_system_prompt=str(step_cfg.get("llm_system_prompt", "You are a verifier.")),
            llm_use_problem_text=bool(step_cfg.get("llm_use_problem_text", False)),
            llm_prompt_template=str(step_cfg.get("llm_prompt_template", _VerifierConfig.llm_prompt_template)),
            combine=str(step_cfg.get("combine", "max")),
            step_scale_by_num_turns=bool(step_cfg.get("step_scale_by_num_turns", False)),
            step_total_weight=float(step_cfg.get("step_total_weight", 1.0)),
            judge_backend=str(step_cfg.get("judge_backend", "policy")),
            judge_model_name_or_path=step_cfg.get("judge_model_name_or_path"),
            judge_device=str(step_cfg.get("judge_device", "cpu")),
            judge_dtype=str(step_cfg.get("judge_dtype", "auto")),
            judge_base_url=step_cfg.get("judge_base_url"),
            judge_api_key=step_cfg.get("judge_api_key"),
            judge_timeout_s=float(step_cfg.get("judge_timeout_s", 120.0)),
            judge_max_concurrency=int(step_cfg.get("judge_max_concurrency", 8)),
            prm_enable=bool(step_cfg.get("prm_enable", False)),
            prm_model_name_or_path=step_cfg.get("prm_model_name_or_path"),
            prm_device=str(step_cfg.get("prm_device", "cuda")),
            prm_dtype=str(step_cfg.get("prm_dtype", "bfloat16")),
            prm_step_sep_token=str(step_cfg.get("prm_step_sep_token", "<extra_0>")),
            prm_split_strategy=str(step_cfg.get("prm_split_strategy", "auto")),
            prm_eval_mode=str(step_cfg.get("prm_eval_mode", "per_turn")),
            final_enable=bool(step_cfg.get("final_enable", True)),
            final_weight=float(step_cfg.get("final_weight", 1.0)),
            final_mode=str(step_cfg.get("final_mode", "answer")),
            final_llm_enable=bool(step_cfg.get("final_llm_enable", False)),
            final_llm_max_new_tokens=int(step_cfg.get("final_llm_max_new_tokens", 128)),
            final_llm_temperature=float(step_cfg.get("final_llm_temperature", 0.0)),
            final_llm_top_p=float(step_cfg.get("final_llm_top_p", 1.0)),
            final_judge_backend=step_cfg.get("final_judge_backend"),
            final_judge_model_name_or_path=step_cfg.get("final_judge_model_name_or_path"),
            final_judge_base_url=step_cfg.get("final_judge_base_url"),
            final_judge_api_key=step_cfg.get("final_judge_api_key"),
            final_judge_timeout_s=(
                float(step_cfg["final_judge_timeout_s"]) if "final_judge_timeout_s" in step_cfg else None
            ),
            final_llm_prompt_template_answer=str(
                step_cfg.get(
                    "final_llm_prompt_template_answer", _VerifierConfig.final_llm_prompt_template_answer
                )
            ),
            final_llm_prompt_template_proof=str(
                step_cfg.get(
                    "final_llm_prompt_template_proof", _VerifierConfig.final_llm_prompt_template_proof
                )
            ),
            judge_trace_enable=bool(step_cfg.get("judge_trace_enable", False)),
            judge_trace_output_dir=str(step_cfg.get("judge_trace_output_dir", "judge_traces")),
            judge_trace_filename_prefix=str(step_cfg.get("judge_trace_filename_prefix", "judge_trace")),
            judge_trace_max_raw_chars=int(step_cfg.get("judge_trace_max_raw_chars", 4000)),
            judge_trace_output_path=step_cfg.get("judge_trace_output_path"),
            judge_trace_flush=bool(step_cfg.get("judge_trace_flush", True)),
            judge_trace_fsync=bool(step_cfg.get("judge_trace_fsync", False)),
            judge_trace_format=str(step_cfg.get("judge_trace_format", "trajectory")),
            judge_train_label=str(step_cfg.get("judge_train_label", "final")),
        )

        cls.rule_verifier: Optional[RuleVerifierFn] = None
        if cls.verifier_cfg.rule_verifier_fn:
            fn = _import_by_path(cls.verifier_cfg.rule_verifier_fn)
            if not callable(fn):
                raise TypeError(f"rule_verifier_fn must be callable, got {type(fn).__name__}")
            cls.rule_verifier = fn  # type: ignore[assignment]

        # Local judge model (optional): load once per worker process
        cls._judge_tokenizer = None
        cls._judge_model = None
        cls._remote_judge_semaphore = None
        # PRM model (optional): load once per worker process
        cls._prm_tokenizer = None
        cls._prm_model = None
        backend = (cls.verifier_cfg.judge_backend or "policy").lower()
        if backend == "local":
            if not cls.verifier_cfg.judge_model_name_or_path:
                raise ValueError(
                    "step_verify.judge_backend='local' requires step_verify.judge_model_name_or_path"
                )
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                judge_path = cls.verifier_cfg.judge_model_name_or_path
                cls._judge_tokenizer = AutoTokenizer.from_pretrained(judge_path, trust_remote_code=True)
                dtype = cls.verifier_cfg.judge_dtype.lower()
                if dtype == "float16":
                    torch_dtype = torch.float16
                elif dtype == "bfloat16":
                    torch_dtype = torch.bfloat16
                elif dtype == "float32":
                    torch_dtype = torch.float32
                else:
                    torch_dtype = "auto"

                cls._judge_model = AutoModelForCausalLM.from_pretrained(
                    judge_path, trust_remote_code=True, torch_dtype=torch_dtype
                )
                cls._judge_model.eval()
                device = cls.verifier_cfg.judge_device.lower()
                if device != "cpu":
                    cls._judge_model = cls._judge_model.to(device)
            except Exception as e:
                raise RuntimeError(f"Failed to load local judge model: {e}") from e
        elif backend == "remote":
            # Validate remote config early to fail fast.
            if not cls.verifier_cfg.judge_base_url:
                raise ValueError("step_verify.judge_backend='remote' requires step_verify.judge_base_url")
            if not cls.verifier_cfg.judge_model_name_or_path:
                raise ValueError(
                    "step_verify.judge_backend='remote' requires step_verify.judge_model_name_or_path "
                    "(remote model name)"
                )
            try:
                import asyncio

                mc = int(cls.verifier_cfg.judge_max_concurrency)
                cls._remote_judge_semaphore = asyncio.Semaphore(mc) if mc and mc > 0 else None
            except Exception:
                cls._remote_judge_semaphore = None

        # Load PRM model if enabled
        if bool(getattr(cls.verifier_cfg, "prm_enable", False)):
            if not cls.verifier_cfg.prm_model_name_or_path:
                raise ValueError("step_verify.prm_enable=True requires step_verify.prm_model_name_or_path")
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer

                prm_path = str(cls.verifier_cfg.prm_model_name_or_path)
                try:
                    logger.warning(
                        "PRM loading start. prm_model=%r requested_prm_device=%r CUDA_VISIBLE_DEVICES=%r",
                        prm_path,
                        str(cls.verifier_cfg.prm_device),
                        os.getenv("CUDA_VISIBLE_DEVICES", None),
                    )
                    print(
                        f"[StepVerifyAgentLoop] PRM loading start. prm_model={prm_path} "
                        f"requested_prm_device={cls.verifier_cfg.prm_device} "
                        f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES',None)}",
                        flush=True,
                    )
                except Exception:
                    pass
                cls._prm_tokenizer = AutoTokenizer.from_pretrained(prm_path, trust_remote_code=True)

                dtype = (cls.verifier_cfg.prm_dtype or "auto").lower()
                if dtype == "float16":
                    torch_dtype = torch.float16
                elif dtype == "bfloat16":
                    torch_dtype = torch.bfloat16
                elif dtype == "float32":
                    torch_dtype = torch.float32
                else:
                    torch_dtype = "auto"

                prm_device = str(cls.verifier_cfg.prm_device or "cuda").lower()
                # If user requests CUDA but this Ray actor has no visible GPUs, fail with a clear actionable message.
                # This typically happens when AgentLoopWorker is created with num_gpus=0, so Ray sets
                # CUDA_VISIBLE_DEVICES="" for the worker process.
                if prm_device != "cpu":
                    try:
                        if not torch.cuda.is_available():
                            raise RuntimeError(
                                "PRM is configured to use CUDA (%r) but no CUDA GPUs are available in this process. "
                                "Fix: ensure AgentLoopWorker requests a GPU (Ray actor num_gpus=1). "
                                "You can force it by setting env VERL_AGENT_LOOP_WORKER_NUM_GPUS=1, "
                                "or set step_verify.prm_device=cpu for a (slow) CPU fallback."
                                % prm_device
                            )
                    except RuntimeError:
                        raise
                    except Exception:
                        # If torch probing fails for any reason, keep going and let model loading raise.
                        pass
                # Support explicit device pinning: "cuda:0", "cuda:1", etc.
                # - If prm_device is "cpu": keep on CPU.
                # - If prm_device is "cuda:N": load on CPU then move to that GPU (avoids spreading across GPUs).
                # - Otherwise (e.g., "cuda"): use device_map="auto".
                if prm_device == "cpu":
                    cls._prm_model = AutoModel.from_pretrained(
                        prm_path,
                        trust_remote_code=True,
                        torch_dtype=torch_dtype,
                        device_map=None,
                    ).eval()
                    cls._prm_model = cls._prm_model.to("cpu")
                elif prm_device.startswith("cuda:"):
                    # In Ray actors that request a single GPU, CUDA_VISIBLE_DEVICES is typically a single entry,
                    # so the only visible device is "cuda:0". Map any explicit index to 0 in that case.
                    try:
                        if torch.cuda.is_available() and torch.cuda.device_count() == 1:
                            prm_device = "cuda:0"
                    except Exception:
                        pass
                    cls._prm_model = AutoModel.from_pretrained(
                        prm_path,
                        trust_remote_code=True,
                        torch_dtype=torch_dtype,
                        device_map=None,
                    ).eval()
                    cls._prm_model = cls._prm_model.to(prm_device)
                else:
                    cls._prm_model = AutoModel.from_pretrained(
                        prm_path,
                        trust_remote_code=True,
                        torch_dtype=torch_dtype,
                        device_map="auto",
                    ).eval()

                # --- Diagnostics: help users confirm PRM GPU placement under Ray ---
                try:
                    cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", None)
                    # In Ray, accelerator ids are per-process "visible ids", not necessarily physical ids.
                    ray_accel = None
                    try:
                        ctx = __import__("ray").get_runtime_context()
                        ray_accel = ctx.get_accelerator_ids()
                    except Exception:
                        ray_accel = None

                    prm_model_device = None
                    try:
                        prm_model_device = str(getattr(cls._prm_model, "device", None))
                    except Exception:
                        prm_model_device = None

                    torch_dev_cnt = None
                    torch_cur = None
                    torch_name = None
                    try:
                        if torch.cuda.is_available():
                            torch_dev_cnt = int(torch.cuda.device_count())
                            torch_cur = int(torch.cuda.current_device())
                            torch_name = str(torch.cuda.get_device_name(torch_cur))
                    except Exception:
                        pass

                    logger.warning(
                        "PRM loaded. requested_prm_device=%r effective_model_device=%r "
                        "CUDA_VISIBLE_DEVICES=%r torch.cuda.device_count=%r torch.cuda.current_device=%r torch.cuda.name=%r "
                        "ray_accelerator_ids=%r prm_model=%r",
                        str(cls.verifier_cfg.prm_device),
                        prm_model_device,
                        cuda_visible,
                        torch_dev_cnt,
                        torch_cur,
                        torch_name,
                        ray_accel,
                        prm_path,
                    )
                    print(
                        "[StepVerifyAgentLoop] PRM loaded. "
                        f"requested_prm_device={cls.verifier_cfg.prm_device} effective_model_device={prm_model_device} "
                        f"CUDA_VISIBLE_DEVICES={cuda_visible} torch.cuda.device_count={torch_dev_cnt} "
                        f"torch.cuda.current_device={torch_cur} torch.cuda.name={torch_name} "
                        f"ray_accelerator_ids={ray_accel} prm_model={prm_path}",
                        flush=True,
                    )
                except Exception:
                    pass
            except Exception as e:
                raise RuntimeError(f"Failed to load PRM model: {e}") from e

        # Periodic sample logging configuration
        cls._log_sample_interval = int(step_cfg.get("log_sample_interval", 50))
        cls._sample_counter = 0
        if cls._log_sample_interval > 0:
            logger.warning(
                "StepVerifyAgentLoop: periodic sample logging enabled (log_sample_interval=%d)",
                cls._log_sample_interval
            )

    def _judge_trace_path(self) -> str:
        # One file per worker process (pid) to avoid cross-process write contention.
        # Allow a fully specified path override.
        override = getattr(self.verifier_cfg, "judge_trace_output_path", None)
        if override:
            try:
                override = str(override).format(pid=os.getpid(), host=socket.gethostname())
            except Exception:
                override = str(override)
            try:
                os.makedirs(os.path.dirname(override) or ".", exist_ok=True)
            except Exception:
                pass
            return override

        out_dir = str(getattr(self.verifier_cfg, "judge_trace_output_dir", "judge_traces") or "judge_traces")
        try:
            out_dir = os.path.expanduser(out_dir)
        except Exception:
            pass
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            out_dir = "."
        prefix = str(getattr(self.verifier_cfg, "judge_trace_filename_prefix", "judge_trace") or "judge_trace")
        return os.path.join(out_dir, f"{prefix}_{socket.gethostname()}_{os.getpid()}.jsonl")

    def _ensure_judge_trace_file(self) -> None:
        """Create the JSONL file early so users can find it even before any valid samples are written."""
        if not bool(getattr(self.verifier_cfg, "judge_trace_enable", False)):
            return
        try:
            path = self._judge_trace_path()
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            # Touch
            with open(path, "a", encoding="utf-8") as f:
                if bool(getattr(self.verifier_cfg, "judge_trace_flush", True)):
                    f.flush()
            logger.warning("Judge trace enabled. Writing to: %s", path)
        except Exception as e:
            logger.warning("Judge trace enabled but failed to create output file (ignored): %s", e)

    def _maybe_write_judge_trace(self, record: dict[str, Any]) -> None:
        if not bool(getattr(self.verifier_cfg, "judge_trace_enable", False)):
            return
        try:
            import json

            path = self._judge_trace_path()
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                if bool(getattr(self.verifier_cfg, "judge_trace_flush", True)):
                    f.flush()
                    if bool(getattr(self.verifier_cfg, "judge_trace_fsync", False)):
                        os.fsync(f.fileno())
        except Exception as e:
            # Never crash training due to debug logging.
            logger.warning("Failed to write judge trace (ignored): %s", e)

    def _maybe_write_judge_trace_many(self, records: list[dict[str, Any]]) -> None:
        if not bool(getattr(self.verifier_cfg, "judge_trace_enable", False)):
            return
        if not records:
            return
        try:
            import json

            path = self._judge_trace_path()
            with open(path, "a", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if bool(getattr(self.verifier_cfg, "judge_trace_flush", True)):
                    f.flush()
                    if bool(getattr(self.verifier_cfg, "judge_trace_fsync", False)):
                        os.fsync(f.fileno())
        except Exception as e:
            logger.warning("Failed to write judge trace batch (ignored): %s", e)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # Ensure trace file exists early (helps debugging paths / node-local disks).
        self._ensure_judge_trace_file()
        messages: list[dict[str, Any]] = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info") or {}
        # Multi-turn user inputs compatibility:
        # - preferred: extra_info.subproblems
        # - legacy: extra_info.backward_hints
        # - legacy(Interaction format): extra_info.interaction_kwargs.backward_hints
        subproblems: Optional[list[str]] = extra_info.get("subproblems")
        if subproblems is None:
            subproblems = extra_info.get("backward_hints")
        if subproblems is None:
            interaction_kwargs = extra_info.get("interaction_kwargs") or {}
            subproblems = interaction_kwargs.get("backward_hints")
        # normalize
        if subproblems is not None and not isinstance(subproblems, list):
            subproblems = None
        expected_answers = extra_info.get("expected_answers")
        reward_model = kwargs.get("reward_model") or {}
        ground_truth = reward_model.get("ground_truth") or extra_info.get("ground_truth")
        # IMPORTANT:
        # Some datasets put the first subproblem in raw_prompt / subproblems[0].
        # For step LLM judge, users may want the ORIGINAL problem only.
        # So we compute `original_problem_text` without falling back to subproblems[0].
        original_problem_text = (
            extra_info.get("problem_text")
            or extra_info.get("question")
            or extra_info.get("question_hint")
            or ""
        )
        # Keep legacy behavior for final judging/logging: if original_problem_text is empty, fall back to subproblems[0].
        problem_text = original_problem_text
        if not str(problem_text or "").strip():
            problem_text = (subproblems[0] if isinstance(subproblems, list) and subproblems else "") or ""

        request_id = uuid4().hex
        metrics: dict[str, Any] = {}

        prompt_ids = await self._encode_full_messages(messages)

        response_mask: list[int] = []
        response_logprobs: list[float] = []
        step_token_spans: list[tuple[int, int]] = []
        turn_scores: list[float] = []
        # Optional component scores for debugging / recombination
        turn_scores_rule: list[Optional[float]] = []
        turn_scores_llm: list[Optional[float]] = []
        turn_scores_prm: list[Optional[float]] = []

        assistant_turns = 0
        user_turns = 0
        assistant_turn_texts: list[str] = []
        first_answer_marker_turn: Optional[int] = None
        judge_trace_steps: list[dict[str, Any]] = []

        # subproblem pointer: we assume subproblems[0] is already in raw_prompt if provided
        next_subproblem_idx = 1 if subproblems else 0

        while True:
            # 1) model generate
            with simple_timer("generate_sequences", metrics):
                out = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, image_data=None
                )

            assistant_turns += 1
            gen_ids = out.token_ids
            start = len(response_mask)
            end = start + len(gen_ids)
            step_token_spans.append((start, end))

            prompt_ids += gen_ids
            response_mask += [1] * len(gen_ids)
            if out.log_probs:
                response_logprobs += out.log_probs

            # 2) append assistant message for context
            assistant_text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            )
            messages.append({"role": "assistant", "content": assistant_text})
            assistant_turn_texts.append(str(assistant_text or ""))
            if first_answer_marker_turn is None and _has_answer_marker(assistant_text):
                first_answer_marker_turn = assistant_turns - 1  # 0-based assistant turn index

            # 3) verify this step (hybrid)
            step_ctx = {
                "step_idx": len(turn_scores),
                "messages": messages,
                "problem_text": str(original_problem_text or ""),
                "question": (subproblems[len(turn_scores)] if subproblems and len(turn_scores) < len(subproblems) else None),
                "expected": (
                    expected_answers[len(turn_scores)]
                    if isinstance(expected_answers, list) and len(turn_scores) < len(expected_answers)
                    else None
                ),
            }
            verdict = await self._verify_step(request_id, assistant_text, step_ctx, sampling_params)
            turn_scores.append(float(verdict["final"]))
            turn_scores_rule.append(verdict.get("rule"))
            turn_scores_llm.append(verdict.get("llm"))
            turn_scores_prm.append(verdict.get("prm"))

            # Optional: record per-step judge trace (subproblem, answer, judge raw+score).
            if bool(getattr(self.verifier_cfg, "judge_trace_enable", False)):
                max_raw = int(getattr(self.verifier_cfg, "judge_trace_max_raw_chars", 4000) or 4000)
                llm_raw = step_ctx.get("_llm_judge_raw") or ""
                if isinstance(llm_raw, str) and max_raw > 0 and len(llm_raw) > max_raw:
                    llm_raw = llm_raw[:max_raw]
                judge_trace_steps.append(
                    {
                        "step_idx": int(step_ctx.get("step_idx", len(judge_trace_steps))),
                        "subproblem": step_ctx.get("question"),
                        "assistant_answer": str(assistant_text or ""),
                        "rule_score": verdict.get("rule"),
                        "prm_score": verdict.get("prm"),
                        "llm_judge": {
                            "used": bool(step_ctx.get("_llm_judge_used", False)),
                            "parse_ok": bool(step_ctx.get("_llm_judge_parse_ok", False)),
                            "score": step_ctx.get("_llm_judge_score"),
                            "raw": llm_raw,
                        },
                        "final_step_score": verdict.get("final"),
                    }
                )

                # If the goal is to build judge training data, write one JSONL line per step immediately.
                fmt = str(getattr(self.verifier_cfg, "judge_trace_format", "trajectory") or "trajectory").lower()
                if fmt == "judge_train_steps":
                    label_src = str(getattr(self.verifier_cfg, "judge_train_label", "final") or "final").lower()
                    # Build history up to (but excluding) current step
                    hist: list[dict[str, Any]] = []
                    for prev in judge_trace_steps[:-1]:
                        hist.append(
                            {
                                "subquestion": prev.get("subproblem"),
                                "step_solution": prev.get("assistant_answer"),
                            }
                        )
                    cur = judge_trace_steps[-1]
                    # Choose label (only record strict binary 0/1; skip non-binary/non-parse-ok).
                    if label_src == "llm":
                        lj = cur.get("llm_judge", {}) or {}
                        if not bool(lj.get("parse_ok", False)):
                            pass
                        else:
                            lab = lj.get("score")
                            if lab is not None:
                                lab = float(lab)
                                if abs(lab - 0.0) < 1e-9:
                                    self._maybe_write_judge_trace(
                                        {
                                            "ts": time.time(),
                                            "request_id": request_id,
                                            "agent": self.__class__.__name__,
                                            "index": kwargs.get("index", None),
                                            "problem": str(problem_text or ""),
                                            "history": hist,
                                            "subquestion": cur.get("subproblem"),
                                            "step_solution": cur.get("assistant_answer"),
                                            "label": 0,
                                        }
                                    )
                                elif abs(lab - 1.0) < 1e-9:
                                    self._maybe_write_judge_trace(
                                        {
                                            "ts": time.time(),
                                            "request_id": request_id,
                                            "agent": self.__class__.__name__,
                                            "index": kwargs.get("index", None),
                                            "problem": str(problem_text or ""),
                                            "history": hist,
                                            "subquestion": cur.get("subproblem"),
                                            "step_solution": cur.get("assistant_answer"),
                                            "label": 1,
                                        }
                                    )
                    else:
                        if label_src == "rule":
                            lab = cur.get("rule_score")
                        else:  # final
                            lab = cur.get("final_step_score")
                        if lab is not None:
                            lab = float(lab)
                            if abs(lab - 0.0) < 1e-9 or abs(lab - 1.0) < 1e-9:
                                self._maybe_write_judge_trace(
                                    {
                                        "ts": time.time(),
                                        "request_id": request_id,
                                        "agent": self.__class__.__name__,
                                        "index": kwargs.get("index", None),
                                        "problem": str(problem_text or ""),
                                        "history": hist,
                                        "subquestion": cur.get("subproblem"),
                                        "step_solution": cur.get("assistant_answer"),
                                        "label": 0 if abs(lab - 0.0) < 1e-9 else 1,
                                    }
                                )

            # termination: token budget / turn limits
            if len(response_mask) >= self.response_length:
                break
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                break
            if self.max_user_turns and user_turns >= self.max_user_turns:
                break

            # termination: subproblems exhausted
            if not subproblems or next_subproblem_idx >= len(subproblems):
                break

            # 4) add next subproblem as user message
            next_user = subproblems[next_subproblem_idx]
            next_subproblem_idx += 1
            add_messages = [{"role": "user", "content": next_user}]
            messages.extend(add_messages)
            user_turns += 1

            user_ids = await self._encode_incremental_messages(add_messages)
            if len(response_mask) + len(user_ids) >= self.response_length:
                break
            prompt_ids += user_ids
            response_mask += [0] * len(user_ids)
            if response_logprobs:
                response_logprobs += [0.0] * len(user_ids)

        # 5) final reward (independent): only depends on final correctness
        final_reward = 0.0
        if self.verifier_cfg.final_enable and ground_truth is not None:
            last_assistant = _extract_last_assistant_text(messages)
            final_reward = await self._verify_final(
                request_id=request_id,
                problem_text=str(problem_text),
                last_assistant_text=str(last_assistant),
                ground_truth=str(ground_truth),
                sampling_params=sampling_params,
            )
            final_reward *= float(self.verifier_cfg.final_weight)

        # Optional PRM "final_once" evaluation: compute PRM scores once over the whole problem and map back to turns.
        if bool(getattr(self.verifier_cfg, "prm_enable", False)) and (self.verifier_cfg.prm_eval_mode or "").lower() == "final_once":
            try:
                # Use the overall problem text for problem-level PRM judging.
                assistant_turn_texts = [m.get("content", "") for m in messages if m.get("role") == "assistant"]
                prm_turn_scores = await self._prm_episode_turn_scores(problem_text=str(problem_text), assistant_turn_texts=assistant_turn_texts)

                # Align length
                if len(prm_turn_scores) < len(turn_scores):
                    prm_turn_scores = prm_turn_scores + [None] * (len(turn_scores) - len(prm_turn_scores))
                if len(prm_turn_scores) > len(turn_scores):
                    prm_turn_scores = prm_turn_scores[: len(turn_scores)]
                turn_scores_prm = prm_turn_scores

                # If combine uses PRM, recompute final per-turn scores using the same combine logic.
                combine = (self.verifier_cfg.combine or "max").lower()
                if combine in {"max", "mean", "prm_only"}:
                    new_scores: list[float] = []
                    for rs, ls, ps in zip(turn_scores_rule, turn_scores_llm, turn_scores_prm, strict=False):
                        if combine == "prm_only":
                            final_s = ps if ps is not None else 0.0
                        elif combine == "mean":
                            vals = [v for v in [rs, ls, ps] if v is not None]
                            final_s = (sum(vals) / len(vals)) if vals else 0.0
                        else:  # max
                            final_s = max([v for v in [rs, ls, ps] if v is not None], default=0.0)
                        new_scores.append(float(max(0.0, min(1.0, final_s))))
                    turn_scores = new_scores
            except Exception as e:
                logger.warning("PRM final_once evaluation failed: %s", e)

        # Optional step reward scaling: make total process reward bounded (e.g., <= 0.3) regardless of #turns.
        if bool(getattr(self.verifier_cfg, "step_scale_by_num_turns", False)):
            try:
                total_w = float(getattr(self.verifier_cfg, "step_total_weight", 1.0))
                n_turns = max(1, len(turn_scores))
                # Each raw turn_score is in [0,1]. After scaling, sum(turn_scores) <= total_w.
                turn_scores = [float(max(0.0, min(1.0, s))) * (total_w / float(n_turns)) for s in turn_scores]
            except Exception as e:
                logger.warning("step reward scaling failed, keep raw turn_scores: %s", e)

        # Optional: dump one JSONL record per trajectory for judge trace analysis.
        if bool(getattr(self.verifier_cfg, "judge_trace_enable", False)):
            try:
                max_raw = int(getattr(self.verifier_cfg, "judge_trace_max_raw_chars", 4000) or 4000)
                final_raw = str(getattr(self, "_final_judge_raw", "") or "")
                if max_raw > 0 and len(final_raw) > max_raw:
                    final_raw = final_raw[:max_raw]
                fmt = str(getattr(self.verifier_cfg, "judge_trace_format", "trajectory") or "trajectory").lower()
                if fmt == "judge_train_steps":
                    # NOTE: For judge_train_steps we write step lines in real-time inside the loop.
                    # So we intentionally skip any end-of-trajectory dump here to avoid duplicates.
                    pass
                else:
                    rec: dict[str, Any] = {
                        "ts": time.time(),
                        "request_id": request_id,
                        "agent": self.__class__.__name__,
                        "problem_text": str(problem_text or ""),
                        "steps": judge_trace_steps,
                        "final": {
                            "reward": float(final_reward),
                            "raw": final_raw,
                            "score": float(getattr(self, "_final_judge_score", 0.0) or 0.0),
                        },
                        "index": kwargs.get("index", None),
                    }
                    self._maybe_write_judge_trace(rec)
            except Exception as e:
                logger.warning("Failed to dump judge trace (ignored): %s", e)

        # Build AgentLoopOutput
        response_ids = prompt_ids[-len(response_mask) :] if response_mask else []
        prompt_only_ids = prompt_ids[: len(prompt_ids) - len(response_mask)] if response_mask else prompt_ids

        # Debug payload: record how many turns we actually generated and a few sample assistant answers.
        # (Truncate to keep JSON small.)
        samples: list[dict[str, Any]] = []
        if assistant_turn_texts:
            take_first = min(3, len(assistant_turn_texts))
            for i in range(take_first):
                samples.append({"turn": i, "text": assistant_turn_texts[i][:300]})
            if len(assistant_turn_texts) > take_first:
                samples.append(
                    {"turn": len(assistant_turn_texts) - 1, "text": assistant_turn_texts[-1][:300]}
                )

        output = AgentLoopOutput(
            prompt_ids=prompt_only_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            multi_modal_data={},
            # IMPORTANT:
            # Set a dummy reward_score to avoid triggering async reward computation in AgentLoopWorkerBase
            # when reward_model.enable is False. Token-level rewards are carried via rm_scores constructed
            # from extra_fields (turn_scores + step_token_spans + final_reward).
            reward_score=0.0,
            num_turns=assistant_turns + user_turns + 1,  # +1 for initial user in raw_prompt
            metrics=metrics,
            extra_fields={
                "turn_scores": turn_scores,
                "turn_scores_rule": turn_scores_rule,
                "turn_scores_llm": turn_scores_llm,
                "turn_scores_prm": turn_scores_prm,
                "step_token_spans": step_token_spans,
                "final_reward": float(final_reward),
                # For proof-style final judging in subclasses: keep assistant turns for concatenation.
                "assistant_turn_texts": assistant_turn_texts,
                "assistant_turns": int(assistant_turns),
                "user_turns": int(user_turns),
                "first_answer_marker_turn": first_answer_marker_turn,
                "assistant_turn_samples": samples,
            },
        )

        # Periodic sample logging for inspection
        self._maybe_log_sample(
            messages=messages,
            extra_info=extra_info,
            reward_model=reward_model,
            output=output,
            subproblems=subproblems,
        )

        return output

    def _maybe_log_sample(
        self,
        messages: list[dict[str, Any]],
        extra_info: dict[str, Any],
        reward_model: dict[str, Any],
        output: AgentLoopOutput,
        subproblems: Optional[list[str]],
    ) -> None:
        """Log a full multi-turn conversation sample at configured intervals."""
        interval = getattr(self.__class__, "_log_sample_interval", 50)
        if interval <= 0:
            return

        # Thread-safe counter increment
        with self.__class__._counter_lock:
            self.__class__._sample_counter += 1
            current_count = self.__class__._sample_counter

        # Check if we should log this sample
        if current_count % interval != 0:
            return

        try:
            # Get problem text
            problem_text = (
                extra_info.get("problem_text")
                or extra_info.get("question")
                or extra_info.get("question_hint")
                or "(no problem text)"
            )

            # Get ground truth
            ground_truth = reward_model.get("ground_truth") or extra_info.get("ground_truth") or "(no ground truth)"

            # Get assistant turns and scores
            assistant_turn_texts = output.extra_fields.get("assistant_turn_texts", [])
            turn_scores = output.extra_fields.get("turn_scores", [])
            final_reward = output.extra_fields.get("final_reward", None)

            # Format messages for logging
            log_lines = []
            log_lines.append("\n" + "=" * 80)
            log_lines.append(f"SAMPLE #{current_count} - MULTI-TURN CONVERSATION")
            log_lines.append("=" * 80)
            log_lines.append(f"Problem: {str(problem_text)[:500]}...")
            log_lines.append(f"Ground Truth: {str(ground_truth)[:200]}")
            log_lines.append(f"Turn Scores: {turn_scores}")
            log_lines.append(f"Final Reward: {final_reward}")
            log_lines.append(f"Num Assistant Turns: {len(assistant_turn_texts)}")
            log_lines.append("-" * 80)

            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown").upper()
                content = str(msg.get("content", ""))
                log_lines.append(f"\n[{i}] {role}:")
                # Truncate very long content
                if len(content) > 1500:
                    content = content[:1500] + "\n... (truncated)"
                log_lines.append(content)

            log_lines.append("\n" + "=" * 80 + "\n")

            logger.warning("\n".join(log_lines))

        except Exception as e:
            logger.warning("Failed to log sample #%d: %s", current_count, e)

    async def _encode_full_messages(self, messages: list[dict[str, Any]]) -> list[int]:
        # no multimodal for now; can be extended similarly to ToolAgentLoop
        return await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
            ),
        )

    async def _encode_incremental_messages(self, add_messages: list[dict[str, Any]]) -> list[int]:
        # Encode only the new turn, then strip system prefix (same pattern as ToolAgentLoop)
        ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
        )
        return ids[len(self.system_prompt) :]

    async def _verify_step(
        self,
        request_id: str,
        assistant_text: str,
        step_ctx: dict[str, Any],
        sampling_params: dict[str, Any],
    ) -> dict[str, Optional[float]]:
        rule_score: Optional[float] = None
        if self.rule_verifier is not None:
            try:
                rule_score = self.rule_verifier(assistant_text, step_ctx)
            except Exception as e:
                logger.warning("rule_verifier failed: %s", e)
                rule_score = None

        # PRM score (optional)
        prm_score: Optional[float] = None
        if bool(getattr(self.verifier_cfg, "prm_enable", False)) and (self.verifier_cfg.prm_eval_mode or "").lower() != "final_once":
            try:
                prm_score = await self._prm_step_score(step_ctx=step_ctx, assistant_text=assistant_text)
            except Exception as e:
                logger.warning("PRM step scoring failed: %s", e)
                prm_score = None

        need_llm = self.verifier_cfg.llm_enable and (
            self.verifier_cfg.llm_always
            or (rule_score is None and self.verifier_cfg.llm_on_none)
            or (rule_score is not None and rule_score <= 0.0 and self.verifier_cfg.llm_on_incorrect)
        )

        llm_score: Optional[float] = None
        if need_llm:
            llm_score = await self._llm_verify(request_id, assistant_text, step_ctx, sampling_params)

        # Combine
        combine = self.verifier_cfg.combine
        if combine == "rule_only":
            final = rule_score if rule_score is not None else 0.0
        elif combine == "llm_only":
            final = llm_score if llm_score is not None else 0.0
        elif combine == "prm_only":
            final = prm_score if prm_score is not None else 0.0
        elif combine == "mean":
            scores = [s for s in [rule_score, llm_score, prm_score] if s is not None]
            final = sum(scores) / len(scores) if scores else 0.0
        else:  # "max" default
            final = max([s for s in [rule_score, llm_score, prm_score] if s is not None], default=0.0)

        # clamp
        final_f = float(max(0.0, min(1.0, final)))
        return {"final": final_f, "rule": rule_score, "llm": llm_score, "prm": prm_score}

    async def _prm_step_score(self, *, step_ctx: dict[str, Any], assistant_text: str) -> float:
        """
        Use a PRM model (e.g. Qwen/Qwen2.5-Math-7B-PRM800K) to produce a step reward in [0,1].
        We build a chat prompt and put the assistant content as concatenated "steps" separated by prm_step_sep_token,
        then extract the positive-class probability at each separator token position. We use the LAST separator prob
        as the reward for this agent-loop turn (aligned with turn_scores being per assistant turn).
        """
        cls = self.__class__
        if cls._prm_model is None or cls._prm_tokenizer is None:
            raise RuntimeError("PRM enabled but PRM model/tokenizer not loaded.")

        sep = str(self.verifier_cfg.prm_step_sep_token or "<extra_0>")
        split_strategy = (self.verifier_cfg.prm_split_strategy or "auto").lower()

        # If assistant_text already contains sep token, use it as step boundaries.
        steps: list[str]
        if sep in (assistant_text or ""):
            parts = [p for p in (assistant_text or "").split(sep) if p.strip()]
            steps = parts if parts else [assistant_text or ""]
        elif split_strategy == "newline":
            steps = [ln.strip() for ln in (assistant_text or "").splitlines() if ln.strip()]
            steps = steps if steps else [assistant_text or ""]
        else:
            # "auto" or "single"
            steps = [assistant_text or ""]

        question = step_ctx.get("question") or ""
        messages = [
            {"role": "system", "content": "You are a process reward model."},
            {"role": "user", "content": str(question)},
            {"role": "assistant", "content": sep.join(steps) + sep},
        ]

        conversation_str = await self.loop.run_in_executor(
            None,
            lambda: cls._prm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False),
        )

        import torch

        input_ids = await self.loop.run_in_executor(
            None,
            lambda: cls._prm_tokenizer.encode(conversation_str, return_tensors="pt"),
        )
        model = cls._prm_model
        try:
            input_ids = input_ids.to(model.device)
        except Exception:
            input_ids = input_ids.to(next(model.parameters()).device)

        # Identify separator token positions
        sep_ids = cls._prm_tokenizer.encode(sep)
        if not sep_ids:
            return 0.0
        step_sep_id = int(sep_ids[0])
        token_mask = (input_ids == step_sep_id)
        if token_mask.sum().item() <= 0:
            return 0.0

        with torch.no_grad():
            # Disable KV cache to avoid incompatibilities between transformers cache objects and
            # some PRM models' forward implementations (e.g., DynamicCache missing attributes).
            outputs = model(input_ids=input_ids, use_cache=False)
            logits = outputs[0]
            probs = torch.softmax(logits, dim=-1)
            # Expect 2 labels; take label=1 as "positive" step score.
            if probs.size(-1) < 2:
                return 0.0
            pos = probs[..., 1]
            vals = pos[token_mask].detach()
            if vals.numel() <= 0:
                return 0.0
            return float(vals[-1].item())

    async def _prm_episode_turn_scores(self, *, problem_text: str, assistant_turn_texts: list[str]) -> list[Optional[float]]:
        """
        PRM "final_once" mode.
        Build ONE PRM input for the whole problem and all assistant turns, then return a list of per-turn scores
        aligned to assistant_turn_texts.

        Note:
        - For stable alignment we treat EACH assistant turn as ONE "step" (i.e., we append exactly one sep token per turn).
        - If you want multiple steps per turn, prefer prm_eval_mode="per_turn" with prm_split_strategy.
        """
        cls = self.__class__
        if cls._prm_model is None or cls._prm_tokenizer is None:
            raise RuntimeError("PRM enabled but PRM model/tokenizer not loaded.")

        sep = str(self.verifier_cfg.prm_step_sep_token or "<extra_0>")
        # Force per-turn single-step packing for alignment
        packed = sep.join([str(t or "") for t in assistant_turn_texts]) + sep
        messages = [
            {"role": "system", "content": "You are a process reward model."},
            {"role": "user", "content": str(problem_text or "")},
            {"role": "assistant", "content": packed},
        ]
        conversation_str = await self.loop.run_in_executor(
            None,
            lambda: cls._prm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False),
        )

        import torch

        input_ids = await self.loop.run_in_executor(
            None,
            lambda: cls._prm_tokenizer.encode(conversation_str, return_tensors="pt"),
        )
        model = cls._prm_model
        try:
            input_ids = input_ids.to(model.device)
        except Exception:
            input_ids = input_ids.to(next(model.parameters()).device)

        sep_ids = cls._prm_tokenizer.encode(sep)
        if not sep_ids:
            return [None for _ in assistant_turn_texts]
        step_sep_id = int(sep_ids[0])
        token_mask = (input_ids == step_sep_id)
        with torch.no_grad():
            # Disable KV cache to avoid incompatibilities between transformers cache objects and
            # some PRM models' forward implementations (e.g., DynamicCache missing attributes).
            outputs = model(input_ids=input_ids, use_cache=False)
            logits = outputs[0]
            probs = torch.softmax(logits, dim=-1)
            if probs.size(-1) < 2:
                return [None for _ in assistant_turn_texts]
            pos = probs[..., 1]
            vals = pos[token_mask].detach()
            # Expect one sep per assistant turn (plus maybe others if tokenizer splits weirdly).
            # Use first N values.
            vals_list = vals.cpu().tolist() if vals.numel() > 0 else []
            out: list[Optional[float]] = [float(x) for x in vals_list[: len(assistant_turn_texts)]]
            if len(out) < len(assistant_turn_texts):
                out.extend([None] * (len(assistant_turn_texts) - len(out)))
            return out

    async def _llm_verify(
        self,
        request_id: str,
        assistant_text: str,
        step_ctx: dict[str, Any],
        sampling_params: dict[str, Any],
    ) -> float:
        # Safety: if llm_enable is false, never call any judge backend.
        if not bool(getattr(self.verifier_cfg, "llm_enable", False)):
            return 0.0
        # Prompt formatting:
        # - Keep backward compatibility with existing templates that use {question}/{answer}.
        # - Also support richer templates that use fields like:
        #   {problem}, {subquestion}, {step_solution}.
        #
        # IMPORTANT: `problem_text` is "original problem only" (no fallback to subproblems[0]),
        # so you won't accidentally feed the first subproblem as the problem.
        problem_text = step_ctx.get("problem_text") or ""
        subquestion_text = step_ctx.get("question") or ""
        # `{question}` slot can be configured to mean either the original problem or the current subquestion.
        question_text = problem_text if bool(getattr(self.verifier_cfg, "llm_use_problem_text", False)) else subquestion_text

        fmt_vars = {
            # legacy / common
            "question": str(question_text or ""),
            "answer": str(assistant_text or ""),
            # explicit, recommended names
            "problem": str(problem_text or ""),
            "subquestion": str(subquestion_text or ""),
            "step_solution": str(assistant_text or ""),
        }
        try:
            prompt = str(self.verifier_cfg.llm_prompt_template).format(**fmt_vars)
        except KeyError as e:
            # Fail-soft: keep old behavior if user template references unknown keys.
            logger.warning("llm_prompt_template has unknown placeholder %r; falling back to {question}/{answer} only.", e)
            prompt = str(self.verifier_cfg.llm_prompt_template).format(
                question=str(question_text or ""),
                answer=str(assistant_text or ""),
            )
        verifier_messages = [
            {"role": "system", "content": str(getattr(self.verifier_cfg, "llm_system_prompt", "You are a verifier.") or "")},
            {"role": "user", "content": prompt},
        ]
        verifier_prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                verifier_messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
            ),
        )

        verifier_sampling = dict(sampling_params)
        # vLLM rollout server expects OpenAI-style `max_tokens` (NOT `max_new_tokens`).
        verifier_sampling.pop("max_new_tokens", None)
        verifier_sampling["max_tokens"] = int(self.verifier_cfg.llm_max_new_tokens)
        verifier_sampling["temperature"] = float(getattr(self.verifier_cfg, "llm_temperature", 0.0))
        verifier_sampling["top_p"] = float(getattr(self.verifier_cfg, "llm_top_p", 1.0))

        # important: use a different request_id so sticky-session KV doesn't mix with policy dialogue
        text = await self._judge_completion_text(
            request_id=f"verifier_{request_id}_{step_ctx.get('step_idx', 0)}",
            messages=verifier_messages,
            sampling_params=verifier_sampling,
        )
        # Attach judge raw output for optional tracing.
        try:
            step_ctx["_llm_judge_used"] = True
            step_ctx["_llm_judge_raw"] = str(text or "")
        except Exception:
            pass
        score = _parse_binary_judge(text or "")
        if score is None:
            logger.warning("LLM step judge returned non-binary output, fallback to 0: %r", (text or "")[:200])
            try:
                step_ctx["_llm_judge_parse_ok"] = False
                # Keep training behavior (return 0) but do NOT record a fake 0 label for dataset dumping.
                step_ctx["_llm_judge_score"] = None
            except Exception:
                pass
            return 0.0
        try:
            step_ctx["_llm_judge_parse_ok"] = True
            step_ctx["_llm_judge_score"] = float(score)
        except Exception:
            pass
        return float(score)

    async def _verify_final(
        self,
        *,
        request_id: str,
        problem_text: str,
        last_assistant_text: str,
        ground_truth: str,
        sampling_params: dict[str, Any],
    ) -> float:
        """
        Final reward: independent from step rewards.
        Supports two modes:
        - answer mode: extract final answer token (<Answer>/####/\\boxed{}) and exact-match to ground_truth.
        - proof mode: judge the full proof/solution correctness via LLM (no extraction).
        """
        mode = (self.verifier_cfg.final_mode or "answer").lower()
        if mode not in {"answer", "proof", "auto"}:
            logger.warning("Unknown final_mode=%r, fallback to 'answer'", mode)
            mode = "answer"

        if mode == "auto":
            gt = (ground_truth or "").strip()
            # Heuristic: short ground truth (<= 64 chars) that contains digits/symbols is likely an "answer" task.
            if gt and len(gt) <= 64:
                mode = "answer"
            else:
                mode = "proof"

        rule_score: Optional[float] = None
        if mode == "answer":
            extracted = _extract_final_answer(last_assistant_text)
            if extracted is not None and ground_truth is not None:
                a = str(extracted).strip()
                gt = str(ground_truth).strip()
                rule_score = 1.0 if a == gt else 0.0

        llm_score: Optional[float] = None
        if mode == "proof":
            # Proof tasks require a judge; if not enabled, we can only fall back to 0.
            if not self.verifier_cfg.final_llm_enable:
                logger.warning("final_mode='proof' but final_llm_enable is False; final reward will be 0.")
                llm_score = None
            else:
                prompt = self.verifier_cfg.final_llm_prompt_template_proof.format(
                    problem=problem_text, answer=last_assistant_text, ground_truth=ground_truth
                )
        else:
            if self.verifier_cfg.final_llm_enable:
                prompt = self.verifier_cfg.final_llm_prompt_template_answer.format(
                    problem=problem_text, answer=last_assistant_text, ground_truth=ground_truth
                )
            else:
                prompt = None

        if prompt is not None and self.verifier_cfg.final_llm_enable:
            verifier_messages = [
                {"role": "system", "content": "You are a verifier."},
                {"role": "user", "content": prompt},
            ]
            verifier_prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    verifier_messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                ),
            )
            verifier_sampling = dict(sampling_params)
            # vLLM rollout server expects OpenAI-style `max_tokens` (NOT `max_new_tokens`).
            verifier_sampling.pop("max_new_tokens", None)
            verifier_sampling["max_tokens"] = int(self.verifier_cfg.final_llm_max_new_tokens)
            verifier_sampling["temperature"] = float(getattr(self.verifier_cfg, "final_llm_temperature", 0.0))
            verifier_sampling["top_p"] = float(getattr(self.verifier_cfg, "final_llm_top_p", 1.0))
            text = await self._judge_completion_text(
                request_id=f"final_verifier_{request_id}",
                messages=verifier_messages,
                sampling_params=verifier_sampling,
                backend_override=getattr(self.verifier_cfg, "final_judge_backend", None),
                model_name_or_path_override=getattr(self.verifier_cfg, "final_judge_model_name_or_path", None),
                base_url_override=getattr(self.verifier_cfg, "final_judge_base_url", None),
                api_key_override=getattr(self.verifier_cfg, "final_judge_api_key", None),
                timeout_s_override=getattr(self.verifier_cfg, "final_judge_timeout_s", None),
            )
            # Help debugging: empty output can happen even on HTTP 200 (e.g., refusal/tool-call/structured content).
            if not str(text or "").strip():
                backend_dbg = (getattr(self.verifier_cfg, "final_judge_backend", None) or self.verifier_cfg.judge_backend)
                model_dbg = (
                    getattr(self.verifier_cfg, "final_judge_model_name_or_path", None)
                    or self.verifier_cfg.judge_model_name_or_path
                )
                base_dbg = getattr(self.verifier_cfg, "final_judge_base_url", None) or self.verifier_cfg.judge_base_url
                logger.warning(
                    "Final judge returned empty text. request_id=%s backend=%s model=%s base_url=%s",
                    request_id,
                    backend_dbg,
                    model_dbg,
                    base_dbg,
                )
            llm_score = _parse_binary_judge(text or "")
            if llm_score is None:
                logger.warning("LLM final judge returned non-binary output, fallback to 0: %r", (text or "")[:200])
                llm_score = 0.0
            # Save raw/score for optional judge trace dumping.
            try:
                self._final_judge_raw = str(text or "")
                self._final_judge_score = float(llm_score)
            except Exception:
                pass

        # Combine final score
        if mode == "proof":
            final = llm_score if llm_score is not None else 0.0
        else:
            # answer mode: prefer rule, allow LLM to rescue
            if llm_score is None:
                final = rule_score if rule_score is not None else 0.0
            else:
                final = max(rule_score or 0.0, llm_score)
        return float(max(0.0, min(1.0, final)))

    async def _judge_completion_text(
        self,
        *,
        request_id: str,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
        backend_override: Optional[str] = None,
        model_name_or_path_override: Optional[str] = None,
        base_url_override: Optional[str] = None,
        api_key_override: Optional[str] = None,
        timeout_s_override: Optional[float] = None,
    ) -> str:
        """
        Run the judge model and return decoded text.
        - policy backend: uses server_manager.generate (OpenAI-compatible rollout server)
        - local backend: uses a separate transformers model loaded in this worker process
        - remote backend: calls an external OpenAI-compatible HTTP endpoint (e.g., vLLM OpenAI server)
        """
        backend = (backend_override or self.verifier_cfg.judge_backend or "policy").lower()
        if backend == "local":
            if self._judge_model is None or self._judge_tokenizer is None:
                raise RuntimeError("Local judge backend requested but judge model/tokenizer not loaded.")

            # Build prompt text using judge tokenizer chat template if available; fallback to simple concat.
            def build_text():
                tok = self._judge_tokenizer
                if hasattr(tok, "apply_chat_template"):
                    try:
                        return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    except Exception:
                        pass
                # simple fallback
                parts = []
                for m in messages:
                    parts.append(f"[{m.get('role','user').upper()}] {m.get('content','')}")
                parts.append("[ASSISTANT] ")
                return "\n".join(parts)

            prompt_text = await self.loop.run_in_executor(None, build_text)
            import torch

            inputs = self._judge_tokenizer(prompt_text, return_tensors="pt")
            device = self.verifier_cfg.judge_device.lower()
            if device != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}

            # Use deterministic decoding for judge
            # Local transformers backend uses max_new_tokens; accept either field.
            max_new = int(sampling_params.get("max_new_tokens") or sampling_params.get("max_tokens") or 128)
            with torch.no_grad():
                out_ids = self._judge_model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,
                    pad_token_id=getattr(self._judge_tokenizer, "pad_token_id", None)
                    or getattr(self._judge_tokenizer, "eos_token_id", None),
                    eos_token_id=getattr(self._judge_tokenizer, "eos_token_id", None),
                )
            gen = out_ids[0][inputs["input_ids"].shape[1] :]
            return self._judge_tokenizer.decode(gen, skip_special_tokens=True)

        if backend == "remote":
            import json

            import aiohttp

            base = str((base_url_override or self.verifier_cfg.judge_base_url or "")).rstrip("/")
            url = f"{base}/v1/chat/completions" if not base.endswith("/v1") else f"{base}/chat/completions"
            responses_url = f"{base}/v1/responses" if not base.endswith("/v1") else f"{base}/responses"

            # Translate sampling params for OpenAI-style API.
            # vLLM uses OpenAI API fields: max_tokens, temperature, top_p, etc.
            max_tokens = sampling_params.get("max_tokens")
            if max_tokens is None:
                # internal code tends to pass max_new_tokens
                max_tokens = sampling_params.get("max_new_tokens", 128)

            model_name = str(model_name_or_path_override or self.verifier_cfg.judge_model_name_or_path)
            # For OpenAI official endpoint + gpt-5* models, prefer Responses API to avoid empty message.content
            # while consuming many completion tokens.
            use_responses_api = ("api.openai.com" in base.lower()) and model_name.lower().startswith("gpt-5")
            # Some OpenAI gpt-5* models have restricted sampling params:
            # - `responses` may reject `temperature` entirely
            # - `chat.completions` may only accept the default temperature (1.0)
            # So we special-case those to avoid 400s.
            is_openai_official = "api.openai.com" in base.lower()
            is_gpt5 = model_name.lower().startswith("gpt-5")

            payload: dict[str, Any] = {
                "model": model_name,
                "messages": messages,
            }
            # Default: pass through temperature/top_p.
            # OpenAI gpt-5*: force/omit as required to avoid unsupported_parameter/unsupported_value.
            if is_openai_official and is_gpt5:
                # Chat Completions: safest is to omit temperature/top_p (use defaults).
                # If caller set temperature explicitly, clamp it to 1.0 for gpt-5 models.
                t = sampling_params.get("temperature", None)
                if t is not None:
                    payload["temperature"] = 1.0
            else:
                payload["temperature"] = float(sampling_params.get("temperature", 0.0))
                payload["top_p"] = float(sampling_params.get("top_p", 1.0))
            # Token budget field name differs across backends/models:
            # - Many OpenAI-compatible servers (e.g. vLLM) accept `max_tokens`.
            # - Some OpenAI models (e.g. gpt-5-mini) require `max_completion_tokens`.
            # We'll pick a best-effort default and also retry once on "unsupported_parameter" errors.
            prefer_max_completion = model_name.lower().startswith("gpt-5")
            token_field = "max_completion_tokens" if prefer_max_completion else "max_tokens"
            payload[token_field] = int(max_tokens)
            # Optional fields passthrough (safe for vLLM)
            for k in ("presence_penalty", "frequency_penalty", "stop"):
                if k in sampling_params:
                    payload[k] = sampling_params[k]

            headers = {}
            api_key = api_key_override if api_key_override is not None else self.verifier_cfg.judge_api_key
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            timeout_s = timeout_s_override if timeout_s_override is not None else float(self.verifier_cfg.judge_timeout_s)
            timeout = aiohttp.ClientTimeout(total=float(timeout_s))
            session = aiohttp.ClientSession(timeout=timeout)
            try:
                try:
                    sem = getattr(self, "_remote_judge_semaphore", None)
                    if sem is None:
                        sem = getattr(self.__class__, "_remote_judge_semaphore", None)

                    async def do_post():
                        async def post_once(pl: dict[str, Any]) -> tuple[int, str]:
                            async with session.post(url, headers=headers, json=pl) as r:
                                return int(r.status), await r.text()

                        async def post_once_responses(pl: dict[str, Any]) -> tuple[int, str]:
                            async with session.post(responses_url, headers=headers, json=pl) as r:
                                return int(r.status), await r.text()

                        # Preferred path for OpenAI gpt-5*: call Responses API and parse output text.
                        if use_responses_api:
                            resp_payload: dict[str, Any] = {
                                "model": model_name,
                                # Best-effort: pass chat-like messages as input. (OpenAI Responses API accepts this shape.)
                                "input": messages,
                                "max_output_tokens": int(max_tokens),
                            }
                            # Do NOT send temperature/top_p for OpenAI gpt-5* responses API: some models reject it.
                            # For non-gpt-5 models, we can pass them through.
                            if not (is_openai_official and is_gpt5):
                                resp_payload["temperature"] = float(sampling_params.get("temperature", 0.0))
                                resp_payload["top_p"] = float(sampling_params.get("top_p", 1.0))
                            for k in ("presence_penalty", "frequency_penalty", "stop"):
                                if k in sampling_params:
                                    resp_payload[k] = sampling_params[k]

                            status_r, raw_r = await post_once_responses(resp_payload)
                            if status_r < 400:
                                try:
                                    data_r = json.loads(raw_r)
                                except Exception as e:
                                    logger.warning(
                                        "Responses API invalid JSON (fallback to chat.completions). request_id=%s model=%s url=%s err=%s body=%r",
                                        request_id,
                                        model_name,
                                        responses_url,
                                        e,
                                        raw_r[:500],
                                    )
                                else:
                                    out_text = ""
                                    if isinstance(data_r, dict) and isinstance(data_r.get("output_text"), str):
                                        out_text = data_r.get("output_text") or ""
                                    if not out_text and isinstance(data_r, dict):
                                        outs = data_r.get("output") or []
                                        if isinstance(outs, list) and outs:
                                            parts: list[str] = []
                                            for o in outs:
                                                if not isinstance(o, dict):
                                                    continue
                                                content_list = o.get("content") or []
                                                if not isinstance(content_list, list):
                                                    continue
                                                for c in content_list:
                                                    if not isinstance(c, dict):
                                                        continue
                                                    t = c.get("text")
                                                    if isinstance(t, str):
                                                        parts.append(t)
                                            out_text = "".join(parts)
                                    if str(out_text).strip():
                                        return str(out_text)
                                    logger.warning(
                                        "Responses API returned empty text (fallback to chat.completions). request_id=%s model=%s url=%s body=%r",
                                        request_id,
                                        model_name,
                                        responses_url,
                                        raw_r[:500],
                                    )
                            else:
                                logger.warning(
                                    "Responses API HTTP %s (fallback to chat.completions). request_id=%s model=%s url=%s body=%r",
                                    status_r,
                                    request_id,
                                    model_name,
                                    responses_url,
                                    raw_r[:500],
                                )

                        status, raw = await post_once(payload)
                        if status >= 400:
                            # Try a one-shot retry by swapping max token field name if the backend complains.
                            alt_field = "max_tokens" if token_field == "max_completion_tokens" else "max_completion_tokens"
                            try:
                                data0 = json.loads(raw)
                            except Exception:
                                data0 = None
                            try_retry = False
                            if isinstance(data0, dict) and isinstance(data0.get("error"), dict):
                                err = data0["error"]
                                if str(err.get("code") or "") == "unsupported_parameter":
                                    # OpenAI style: {"error": {"param": "max_tokens", "message": "... Use 'max_completion_tokens' instead."}}
                                    bad = str(err.get("param") or "")
                                    if bad in {"max_tokens", "max_completion_tokens"}:
                                        try_retry = True
                                    else:
                                        # Also support message-based detection.
                                        msg = str(err.get("message") or "")
                                        if ("max_completion_tokens" in msg) or ("max_tokens" in msg):
                                            try_retry = True
                            if try_retry:
                                payload2 = dict(payload)
                                # Swap fields: remove both then set the alternate.
                                payload2.pop("max_tokens", None)
                                payload2.pop("max_completion_tokens", None)
                                payload2[alt_field] = int(max_tokens)
                                status2, raw2 = await post_once(payload2)
                                if status2 < 400:
                                    status, raw = status2, raw2
                                else:
                                    # Keep the original error context but log both.
                                    logger.warning(
                                        "Remote judge retry also failed. request_id=%s model=%s url=%s status1=%s status2=%s body1=%r body2=%r",
                                        request_id,
                                        model_name,
                                        url,
                                        status,
                                        status2,
                                        raw[:500],
                                        raw2[:500],
                                    )
                            # fall through to normal error handling with (status, raw)

                        if status >= 400:
                            # Don't crash the whole rollout; log and fall back.
                            logger.warning(
                                "Remote judge HTTP %s (fallback to empty). request_id=%s model=%s url=%s body=%r",
                                status,
                                request_id,
                                model_name,
                                url,
                                raw[:500],
                            )
                            return ""

                        try:
                            data = json.loads(raw)
                        except Exception as e:
                            logger.warning(
                                "Remote judge invalid JSON (fallback to empty). request_id=%s model=%s url=%s err=%s body=%r",
                                request_id,
                                model_name,
                                url,
                                e,
                                raw[:500],
                            )
                            return ""

                        # OpenAI-compatible error shape even on 200 (rare but happens with proxies)
                        if isinstance(data, dict) and data.get("error") is not None:
                            logger.warning(
                                "Remote judge returned error field (fallback to empty). request_id=%s model=%s url=%s error=%r",
                                request_id,
                                model_name,
                                url,
                                data.get("error"),
                            )
                            return ""

                        # OpenAI-compatible response
                        choices = data.get("choices") or []
                        if not choices:
                            logger.warning(
                                "Remote judge returned empty choices (fallback to empty). request_id=%s model=%s url=%s keys=%s body=%r",
                                request_id,
                                model_name,
                                url,
                                list(data.keys()) if isinstance(data, dict) else type(data),
                                raw[:500],
                            )
                            return ""

                        choice0 = choices[0] or {}
                        msg = (choice0.get("message") or {})
                        content = msg.get("content")
                        if isinstance(content, str):
                            out_text = content
                        elif isinstance(content, list):
                            # Some backends may return structured content parts; best-effort concat.
                            parts: list[str] = []
                            for p in content:
                                if isinstance(p, str):
                                    parts.append(p)
                                elif isinstance(p, dict):
                                    if "text" in p:
                                        parts.append(str(p.get("text") or ""))
                                    elif "content" in p:
                                        parts.append(str(p.get("content") or ""))
                            out_text = "".join(parts)
                        else:
                            out_text = ""

                        if not str(out_text).strip():
                            logger.warning(
                                "Remote judge returned empty message content (fallback to empty). request_id=%s model=%s url=%s finish_reason=%r message_keys=%s body=%r",
                                request_id,
                                model_name,
                                url,
                                choice0.get("finish_reason"),
                                sorted(list(msg.keys())) if isinstance(msg, dict) else type(msg),
                                raw[:500],
                            )
                            return ""
                        return str(out_text)

                    if sem is None:
                        return await do_post()
                    async with sem:
                        return await do_post()
                except (TimeoutError, OSError, aiohttp.ClientError) as e:
                    # Be robust: remote judge failures should not crash the whole rollout batch.
                    logger.warning(
                        "Remote judge request failed (fallback to empty): request_id=%s model=%s url=%s err=%s",
                        request_id,
                        model_name,
                        url,
                        e,
                        exc_info=True,
                    )
                    return ""
            finally:
                await session.close()

        # policy backend
        # vLLM rollout server expects OpenAI-style `max_tokens`.
        sampling_params = dict(sampling_params)
        if "max_new_tokens" in sampling_params and "max_tokens" not in sampling_params:
            sampling_params["max_tokens"] = sampling_params.pop("max_new_tokens")
        else:
            sampling_params.pop("max_new_tokens", None)
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
            ),
        )
        out = await self.server_manager.generate(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=None,
        )
        return await self.loop.run_in_executor(None, lambda: self.tokenizer.decode(out.token_ids, skip_special_tokens=True))



