"""
Step-causal verifier agent loop:
- Multi-turn "subproblem" conversations (extra_info.subproblems)
- For each assistant response, call a judge model to output a continuous PNS score in [0,1]
- If PNS >= threshold: give a small positive step reward
- If PNS < threshold: append a user feedback message ("causal incoherent") and retry generating this step
  up to max_retries. If still failing, apply a penalty.

This loop outputs token-level credit assignment metadata:
  extra_fields["turn_scores"]: list[float]
  extra_fields["step_token_spans"]: list[tuple[int,int]]  # spans in response token space
  extra_fields["final_reward"]: float

Then `AgentLoopWorkerBase._postprocess()` will convert them into per-token rm_scores.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.step_verify_agent_loop import StepVerifyAgentLoop, _has_answer_marker
from verl.utils.profiler import simple_timer


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_CAUSAL_REASONINGMODEL_NAME = "/home/projects/hku-medai/larrypl/code/mq/causal-reasoning/hf_models/gsm8k_deberta_regression"


_FLOAT_RE = re.compile(r"(?<![\d.])([01](?:\.\d+)?|\.\d+)(?![\d.])")


def _parse_float_0_1(text: str) -> Optional[float]:
    if not isinstance(text, str):
        return None
    t = text.strip()
    if not t:
        return None
    m = _FLOAT_RE.search(t)
    if not m:
        return None
    try:
        x = float(m.group(1))
    except Exception:
        return None
    if x < 0.0:
        x = 0.0
    if x > 1.0:
        x = 1.0
    return float(x)


def _has_answer_marker_bak2(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text
    if re.search(r"<\s*answer\s*>", t, flags=re.IGNORECASE):
        return True
    if re.search(r"####\s*\S+", t):  # GSM8K
        return True
    return False


def _has_answer_marker_bak(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text
    tl = t.lower()
    # XML-ish tags
    if "<answer" in tl or "</answer" in tl:
        return True
    # GSM8K marker
    if re.search(r"####\s*\S+", t):
        return True
    # LaTeX boxed
    if "\\boxed" in t:
        return True
    # Common phrases / formats
    if re.search(r"final\s+answer", t, flags=re.IGNORECASE):
        return True
    if re.search(r"^\s*answer\s*:\s*\S+", t, flags=re.IGNORECASE | re.MULTILINE):
        return True
    if re.search(r"the\s+answer\s+is\s+\S+", t, flags=re.IGNORECASE):
        return True
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
    """
    Match the DeBERTa judge input template used in `run_deberta_judge_batch.py`:
      Question:\n{question}\n\nPrefix steps:\n{prefix_text}\n\nCandidate step:\n{candidate_step}
    """
    prefix_text = "\n".join([str(s).strip() for s in (prefix_steps or []) if str(s).strip()])
    return (
        f"Question:\n{str(question or '').strip()}\n\n"
        f"Prefix steps:\n{prefix_text}\n\n"
        f"Candidate step:\n{str(candidate_step or '').strip()}"
    )


@dataclass
class _CausalCfg:
    # Judge prompt to predict pns in [0,1]
    pns_prompt_template: str = (
        "You are a strict causal coherence / necessity judge.\n"
        "Given the problem, the current subproblem, and the assistant answer, output a PNS score in [0,1].\n"
        "Higher means the step is more causally coherent/necessary; lower means it is not necessary or is incoherent.\n"
        "Output ONLY one number in [0,1].\n\n"
        "Problem:\n{problem}\n\n"
        "Subproblem:\n{subproblem}\n\n"
        "Assistant answer:\n{answer}\n\n"
        "pns="
    )
    pns_max_new_tokens: int = 8
    pns_temperature: float = 0.0
    pns_top_p: float = 1.0

    # How to interpret judge output:
    # - "float": pns is a float in [0,1], pass iff pns >= pns_threshold
    # - "discrete": pns is treated as {0,1}, pass iff pns == 1 (threshold is ignored)
    pns_mode: str = "float"  # float | discrete

    # NEW: In float mode, skip retry logic and directly penalize low-pns steps
    # When True: step_reward = pns * pass_reward if pns >= threshold, else (pns - threshold) * penalty_scale
    # This allows continuous reward shaping based on PNS score without retry overhead.
    pns_float_no_retry: bool = False
    pns_penalty_scale: float = 0.1  # scale factor for penalty when pns < threshold

    # Reward / retry behavior
    pns_threshold: float = 0.6
    pass_reward: float = 0.05
    fail_reward: float = 0.0
    exceed_retry_penalty: float = 0.05
    max_retries: int = 2
    max_steps: int = 32  # maximum accepted steps before forcing stop (safety)

    # Anti-early-final gate (decision-level):
    # Require at least this many assistant turns before allowing any final-answer marker.
    min_assistant_turns_before_answer: int = 0
    # If a final-answer marker appears earlier than allowed, apply this penalty to final reward.
    early_answer_penalty: float = 0.0

    # Optional step reward scaling (process reward):
    # If enabled, scale all per-turn rewards/penalties by (step_total_weight / N),
    # where N is the number of assistant attempts that produced a turn_score.
    # This makes the total process reward magnitude be on the order of step_total_weight.
    step_scale_by_num_turns: bool = True
    step_total_weight: float = 0.3
    step_clip_min: float = -1.0
    step_clip_max: float = 1.0

    # Feedback message when PNS < threshold
    feedback_template: str = (
        "This step is not acceptable: causal logic is not rigorous "
        "(pns={pns:.3f} < threshold={threshold:.3f}).\n"
        "Please rewrite ONLY this step and try again. Requirements:\n"
        "- Make the reasoning causally coherent with the prior context\n"
        "- Keep it brief (1-3 sentences)\n"
        "- Do not add multiple steps at once\n"
    )

    ok_continue_msg: str = "Good. Continue with the next step (one step only)."
    ok_finish_msg: str = "Good. Now provide the complete final solution and end with <ANSWER>...</ANSWER>."
    proceed_after_penalty_msg: str = (
        "Too many retries. Proceed to the next step anyway (one step only)."
    )


@register("step_causal_verifier_agent")
class StepCausalVerifierAgentLoop(StepVerifyAgentLoop):
    """
    Inherits judge backend + final reward logic from StepVerifyAgentLoop, but:
    - step judge returns continuous pns in [0,1] (not binary)
    - adds retry-with-feedback when pns < threshold
    - step reward uses pass_reward/fail_reward/penalty
    """

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        # IMPORTANT: init_class may be called multiple times by hydra.instantiate, but we only want to
        # do heavy initialization once per worker process.
        if getattr(cls, "_class_initialized", False):
            return

        try:
            logger.warning("StepCausalVerifierAgentLoop.init_class kwargs keys=%s", sorted(list(kwargs.keys())))
        except Exception:
            pass

        # For this agent, we REQUIRE step_verify config to be passed via YAML.
        # Otherwise we may silently fall back to generation-based judge and hit context-length issues.
        if "step_verify" not in kwargs:
            raise RuntimeError(
                "StepCausalVerifierAgentLoop requires `step_verify` config, but it was not passed into init_class(). "
                "This usually means your agent_loop_config_path YAML was not loaded/visible on this worker, or the YAML "
                "nesting is wrong. Fix: ensure actor_rollout_ref.rollout.agent.agent_loop_config_path points to "
                "`step_causal_verifier.yaml` on a shared filesystem accessible to all Ray workers, and that the YAML "
                "entry contains `step_verify:` under the agent config."
            )
        # IMPORTANT:
        # StepVerifyAgentLoop.init_class() will try to load a LOCAL judge backend via AutoModelForCausalLM when
        # judge_backend == "local". This breaks for DeBERTa-style sequence classification judges.
        # So we detect that case early, and temporarily force judge_backend="policy" when calling super(),
        # while still initializing our own seq-classifier PNS judge below.
        step_cfg = kwargs.get("step_verify") or {}
        # Convert OmegaConf -> dict if needed (super already does but keep robust)
        if step_cfg and not isinstance(step_cfg, dict):
            try:
                from omegaconf import OmegaConf

                step_cfg = OmegaConf.to_container(step_cfg, resolve=True)  # type: ignore[assignment]
            except Exception:
                step_cfg = {}

        backend_req = str(step_cfg.get("judge_backend", "") or "").lower()
        model_req = str(step_cfg.get("judge_model_name_or_path", "") or "")
        seqcls_req = bool(step_cfg.get("pns_seqcls_enable", False))
        if not seqcls_req:
            # Back-compat: auto-detect the known DeBERTa judge by name OR by config.model_type.
            if model_req == _CAUSAL_REASONINGMODEL_NAME or ("CausalReasoningModel" in model_req):
                seqcls_req = True
            elif backend_req == "local" and model_req:
                try:
                    from transformers import AutoConfig

                    cfg = AutoConfig.from_pretrained(model_req)
                    # DeBERTa / RoBERTa / BERT-style seq-classification models should be treated as seqcls.
                    mt = str(getattr(cfg, "model_type", "") or "").lower()
                    if mt in {"deberta", "deberta-v2", "deberta_v2", "bert", "roberta", "electra"}:
                        seqcls_req = True
                except Exception:
                    pass

        # Call super with a possibly-overridden judge backend to avoid AutoModelForCausalLM loading errors.
        super_kwargs = dict(kwargs)
        if backend_req == "local" and seqcls_req:
            step_cfg2 = dict(step_cfg)
            # Force parent judge backend to "policy" so it doesn't try to load a local CausalLM judge.
            step_cfg2["judge_backend"] = "policy"
            super_kwargs["step_verify"] = step_cfg2
        super().init_class(config=config, tokenizer=tokenizer, processor=processor, **super_kwargs)
        # Mark initialized AFTER super() so tokenizer/system_prompt/verifier_cfg are ready.
        cls._class_initialized = True

        # Record what the user requested for debugging / fail-fast.
        cls._pns_backend_req = backend_req
        cls._pns_model_req = model_req
        cls._pns_seqcls_req = bool(seqcls_req)

        cls.causal_cfg = _CausalCfg(
            pns_prompt_template=str(step_cfg.get("pns_prompt_template", _CausalCfg.pns_prompt_template)),
            pns_max_new_tokens=int(step_cfg.get("pns_max_new_tokens", _CausalCfg.pns_max_new_tokens)),
            pns_temperature=float(step_cfg.get("pns_temperature", _CausalCfg.pns_temperature)),
            pns_top_p=float(step_cfg.get("pns_top_p", _CausalCfg.pns_top_p)),
            pns_mode=str(step_cfg.get("pns_mode", _CausalCfg.pns_mode)),
            pns_float_no_retry=bool(step_cfg.get("pns_float_no_retry", _CausalCfg.pns_float_no_retry)),
            pns_penalty_scale=float(step_cfg.get("pns_penalty_scale", _CausalCfg.pns_penalty_scale)),
            pns_threshold=float(step_cfg.get("pns_threshold", _CausalCfg.pns_threshold)),
            pass_reward=float(step_cfg.get("pns_pass_reward", _CausalCfg.pass_reward)),
            fail_reward=float(step_cfg.get("pns_fail_reward", _CausalCfg.fail_reward)),
            exceed_retry_penalty=float(step_cfg.get("pns_exceed_retry_penalty", _CausalCfg.exceed_retry_penalty)),
            max_retries=int(step_cfg.get("pns_max_retries", _CausalCfg.max_retries)),
            max_steps=int(step_cfg.get("pns_max_steps", _CausalCfg.max_steps)),
            min_assistant_turns_before_answer=int(step_cfg.get("min_assistant_turns_before_answer", _CausalCfg.min_assistant_turns_before_answer)),
            early_answer_penalty=float(step_cfg.get("early_answer_penalty", _CausalCfg.early_answer_penalty)),
            step_scale_by_num_turns=bool(step_cfg.get("step_scale_by_num_turns", _CausalCfg.step_scale_by_num_turns)),
            step_total_weight=float(step_cfg.get("step_total_weight", _CausalCfg.step_total_weight)),
            step_clip_min=float(step_cfg.get("step_clip_min", _CausalCfg.step_clip_min)),
            step_clip_max=float(step_cfg.get("step_clip_max", _CausalCfg.step_clip_max)),
            feedback_template=str(step_cfg.get("pns_feedback_template", _CausalCfg.feedback_template)),
            ok_continue_msg=str(step_cfg.get("pns_ok_continue_msg", _CausalCfg.ok_continue_msg)),
            ok_finish_msg=str(step_cfg.get("pns_ok_finish_msg", _CausalCfg.ok_finish_msg)),
            proceed_after_penalty_msg=str(
                step_cfg.get("pns_proceed_after_penalty_msg", _CausalCfg.proceed_after_penalty_msg)
            ),
        )
        # normalize mode
        try:
            m = str(getattr(cls.causal_cfg, "pns_mode", "float") or "float").lower()
            cls.causal_cfg.pns_mode = "discrete" if m in {"discrete", "binary", "01", "0/1"} else "float"
        except Exception:
            cls.causal_cfg.pns_mode = "float"

        # Optional: local DeBERTa-style classifier judge for PNS (MakimaSasha/CausalReasoningModel).
        # NOTE: StepVerifyAgentLoop's local judge backend is CausalLM-generation based; it will NOT work for
        # sequence classification models. So we special-case this model name here.
        cls._pns_is_seqcls = False
        cls._pns_tokenizer = None
        cls._pns_model = None
        try:
            backend = backend_req
            model_name = model_req
            if backend == "local" and model_name and seqcls_req:
                cls._pns_is_seqcls = True
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                cls._pns_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                cls._pns_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                cls._pns_model.eval()
                # Place on requested device (CPU recommended for this path)
                import torch

                dev = str(getattr(cls.verifier_cfg, "judge_device", "cpu") or "cpu").lower()
                if dev != "cpu":
                    cls._pns_model = cls._pns_model.to(dev)
        except Exception as e:
            # Fail fast with actionable error, rather than silently falling back to generation judge.
            raise RuntimeError(f"Failed to init local seq-classifier PNS judge ({model_req}): {e}") from e

        try:
            logger.warning(
                "StepCausalVerifierAgentLoop judge route: backend_req=%r model_req=%r seqcls_req=%r pns_is_seqcls=%r pns_mode=%r",
                cls._pns_backend_req,
                cls._pns_model_req,
                cls._pns_seqcls_req,
                cls._pns_is_seqcls,
                getattr(cls.causal_cfg, "pns_mode", None),
            )
        except Exception:
            pass

    async def _judge_pns(
        self,
        *,
        request_id: str,
        step_idx: int,
        attempt_idx: int,
        problem_text: str,
        subproblem_text: str,
        assistant_text: str,
        prefix_steps: list[str],
        sampling_params: dict[str, Any],
    ) -> float:
        cfg: _CausalCfg = self.__class__.causal_cfg  # type: ignore[attr-defined]
        cls = self.__class__
        if bool(getattr(cls, "_pns_is_seqcls", False)):
            # Sequence classification path (MakimaSasha/CausalReasoningModel)
            import torch

            tok = getattr(cls, "_pns_tokenizer", None)
            model = getattr(cls, "_pns_model", None)
            if tok is None or model is None:
                raise RuntimeError("Seq-classifier PNS judge is enabled but model/tokenizer is not initialized.")
            text = _build_causalreasoningmodel_text(
                question=str(problem_text or ""),
                prefix_steps=prefix_steps,
                candidate_step=str(assistant_text or ""),
            )
            dev = str(getattr(cls.verifier_cfg, "judge_device", "cpu") or "cpu").lower()
            device = torch.device("cuda" if (dev != "cpu" and torch.cuda.is_available()) else "cpu")
            inputs = tok(
                [text],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
                logits = out.logits
                # 2-logit head preferred; else sigmoid on single logit
                if logits.dim() == 2 and logits.size(-1) == 2:
                    prob1 = torch.softmax(logits, dim=-1)[:, 1]
                    pred1 = torch.argmax(logits, dim=-1).float()
                else:
                    prob1 = torch.sigmoid(logits.view(-1))
                    pred1 = (prob1 >= 0.5).float()
                if str(cfg.pns_mode).lower() == "discrete":
                    return float(pred1.detach().float().cpu().item())
                return float(prob1.detach().float().cpu().item())

        # If user requested local + seqcls but we did not initialize it, fail with a clear message.
        if str(getattr(cls, "_pns_backend_req", "") or "").lower() == "local" and bool(
            getattr(cls, "_pns_seqcls_req", False)
        ):
            raise RuntimeError(
                "PNS judge is configured as local seq-classifier, but seq-classifier is not initialized. "
                "Check step_causal_verifier.yaml: set judge_backend=local and judge_model_name_or_path to your DeBERTa "
                "(e.g., MakimaSasha/CausalReasoningModel) and ensure the file is the one actually loaded by "
                "actor_rollout_ref.rollout.agent.agent_loop_config_path."
            )

        # Default: generation-based judge (expects a float or {0,1} in output text)
        prompt = cfg.pns_prompt_template.format(problem=problem_text, subproblem=subproblem_text, answer=assistant_text)
        messages = [{"role": "system", "content": "You are a verifier."}, {"role": "user", "content": prompt}]

        verifier_sampling = dict(sampling_params)
        verifier_sampling.pop("max_new_tokens", None)
        verifier_sampling["max_tokens"] = int(cfg.pns_max_new_tokens)
        verifier_sampling["temperature"] = float(cfg.pns_temperature)
        verifier_sampling["top_p"] = float(cfg.pns_top_p)

        text = await self._judge_completion_text(
            request_id=f"pns_{request_id}_{step_idx}_{attempt_idx}",
            messages=messages,
            sampling_params=verifier_sampling,
        )
        p = _parse_float_0_1(text or "")
        if p is None:
            logger.warning("PNS judge returned non-numeric output, fallback to 0.0: %r", (text or "")[:200])
            return 0.0
        if str(cfg.pns_mode).lower() == "discrete":
            return 1.0 if float(p) >= 0.5 else 0.0
        return float(p)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # Start from initial raw prompt (system + user problem prompt)
        messages: list[dict[str, Any]] = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info") or {}
        reward_model = kwargs.get("reward_model") or {}

        # For final reward
        ground_truth = reward_model.get("ground_truth") or extra_info.get("ground_truth")
        problem_text = (
            extra_info.get("problem_text")
            or extra_info.get("question")
            or extra_info.get("question_hint")
            or ""
        )

        request_id = uuid4().hex
        metrics: dict[str, Any] = {}

        prompt_ids = await self._encode_full_messages(messages)

        response_mask: list[int] = []
        response_logprobs: list[float] = []
        step_token_spans: list[tuple[int, int]] = []
        turn_scores: list[float] = []
        pns_scores: list[float] = []
        retry_counts: list[int] = []

        assistant_turns = 0
        user_turns = 0

        # Track when the first final-answer marker appears (1-based assistant turn index)
        first_answer_marker_turn: Optional[int] = None

        # Track the last assistant text for final reward
        last_assistant_text: str = ""
        accepted_steps: list[str] = []

        # Interactive loop: each assistant turn should be one step; we provide feedback as user turns.
        step_idx = 0  # accepted-step counter
        cfg: _CausalCfg = self.__class__.causal_cfg  # type: ignore[attr-defined]
        retry_for_current_step = 0
        while True:
            # termination: token budget / turn limits

            # if self.prompt_length and (len(prompt_ids) + len(user_ids)) >= int(self.prompt_length):
            #    break

            if len(response_mask) >= self.response_length:
                break
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                break
            if self.max_user_turns and user_turns >= self.max_user_turns:
                break
            if int(cfg.max_steps) > 0 and step_idx >= int(cfg.max_steps):
                break

            # 1) policy generate one step
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

            assistant_text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            )
            last_assistant_text = str(assistant_text or "")
            messages.append({"role": "assistant", "content": assistant_text})


            # --- Anti-early-final gate + FINAL BYPASS PNS ---
            has_marker = bool(_has_answer_marker(last_assistant_text))

            # Record first marker turn (for optional penalty/debug)
            if has_marker and first_answer_marker_turn is None:
                first_answer_marker_turn = int(assistant_turns)  # 1-based assistant turn index

            min_turns = int(getattr(cfg, "min_assistant_turns_before_answer", 0) or 0)

            # IMPORTANT: interpret min_turns as "min accepted steps before final", not raw assistant_turns
            # step_idx is the accepted-step counter in this loop.
            allow_finish = (min_turns <= 0) or (step_idx >= min_turns)

            # If final marker appears AND we are allowed to finish:
            #   -> DO NOT run PNS on this final output (final is judged by final reward), just break.
            if has_marker and allow_finish:
                finished = True
                force_early_marker_fail = False
                break

            # If marker appears too early:
            #   -> force rewrite (skip judge) and continue retry flow
            finished = False
            force_early_marker_fail = bool(has_marker and (not allow_finish))
            # --- END gate ---


            # 2) judge pns for this step
            with simple_timer("tool_calls", metrics):
                if force_early_marker_fail:
                    # Skip judge: force rewrite when answer marker appears too early
                    pns = 0.0
                else:
                    pns = await self._judge_pns(
                    request_id=request_id,
                    step_idx=step_idx,
                    attempt_idx=retry_for_current_step,
                    problem_text=str(problem_text or ""),
                    subproblem_text=str(problem_text or ""),
                    assistant_text=str(assistant_text or ""),
                    prefix_steps=list(accepted_steps),
                    sampling_params=sampling_params,
                    )
            pns_scores.append(float(pns))

            # Check if using float mode with no-retry (continuous reward shaping)
            is_float_no_retry = (
                (not force_early_marker_fail)
                and str(cfg.pns_mode).lower() != "discrete" 
                and bool(getattr(cfg, "pns_float_no_retry", False))
            )

            if is_float_no_retry:
                # Float no-retry mode: directly compute reward based on PNS, no retry logic
                # Reward = pns * pass_reward if pns >= threshold, else (pns - threshold) * penalty_scale
                threshold = float(cfg.pns_threshold)
                if float(pns) >= threshold:
                    # Positive reward scaled by how much above threshold
                    step_reward = float(pns) * float(cfg.pass_reward)
                else:
                    # Penalty scaled by how much below threshold (negative value)
                    step_reward = (float(pns) - threshold) * float(cfg.pns_penalty_scale)
                
                turn_scores.append(step_reward)
                retry_counts.append(0)  # No retries in this mode
                accepted_steps.append(str(assistant_text or ""))
                step_idx += 1
                
                if finished:
                    break
                
                # In no-retry mode, just provide a simple continue message
                add_messages = [{"role": "user", "content": str(cfg.ok_continue_msg)}]
            else:
                # Original discrete/retry mode
                if force_early_marker_fail:
                    passed = False
                else:
                    if str(cfg.pns_mode).lower() == "discrete":
                        passed = bool(float(pns) >= 0.5)  # treat as 0/1
                    else:
                        passed = bool(float(pns) >= float(cfg.pns_threshold))
                
                if passed:
                    turn_scores.append(float(cfg.pass_reward))
                    retry_counts.append(int(retry_for_current_step))
                    retry_for_current_step = 0
                    accepted_steps.append(str(assistant_text or ""))
                    step_idx += 1
                    if finished:
                        break
                    # Provide "continue" feedback
                    add_messages = [{"role": "user", "content": str(cfg.ok_continue_msg)}]
                else:
                    # failed: ask to rethink this step; retry up to max_retries, then penalize and proceed.
                    retry_for_current_step += 1
                    if retry_for_current_step > int(cfg.max_retries):
                        turn_scores.append(-float(cfg.exceed_retry_penalty))
                        retry_counts.append(int(retry_for_current_step))
                        retry_for_current_step = 0
                        accepted_steps.append(str(assistant_text or ""))
                        step_idx += 1  # proceed to next step anyway
                        add_messages = [{"role": "user", "content": str(cfg.proceed_after_penalty_msg)}]
                    else:
                        turn_scores.append(float(cfg.fail_reward))
                        retry_counts.append(int(retry_for_current_step))
                        if force_early_marker_fail:
                            fb = f"Too early. No <ANSWER>/#### until step >= {min_turns}. Rewrite 1 step (1-3 sentences)."
                            #fb = (
                            #   "This step is not acceptable. You revealed a final answer too early.\n"
                            #    f"Do NOT output any final answer marker (<ANSWER>/####/\\\\boxed/Answer:) until step >= {min_turns}.\n"
                            #    "Rewrite ONLY this single step (1-3 sentences) and continue."
                            #)
                        else:
                            fb = cfg.feedback_template.format(pns=float(pns), threshold=float(cfg.pns_threshold))
                        add_messages = [{"role": "user", "content": fb}]

            # Append feedback user message and continue
            if self.max_user_turns and user_turns >= self.max_user_turns:
                break
            messages.extend(add_messages)
            user_turns += 1

            # user_ids = await self._encode_incremental_messages(add_messages)
            # if len(response_mask) + len(user_ids) >= self.response_length:
            #     break
            # 用 prompt_length 做上限（通常是 4096），避免 feedback 一加就 break
            # if self.prompt_length and (len(prompt_ids) + len(user_ids)) >= int(self.prompt_length):
            #     break
            # prompt_ids += user_ids

            user_ids = await self._encode_incremental_messages(add_messages)
            if self.prompt_length and (len(prompt_ids) + len(user_ids)) >= int(self.prompt_length):
                break
            prompt_ids += user_ids

            response_mask += [0] * len(user_ids)
            if response_logprobs:
                response_logprobs += [0.0] * len(user_ids)

        # Final reward (answer correctness) - reuse parent implementation
        final_reward = 0.0
        # If a final-answer marker appeared too early, penalize and suppress final correctness reward.
        min_turns = int(getattr(cfg, "min_assistant_turns_before_answer", 0) or 0)
        early_pen = float(getattr(cfg, "early_answer_penalty", 0.0) or 0.0)
        early_answer = bool(min_turns > 0 and first_answer_marker_turn is not None and int(first_answer_marker_turn) < min_turns)
        if early_answer and early_pen > 0:
            final_reward = -early_pen
        elif self.verifier_cfg.final_enable and ground_truth is not None:
            try:
                final_reward = await self._verify_final(
                    request_id=request_id,
                    problem_text=str(problem_text),
                    last_assistant_text=str(last_assistant_text),
                    ground_truth=str(ground_truth),
                    sampling_params=sampling_params,
                )
            except Exception as e:
                logger.warning("final verification failed: %s", e)
                final_reward = 0.0
            final_reward *= float(self.verifier_cfg.final_weight)

        # Optional: scale step rewards/penalties by number of turns, then map into a bounded weight range.
        # Per your requirement: sum rewards+penalties, average by N turns, then scale to step_total_weight.
        # This is equivalent to multiplying each per-turn score by (step_total_weight / N).
        if bool(getattr(cfg, "step_scale_by_num_turns", False)) and turn_scores:
            try:
                n = max(1, len(turn_scores))
                total_w = float(getattr(cfg, "step_total_weight", 0.3))
                clip_min = float(getattr(cfg, "step_clip_min", -1.0))
                clip_max = float(getattr(cfg, "step_clip_max", 1.0))
                factor = float(total_w) / float(n)
                # Clip each raw score to keep extreme outliers from dominating, while preserving penalty sign.
                turn_scores = [
                    float(max(clip_min, min(clip_max, float(s)))) * factor  # type: ignore[arg-type]
                    for s in turn_scores
                ]
            except Exception as e:
                logger.warning("step reward scaling failed; keep raw turn_scores: %s", e)

        # Build AgentLoopOutput (same shape convention as StepVerifyAgentLoop)
        response_ids = prompt_ids[-len(response_mask) :] if response_mask else []
        prompt_only_ids = prompt_ids[: len(prompt_ids) - len(response_mask)] if response_mask else prompt_ids

        output = AgentLoopOutput(
            prompt_ids=prompt_only_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            multi_modal_data={},
            reward_score=0.0,
            num_turns=assistant_turns + user_turns + 1,
            metrics=metrics,
            extra_fields={
                "turn_scores": turn_scores,
                "step_token_spans": step_token_spans,
                "final_reward": float(final_reward),
                # debug
                "pns_scores": pns_scores,
                "pns_threshold": float(getattr(self.__class__.causal_cfg, "pns_threshold", 0.0)),  # type: ignore[attr-defined]
                "retry_counts": retry_counts,
                "assistant_turns": int(assistant_turns),
                "user_turns": int(user_turns),
                "request_id": request_id,
                "ts": time.time(),
                "assistant_turn_texts": accepted_steps,  # For logging
            },
        )

        # Periodic sample logging (inherited from StepVerifyAgentLoop)
        self._maybe_log_sample(
            messages=messages,
            extra_info=extra_info,
            reward_model=reward_model,
            output=output,
            subproblems=None,
        )

        return output




