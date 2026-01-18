"""
Backward-agent loop for stage-2 training: learn to decompose problems and orchestrate a frozen forward agent.

This agent loop implements a multi-turn environment with TWO models:
- Backward agent (TRAINED): the policy behind this VERL rollout server_manager (token-in token-out).
- Forward agent (FROZEN): a remote OpenAI-compatible chat-completions endpoint (e.g., vLLM OpenAI server).

High-level flow per episode:
1) Start with ONLY the original problem (no subproblem list given in the dataset prompt).
2) Backward proposes the next subproblem to ask (or chooses to request the final answer).
3) Forward answers the subproblem using full conversation history AND a shared "known clues" list.
4) Backward evaluates the forward step solution, updates clues (<=10 items), manages a FIFO queue of pending subproblems,
   and proposes the next subproblem (queue has priority).
5) When requesting final answer, forward must output final answer in <Answer>...</Answer>.
6) Backward performs a "reverse verification" on the final answer. If wrong, it can restate/simplify or decompose further.

Reward construction (token-level credit assignment metadata):
------------------------------------------------------------
This loop outputs:
  - extra_fields["turn_scores"]: list[float]              # per-backward-turn step rewards (default 0.0)
  - extra_fields["step_token_spans"]: list[tuple[int,int]]# spans in *backward response token space* (start, end)
  - extra_fields["final_reward"]: float                   # objective final reward (0/1 by default)

Then `AgentLoopWorkerBase._postprocess()` will distribute each step reward across tokens in its span,
and also distribute `final_reward` uniformly over all assistant tokens.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _extract_last_user_text(messages: list[dict[str, Any]]) -> str:
    for item in reversed(messages):
        if item.get("role") == "user":
            return str(item.get("content", ""))
    return ""


def _extract_last_assistant_text(messages: list[dict[str, Any]]) -> str:
    for item in reversed(messages):
        if item.get("role") == "assistant":
            return str(item.get("content", ""))
    return ""


def _extract_tagged_answer(text: str) -> Optional[str]:
    """Extract <Answer>...</Answer> (case-insensitive)."""
    if not isinstance(text, str):
        return None
    m = re.findall(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    ans = (m[-1] or "").strip()
    return ans if ans else None


def _extract_final_answer(text: str) -> Optional[str]:
    """
    Best-effort final answer extraction (copied/adapted from step_verify_agent_loop.py).
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
    m = re.findall(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", t, flags=re.IGNORECASE | re.DOTALL)
    if m:
        ans = m[-1].strip()
        t2 = ans if ans else ""
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


def _parse_strict_json_object(text: str) -> Optional[dict[str, Any]]:
    """
    Strict JSON parsing for format reward:
    - must be exactly ONE JSON object (allow surrounding whitespace only)
    - no extra text before/after
    """
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


def _validate_backward_action_schema(
    action: dict[str, Any],
    *,
    pending_queue_nonempty: bool,
) -> bool:
    """
    Validate the backward JSON action schema/constraints.
    This is intentionally separate from strict JSON parsing so it can be used as a separate reward component.

    Required fixed keys:
      - is_correct: bool
      - hint_updates: list[str]
      - next_subproblem: str
      - request_final: bool
      - restate_subproblem: str
      - queue_push: list[str]

    Constraints:
    - If is_correct == True:
        - restate_subproblem == "" AND queue_push == []
        - If pending_queue_nonempty: next_subproblem=="" and request_final==False
        - Else (queue empty): exactly one of:
            A) request_final==True and next_subproblem==""
            B) request_final==False and next_subproblem!=""
    - If is_correct == False:
        - hint_updates==[] AND next_subproblem=="" AND request_final==False
        - Exactly one of:
            A) restate_subproblem!="" and queue_push==[]
            B) restate_subproblem=="" and queue_push is non-empty
    """
    if not isinstance(action, dict):
        return False

    required = [
        "is_correct",
        "hint_updates",
        "next_subproblem",
        "request_final",
        "restate_subproblem",
        "queue_push",
    ]
    if any(k not in action for k in required):
        return False

    is_correct = action.get("is_correct")
    hint_updates = action.get("hint_updates")
    next_subproblem = action.get("next_subproblem")
    request_final = action.get("request_final")
    restate_subproblem = action.get("restate_subproblem")
    queue_push = action.get("queue_push")

    if not isinstance(is_correct, bool):
        return False
    if not isinstance(request_final, bool):
        return False
    if not isinstance(next_subproblem, str):
        return False
    if not isinstance(restate_subproblem, str):
        return False
    if not isinstance(hint_updates, list) or any(not isinstance(x, str) for x in hint_updates):
        return False
    if not isinstance(queue_push, list) or any(not isinstance(x, str) for x in queue_push):
        return False

    next_subproblem_s = next_subproblem.strip()
    restate_s = restate_subproblem.strip()
    queue_push_clean = [str(x).strip() for x in queue_push if str(x).strip()]
    hint_updates_clean = [str(x).strip() for x in hint_updates if str(x).strip()]

    if is_correct:
        # restate/queue must be empty
        if restate_s != "":
            return False
        if queue_push_clean:
            return False

        if pending_queue_nonempty:
            # queue has priority => do not propose next_subproblem or request_final
            if next_subproblem_s != "":
                return False
            if request_final:
                return False
            return True

        # queue empty => must choose exactly one of (request_final) vs (next_subproblem)
        if request_final and next_subproblem_s == "":
            return True
        if (not request_final) and next_subproblem_s != "":
            return True
        return False

    # is_correct == False
    if hint_updates_clean:
        return False
    if next_subproblem_s != "":
        return False
    if request_final:
        return False

    choose_restate = (restate_s != "") and (not queue_push_clean)
    choose_split = (restate_s == "") and bool(queue_push_clean)
    return (choose_restate or choose_split) and not (choose_restate and choose_split)


def _safe_json_first_object(text: str) -> Optional[dict[str, Any]]:
    """
    Parse the first JSON object in a model output. Fail-soft: returns None if cannot parse.
    We intentionally accept noisy outputs that contain JSON somewhere in the text.
    """
    if not isinstance(text, str):
        return None
    t = text.strip()
    if not t:
        return None
    # Find first {...} block (best-effort, non-greedy).
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return None
    blob = m.group(0)
    try:
        obj = json.loads(blob)
    except Exception:
        # Attempt a minimal cleanup (common trailing commas / single quotes are NOT fixed here on purpose).
        return None
    return obj if isinstance(obj, dict) else None


def _clamp_clues(clues: list[str], *, max_items: int = 10, max_chars_total: int = 2200) -> list[str]:
    """
    Enforce:
    - max_items (FIFO): keep most recent max_items
    - max_chars_total: keep from the end until within budget (rough proxy for <=512 tokens)
    """
    # Normalize and drop empties
    norm: list[str] = []
    for c in clues:
        s = str(c or "").strip()
        if s:
            norm.append(s)

    # Keep order but dedupe (stable)
    dedup: list[str] = []
    seen = set()
    for s in norm:
        if s in seen:
            continue
        seen.add(s)
        dedup.append(s)

    # Keep last max_items
    if max_items and len(dedup) > max_items:
        dedup = dedup[-max_items:]

    # Char budget from the end
    if max_chars_total and max_chars_total > 0:
        out: list[str] = []
        total = 0
        for s in reversed(dedup):
            add = len(s) + 2
            if total + add > max_chars_total and out:
                break
            if add > max_chars_total and not out:
                # Single item too long: hard truncate
                out = [s[: max_chars_total - 1]]
                total = len(out[0])
                break
            out.append(s)
            total += add
        dedup = list(reversed(out))
    return dedup


@dataclass
class _BackwardAgentConfig:
    # Backward agent prompting
    backward_system_prompt: str = (
        "You are a backward planning agent for math problems.\n"
        "Your job: decompose the original problem into subproblems, orchestrate a frozen forward solver, and verify.\n"
        "Be concise. Output STRICT JSON only."
    )

    # Forward agent remote endpoint
    forward_base_url: Optional[str] = None  # e.g. "http://127.0.0.1:8000/v1" or "http://127.0.0.1:8000"
    forward_api_key: Optional[str] = None
    forward_model: Optional[str] = None  # model name served by vLLM OpenAI server
    forward_timeout_s: float = 120.0
    forward_max_tokens: int = 1024
    forward_temperature: float = 0.0
    forward_top_p: float = 1.0
    # Limit in-flight forward remote requests per worker process.
    # 0 or negative means "no limit".
    forward_max_concurrency: int = 8

    # Forward agent prompting: forward sees history + clues each turn.
    forward_system_prompt: str = (
        "You are a forward problem-solving agent.\n"
        "You will be asked one subproblem at a time.\n"
        "Answer that subproblem correctly and briefly.\n"
        "Use the provided KnownClues as context.\n"
        "Do NOT ask questions back.\n"
        "When explicitly asked for FINAL answer, output it as <Answer>...</Answer>."
    )

    # Loop control
    max_rounds: int = 24
    max_queue_items: int = 32

    # Clues constraints (rough proxy for <=10 items and <=512 tokens)
    max_clues: int = 10
    max_clues_chars_total: int = 2200

    # Final reward settings (objective)
    final_enable: bool = True
    final_exact_match: bool = True

    # Step reward components (process-style):
    # - format reward: strict JSON object (no extra text) -> +step_format_reward
    # - schema reward: JSON passes fixed-key + constraint checks -> +step_schema_reward
    step_format_reward: float = 0.05
    step_schema_reward: float = 0.05

    # ------------------------------------------------------------
    # Optional: SFT data collection using a remote "teacher" model
    # ------------------------------------------------------------
    # If enabled, for EACH backward turn input (system+user_state), call teacher model to produce a label JSON.
    # We store ONE single-turn SFT record per backward decision as JSONL:
    #   {"prompt":[...messages...], "response":"...teacher json...", "meta": {...}}
    teacher_enable: bool = False
    teacher_base_url: Optional[str] = None  # e.g. "https://api.openai.com/v1"
    teacher_api_key: Optional[str] = None
    teacher_model: str = "gpt-5-mini"
    teacher_timeout_s: float = 120.0
    teacher_max_output_tokens: int = 256
    # Limit in-flight teacher remote requests per worker process.
    # 0 or negative means "no limit".
    teacher_max_concurrency: int = 4
    # Sampling controls to reduce cost:
    # - every_n: collect at most once every N backward inputs (round index based). 1 means collect every time.
    # - sample_prob: after passing every_n gate, collect with probability in [0,1]. 1.0 means always.
    teacher_every_n: int = 1
    teacher_sample_prob: float = 1.0
    # If empty: reuse backward_system_prompt.
    teacher_system_prompt: str = ""
    # JSONL output path; supports {pid} and {host}.
    teacher_output_path: str = "teacher_sft_{host}_{pid}.jsonl"
    teacher_flush: bool = True
    teacher_fsync: bool = False
    teacher_max_chars_per_text: int = 8000

    # Optional debugging trace in extra_fields (can be large)
    trace_enable: bool = True
    trace_max_chars_per_text: int = 4000


async def _remote_teacher_text(
    *,
    base_url: str,
    api_key: Optional[str],
    model: str,
    messages: list[dict[str, Any]],
    max_output_tokens: int,
    timeout_s: float,
) -> str:
    """
    Call a remote OpenAI/OpenAI-compatible endpoint and return text.
    - For OpenAI official + gpt-5*, prefer Responses API (/v1/responses).
    - Otherwise use Chat Completions (/v1/chat/completions).
    """
    import aiohttp

    base = str(base_url or "").rstrip("/")
    is_openai_official = "api.openai.com" in base.lower()
    is_gpt5 = str(model or "").lower().startswith("gpt-5")

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    timeout = aiohttp.ClientTimeout(total=float(timeout_s))
    async with aiohttp.ClientSession(timeout=timeout) as session:
        if is_openai_official and is_gpt5:
            url = f"{base}/v1/responses" if not base.endswith("/v1") else f"{base}/responses"
            payload: dict[str, Any] = {
                "model": str(model),
                "input": messages,
                "max_output_tokens": int(max_output_tokens),
            }
            try:
                async with session.post(url, headers=headers, json=payload) as r:
                    raw = await r.text()
                    if int(r.status) >= 400:
                        logger.warning("Teacher Responses API HTTP %s. body=%r", r.status, raw[:500])
                        return ""
            except Exception as e:
                logger.warning("Teacher Responses API request failed: %s", e, exc_info=True)
                return ""

            try:
                data = json.loads(raw)
            except Exception:
                return ""
            if isinstance(data, dict) and isinstance(data.get("output_text"), str) and str(data.get("output_text")).strip():
                return str(data.get("output_text") or "")
            # Fallback parse: output[].content[].text
            try:
                outs = data.get("output") or []
                parts: list[str] = []
                if isinstance(outs, list):
                    for o in outs:
                        if not isinstance(o, dict):
                            continue
                        content_list = o.get("content") or []
                        if not isinstance(content_list, list):
                            continue
                        for c in content_list:
                            if isinstance(c, dict) and isinstance(c.get("text"), str):
                                parts.append(str(c.get("text") or ""))
                return "".join(parts)
            except Exception:
                return ""

        url = f"{base}/v1/chat/completions" if not base.endswith("/v1") else f"{base}/chat/completions"
        payload2: dict[str, Any] = {
            "model": str(model),
            "messages": messages,
            "max_tokens": int(max_output_tokens),
            "temperature": 0.0,
            "top_p": 1.0,
        }
        try:
            async with session.post(url, headers=headers, json=payload2) as r:
                raw2 = await r.text()
                if int(r.status) >= 400:
                    logger.warning("Teacher chat.completions HTTP %s. body=%r", r.status, raw2[:500])
                    return ""
        except Exception as e:
            logger.warning("Teacher chat.completions request failed: %s", e, exc_info=True)
            return ""

    try:
        data2 = json.loads(raw2)
    except Exception:
        return ""
    try:
        choices = data2.get("choices") or []
        if not choices:
            return ""
        msg = (choices[0] or {}).get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts2: list[str] = []
            for p in content:
                if isinstance(p, str):
                    parts2.append(p)
                elif isinstance(p, dict) and "text" in p:
                    parts2.append(str(p.get("text") or ""))
            return "".join(parts2)
    except Exception:
        return ""
    return ""

async def _remote_chat_completion_text(
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
    """
    Minimal OpenAI-compatible chat.completions client returning assistant message content as text.
    Designed for vLLM OpenAI server.
    """
    import aiohttp

    base = str(base_url or "").rstrip("/")
    url = f"{base}/v1/chat/completions" if not base.endswith("/v1") else f"{base}/chat/completions"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {
        "model": str(model),
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    timeout = aiohttp.ClientTimeout(total=float(timeout_s))
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(url, headers=headers, json=payload) as r:
                raw = await r.text()
                if int(r.status) >= 400:
                    logger.warning("Forward remote HTTP %s. body=%r", r.status, raw[:500])
                    return ""
        except Exception as e:
            logger.warning("Forward remote request failed: %s", e, exc_info=True)
            return ""

    try:
        data = json.loads(raw)
    except Exception:
        logger.warning("Forward remote invalid JSON. body=%r", (raw or "")[:500])
        return ""

    try:
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
    except Exception:
        return ""
    return ""


@register("backward_agent")
class BackwardAgentLoop(AgentLoopBase):
    """
    Backward agent loop:
    - Backward agent is the policy (this rollout server).
    - Forward agent is an external frozen model behind OpenAI-compatible endpoint.
    """

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

        # Used to strip system prefix for incremental turns (consistent with ToolAgentLoop/StepVerifyAgentLoop).
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )

        # Custom config passed via agent_loop_config_path as kwargs
        cfg = kwargs.get("backward_agent") or kwargs.get("backward") or {}
        if cfg and not isinstance(cfg, dict):
            try:
                from omegaconf import OmegaConf

                cfg = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
            except Exception:
                cfg = {}
        cls.loop_cfg = _BackwardAgentConfig(
            backward_system_prompt=str(cfg.get("backward_system_prompt", _BackwardAgentConfig.backward_system_prompt)),
            forward_base_url=cfg.get("forward_base_url"),
            forward_api_key=cfg.get("forward_api_key"),
            forward_model=cfg.get("forward_model"),
            forward_timeout_s=float(cfg.get("forward_timeout_s", 120.0)),
            forward_max_tokens=int(cfg.get("forward_max_tokens", 1024)),
            forward_temperature=float(cfg.get("forward_temperature", 0.0)),
            forward_top_p=float(cfg.get("forward_top_p", 1.0)),
            forward_max_concurrency=int(cfg.get("forward_max_concurrency", 8)),
            forward_system_prompt=str(cfg.get("forward_system_prompt", _BackwardAgentConfig.forward_system_prompt)),
            max_rounds=int(cfg.get("max_rounds", 24)),
            max_queue_items=int(cfg.get("max_queue_items", 32)),
            max_clues=int(cfg.get("max_clues", 10)),
            max_clues_chars_total=int(cfg.get("max_clues_chars_total", 2200)),
            final_enable=bool(cfg.get("final_enable", True)),
            final_exact_match=bool(cfg.get("final_exact_match", True)),
            step_format_reward=float(cfg.get("step_format_reward", 0.05)),
            step_schema_reward=float(cfg.get("step_schema_reward", 0.05)),
            teacher_enable=bool(cfg.get("teacher_enable", False)),
            teacher_base_url=cfg.get("teacher_base_url"),
            teacher_api_key=cfg.get("teacher_api_key"),
            teacher_model=str(cfg.get("teacher_model", "gpt-5-mini")),
            teacher_timeout_s=float(cfg.get("teacher_timeout_s", 120.0)),
            teacher_max_output_tokens=int(cfg.get("teacher_max_output_tokens", 256)),
            teacher_max_concurrency=int(cfg.get("teacher_max_concurrency", 4)),
            teacher_every_n=int(cfg.get("teacher_every_n", 1)),
            teacher_sample_prob=float(cfg.get("teacher_sample_prob", 1.0)),
            teacher_system_prompt=str(cfg.get("teacher_system_prompt", "")),
            teacher_output_path=str(cfg.get("teacher_output_path", "teacher_sft_{host}_{pid}.jsonl")),
            teacher_flush=bool(cfg.get("teacher_flush", True)),
            teacher_fsync=bool(cfg.get("teacher_fsync", False)),
            teacher_max_chars_per_text=int(cfg.get("teacher_max_chars_per_text", 8000)),
            trace_enable=bool(cfg.get("trace_enable", True)),
            trace_max_chars_per_text=int(cfg.get("trace_max_chars_per_text", 4000)),
        )

        # Concurrency limiters (per worker process)
        cls._forward_semaphore = None
        cls._teacher_semaphore = None
        try:
            import asyncio

            mc_f = int(getattr(cls.loop_cfg, "forward_max_concurrency", 0) or 0)
            cls._forward_semaphore = asyncio.Semaphore(mc_f) if mc_f and mc_f > 0 else None
            mc_t = int(getattr(cls.loop_cfg, "teacher_max_concurrency", 0) or 0)
            cls._teacher_semaphore = asyncio.Semaphore(mc_t) if mc_t and mc_t > 0 else None
        except Exception:
            cls._forward_semaphore = None
            cls._teacher_semaphore = None

    async def _call_forward_remote(self, **kwargs) -> str:
        sem = getattr(self.__class__, "_forward_semaphore", None)
        if sem is None:
            return await _remote_chat_completion_text(**kwargs)
        async with sem:
            return await _remote_chat_completion_text(**kwargs)

    async def _call_teacher_remote(self, **kwargs) -> str:
        sem = getattr(self.__class__, "_teacher_semaphore", None)
        if sem is None:
            return await _remote_teacher_text(**kwargs)
        async with sem:
            return await _remote_teacher_text(**kwargs)

    def _should_collect_teacher(self, round_idx: int) -> bool:
        cfg = self.loop_cfg
        if not bool(getattr(cfg, "teacher_enable", False)):
            return False
        if not str(getattr(cfg, "teacher_base_url", "") or "").strip():
            return False
        every_n = int(getattr(cfg, "teacher_every_n", 1) or 1)
        if every_n < 1:
            every_n = 1
        if (int(round_idx) % every_n) != 0:
            return False
        p = float(getattr(cfg, "teacher_sample_prob", 1.0))
        if p >= 1.0:
            return True
        if p <= 0.0:
            return False
        return random.random() < p

    def _teacher_sft_path(self) -> str:
        import socket

        p = str(getattr(self.loop_cfg, "teacher_output_path", "teacher_sft_{host}_{pid}.jsonl") or "")
        try:
            p = p.format(pid=os.getpid(), host=socket.gethostname())
        except Exception:
            p = str(p)
        try:
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        except Exception:
            pass
        return p

    def _truncate_teacher_text(self, s: Any) -> str:
        maxc = int(getattr(self.loop_cfg, "teacher_max_chars_per_text", 8000) or 8000)
        t = str(s or "")
        if maxc > 0 and len(t) > maxc:
            return t[:maxc]
        return t

    def _maybe_write_teacher_sft(self, rec: dict[str, Any]) -> None:
        if not bool(getattr(self.loop_cfg, "teacher_enable", False)):
            return
        try:
            path = self._teacher_sft_path()
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if bool(getattr(self.loop_cfg, "teacher_flush", True)):
                    f.flush()
                    if bool(getattr(self.loop_cfg, "teacher_fsync", False)):
                        os.fsync(f.fileno())
        except Exception as e:
            logger.warning("Failed to write teacher SFT record (ignored): %s", e)

    async def _encode_full_messages(self, messages: list[dict[str, Any]]) -> list[int]:
        return await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
            ),
        )

    async def _encode_incremental_messages(self, add_messages: list[dict[str, Any]]) -> list[int]:
        ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
        )
        return ids[len(self.system_prompt) :]

    def _build_backward_user_state(
        self,
        *,
        problem_text: str,
        known_clues: list[str],
        asked_subproblems: list[str],
        forward_step_solutions: list[str],
        last_forward_solution: str,
        pending_queue: deque[str],
        forward_last_was_final: bool,
        final_checked_failed_once: bool,
    ) -> str:
        clues_txt = "\n".join([f"- {c}" for c in known_clues]) if known_clues else "(empty)"
        subs_txt = "\n".join([f"{i+1}. {s}" for i, s in enumerate(asked_subproblems)]) if asked_subproblems else "(none)"
        q_txt = "\n".join([f"- {s}" for s in list(pending_queue)]) if pending_queue else "(empty)"
        last_sol = (last_forward_solution or "").strip()

        # Show as many (subproblem, solution) pairs as possible under a soft char budget to avoid prompt blow-up.
        # We keep the MOST RECENT pairs first (from the end).
        pairs_txt = "(none)"
        if asked_subproblems and forward_step_solutions:
            n = min(len(asked_subproblems), len(forward_step_solutions))
            rows_rev: list[str] = []
            total = 0
            budget = 4000
            for i in range(n - 1, -1, -1):
                sp = str(asked_subproblems[i] or "").strip()
                sol = str(forward_step_solutions[i] or "").strip()
                if len(sol) > 240:
                    sol = sol[:240] + "..."
                row = f"{i+1}. Subproblem: {sp}\n   StepSolution: {sol}"
                add = len(row) + 2
                if rows_rev and total + add > budget:
                    break
                rows_rev.append(row)
                total += add
            rows = list(reversed(rows_rev))
            pairs_txt = "\n".join(rows) if rows else "(none)"

        # JSON contract: keep it explicit and machine-parseable.
        # Backward sees: total problem, all subproblems (titles only), forward step solution, FIFO queue.
        # It must decide what to do next.
        # IMPORTANT: make this easy for small models.
        # - fixed keys (always present) -> stable extraction
        # - binary correctness only (no unknown)
        # - progressive choice constraints (correct vs incorrect branches)
        instr = (
            "You MUST output ONLY one STRICT JSON object (no extra characters).\n"
            "The JSON MUST contain the following fixed keys:\n"
            "{\n"
            '  "is_correct": true | false,\n'
            '  "hint_updates": ["..."],\n'
            '  "next_subproblem": "",\n'
            '  "request_final": true | false,\n'
            '  "restate_subproblem": "",\n'
            '  "queue_push": []\n'
            "}\n"
            "Constraints (MUST follow):\n"
            "1) If is_correct=true (you believe the last forward answer is correct):\n"
            "   - You MAY output hint_updates (can be empty).\n"
            '   - If pending_queue is NON-empty: set next_subproblem="" and request_final=false.\n'
            "   - Else (queue empty): choose EXACTLY ONE:\n"
            '       A) request_final=true and next_subproblem=""\n'
            "       B) request_final=false and next_subproblem is a NON-empty string\n"
            '   - In this case you MUST set restate_subproblem="" and queue_push=[]\n'
            "2) If is_correct=false (you believe the last forward answer is incorrect):\n"
            "   - You MUST choose EXACTLY ONE option:\n"
            "       A) restate_subproblem: a clearer restatement of the SAME subproblem (non-empty string)\n"
            "       B) queue_push: a NON-empty list of smaller subproblems to solve first\n"
            '   - In this case you MUST set hint_updates=[], next_subproblem="", request_final=false\n'
            "3) Only set request_final=true when pending_queue is empty and you are ready for the final answer.\n"
            f"4) Final attempt limit: final_checked_failed_once={bool(final_checked_failed_once)}.\n"
            "   If it is true and the final answer is still objectively wrong again, the episode will end with 0 reward.\n"
        )

        phase = "FINAL_VERIFICATION" if forward_last_was_final else "STEP_VERIFICATION_AND_PLANNING"
        return (
            f"PHASE: {phase}\n\n"
            f"ORIGINAL PROBLEM:\n{problem_text}\n\n"
            f"KNOWN CLUES (shared across turns, max 10, keep short):\n{clues_txt}\n\n"
            f"HISTORY OF SUBPROBLEMS (no dialogue content):\n{subs_txt}\n\n"
            f"SUBPROBLEMS + FORWARD STEP SOLUTIONS (recent):\n{pairs_txt}\n\n"
            f"PENDING FIFO QUEUE (priority to ask next):\n{q_txt}\n\n"
            f"LATEST FORWARD STEP SOLUTION:\n{last_sol if last_sol else '(none yet)'}\n\n"
            f"{instr}"
        )

    def _build_forward_user_prompt(self, *, problem_text: str, known_clues: list[str], request: str) -> str:
        clues_txt = "\n".join([f"- {c}" for c in known_clues]) if known_clues else "(none)"
        return (
            f"Original problem:\n{problem_text}\n\n"
            f"So far, from previous subproblems, we know:\n{clues_txt}\n\n"
            f"Next subproblem (please solve it correctly and briefly):\n{request}\n"
        )

    def _truncate_trace(self, s: Any) -> str:
        maxc = int(getattr(self.loop_cfg, "trace_max_chars_per_text", 4000) or 4000)
        t = str(s or "")
        if maxc > 0 and len(t) > maxc:
            return t[:maxc]
        return t

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # -------------------------
        # Episode initialization
        # -------------------------
        cfg = self.loop_cfg
        request_id = uuid4().hex
        metrics: dict[str, Any] = {}

        raw_messages = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info") or {}
        reward_model = kwargs.get("reward_model") or {}

        # Determine problem text (robust): prefer explicit fields, fallback to last user message.
        problem_text = (
            extra_info.get("problem_text")
            or extra_info.get("question")
            or extra_info.get("question_hint")
            or reward_model.get("question")
            or _extract_last_user_text(raw_messages)
            or ""
        )
        problem_text = str(problem_text or "")

        ground_truth = reward_model.get("ground_truth") or extra_info.get("ground_truth")
        ground_truth = str(ground_truth) if ground_truth is not None else None

        # Validate forward remote config
        if not cfg.forward_base_url or not cfg.forward_model:
            raise ValueError(
                "BackwardAgentLoop requires forward_base_url and forward_model. "
                "Set them under agent_loop_config_path kwargs: backward_agent.forward_base_url/forward_model."
            )

        # Backward policy chat messages (token-consistent) start with our system prompt.
        backward_messages: list[dict[str, Any]] = [{"role": "system", "content": str(cfg.backward_system_prompt or "")}]
        # We'll add a structured state message every round (including round 0) to force JSON output.
        initial_user_state = self._build_backward_user_state(
            problem_text=problem_text,
            known_clues=[],
            asked_subproblems=[],
            forward_step_solutions=[],
            last_forward_solution="",
            pending_queue=deque(),
            forward_last_was_final=False,
            final_checked_failed_once=False,
        )
        backward_messages.append({"role": "user", "content": initial_user_state})
        prompt_ids = await self._encode_full_messages(backward_messages)

        # Teacher SFT collection for round0 input
        if self._should_collect_teacher(0):
            teacher_sys = str(getattr(cfg, "teacher_system_prompt", "") or "").strip() or str(cfg.backward_system_prompt or "")
            teacher_msgs = [{"role": "system", "content": teacher_sys}, {"role": "user", "content": str(initial_user_state)}]
            teacher_text = await self._call_teacher_remote(
                base_url=str(cfg.teacher_base_url),
                api_key=cfg.teacher_api_key,
                model=str(cfg.teacher_model),
                messages=teacher_msgs,
                max_output_tokens=int(cfg.teacher_max_output_tokens),
                timeout_s=float(cfg.teacher_timeout_s),
            )
            self._maybe_write_teacher_sft(
                {
                    "ts": time.time(),
                    "request_id": request_id,
                    "round": 0,
                    "prompt": teacher_msgs,
                    "response": self._truncate_teacher_text(teacher_text),
                    "meta": {
                        "problem_text": self._truncate_teacher_text(problem_text),
                        "pending_queue": [],
                        "asked_subproblems": [],
                        "forward_step_solutions_n": 0,
                    },
                }
            )

        # Forward remote chat history (string-based). Forward must see full history + clues each turn.
        forward_messages: list[dict[str, Any]] = [
            {"role": "system", "content": str(cfg.forward_system_prompt or "")},
        ]

        # Shared state
        known_clues: list[str] = []
        asked_subproblems: list[str] = []
        forward_step_solutions: list[str] = []
        pending_queue: deque[str] = deque()

        # Token-level metadata for BACKWARD model only
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        step_token_spans: list[tuple[int, int]] = []
        # Step reward components:
        # - turn_scores: total per-turn step reward used for rm_scores construction
        # - turn_scores_format: strict JSON format reward component
        # - turn_scores_schema: schema/constraint validation reward component
        turn_scores: list[float] = []
        turn_scores_format: list[float] = []
        turn_scores_schema: list[float] = []
        # For logging/monitoring (episode-level stats)
        strict_json_flags: list[int] = []
        schema_ok_flags: list[int] = []

        # Trace/debug
        trace: dict[str, Any] = {
            "problem_text": self._truncate_trace(problem_text),
            "ground_truth": self._truncate_trace(ground_truth) if ground_truth is not None else None,
            "steps": [],
        }

        assistant_turns = 0
        user_turns = 0

        last_forward_solution = ""
        forward_last_was_final = False
        episode_done = False
        final_reward = 0.0
        final_checked_failed_once = False

        # -------------------------
        # Main loop
        # -------------------------
        for round_idx in range(int(cfg.max_rounds or 24)):
            if episode_done:
                break

            # 1) Backward agent: judge + plan next step (or propose first step if no forward solution yet).
            if round_idx > 0:
                user_state = self._build_backward_user_state(
                    problem_text=problem_text,
                    known_clues=known_clues,
                    asked_subproblems=asked_subproblems,
                    forward_step_solutions=forward_step_solutions,
                    last_forward_solution=last_forward_solution,
                    pending_queue=pending_queue,
                    forward_last_was_final=forward_last_was_final,
                    final_checked_failed_once=final_checked_failed_once,
                )
                add_messages = [{"role": "user", "content": user_state}]
                backward_messages.extend(add_messages)
                user_turns += 1

                # Teacher SFT collection for each backward input state (single-turn)
                if self._should_collect_teacher(int(round_idx)):
                    teacher_sys = (
                        str(getattr(cfg, "teacher_system_prompt", "") or "").strip()
                        or str(cfg.backward_system_prompt or "")
                    )
                    teacher_msgs = [{"role": "system", "content": teacher_sys}, {"role": "user", "content": str(user_state)}]
                    teacher_text = await self._call_teacher_remote(
                        base_url=str(cfg.teacher_base_url),
                        api_key=cfg.teacher_api_key,
                        model=str(cfg.teacher_model),
                        messages=teacher_msgs,
                        max_output_tokens=int(cfg.teacher_max_output_tokens),
                        timeout_s=float(cfg.teacher_timeout_s),
                    )
                    self._maybe_write_teacher_sft(
                        {
                            "ts": time.time(),
                            "request_id": request_id,
                            "round": int(round_idx),
                            "prompt": teacher_msgs,
                            "response": self._truncate_teacher_text(teacher_text),
                            "meta": {
                                "problem_text": self._truncate_teacher_text(problem_text),
                                "pending_queue": list(pending_queue),
                                "asked_subproblems": list(asked_subproblems),
                                "forward_step_solutions_n": int(len(forward_step_solutions)),
                            },
                        }
                    )

                user_ids = await self._encode_incremental_messages(add_messages)
                if len(response_mask) + len(user_ids) >= self.response_length:
                    break
                prompt_ids += user_ids
                response_mask += [0] * len(user_ids)
                if response_logprobs:
                    response_logprobs += [0.0] * len(user_ids)

            with simple_timer("generate_sequences", metrics):
                out = await self.server_manager.generate(
                    request_id=f"backward_{request_id}",
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=None,
                )

            assistant_turns += 1
            gen_ids = out.token_ids
            start = len(response_mask)
            end = start + len(gen_ids)
            step_token_spans.append((start, end))
            turn_scores.append(0.0)
            turn_scores_format.append(0.0)
            turn_scores_schema.append(0.0)

            prompt_ids += gen_ids
            response_mask += [1] * len(gen_ids)
            if out.log_probs:
                response_logprobs += out.log_probs

            backward_text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            )
            backward_messages.append({"role": "assistant", "content": str(backward_text or "")})

            # Snapshot queue state BEFORE applying this action (needed for schema constraints).
            pending_queue_nonempty_before = bool(pending_queue)

            # Step format reward: strict JSON object, no extra text.
            strict_obj = _parse_strict_json_object(str(backward_text or ""))
            step_format_reward = float(cfg.step_format_reward) if strict_obj is not None else 0.0
            strict_json_flags.append(1 if strict_obj is not None else 0)

            # Step schema reward (separate from format reward):
            # must be strict JSON AND pass schema+constraint validation.
            schema_ok = (
                strict_obj is not None
                and _validate_backward_action_schema(strict_obj, pending_queue_nonempty=pending_queue_nonempty_before)
            )
            step_schema_reward = float(cfg.step_schema_reward) if schema_ok else 0.0
            schema_ok_flags.append(1 if schema_ok else 0)

            # For behavior, prefer strict parse; fallback to best-effort extraction.
            action = strict_obj if strict_obj is not None else (_safe_json_first_object(str(backward_text or "")) or {})

            # overwrite the last appended default turn scores with the actual step rewards
            turn_scores_format[-1] = float(step_format_reward)
            turn_scores_schema[-1] = float(step_schema_reward)
            turn_scores[-1] = float(step_format_reward) + float(step_schema_reward)

            # Apply backward updates (clues + queue)
            hint_updates = action.get("hint_updates") if isinstance(action, dict) else None
            if isinstance(hint_updates, list) and hint_updates:
                known_clues.extend([str(x) for x in hint_updates if str(x or "").strip()])
                known_clues = _clamp_clues(
                    known_clues, max_items=int(cfg.max_clues), max_chars_total=int(cfg.max_clues_chars_total)
                )

            queue_push = action.get("queue_push") if isinstance(action, dict) else None
            if isinstance(queue_push, list) and queue_push:
                for item in queue_push:
                    s = str(item or "").strip()
                    if not s:
                        continue
                    pending_queue.append(s)
                # Clamp queue
                while int(cfg.max_queue_items or 32) > 0 and len(pending_queue) > int(cfg.max_queue_items):
                    pending_queue.popleft()

            # New action schema
            is_correct = bool(action.get("is_correct", True)) if isinstance(action, dict) else True
            request_final = bool(action.get("request_final", False)) if isinstance(action, dict) else False
            restate = str(action.get("restate_subproblem") or "").strip() if isinstance(action, dict) else ""
            next_sub = str(action.get("next_subproblem") or "").strip() if isinstance(action, dict) else ""

            # FINAL_VERIFICATION:
            # - Backward must output is_correct=true to claim final answer correct.
            # - Objective final reward is ONLY based on forward final answer correctness vs ground_truth.
            # - If objective check fails, we allow at most ONE extra attempt (restate or decompose),
            #   controlled by final_checked_failed_once flag.
            if forward_last_was_final and bool(cfg.final_enable):
                extracted = _extract_final_answer(last_forward_solution) or ""
                objective_ok = None
                if ground_truth is not None and bool(cfg.final_exact_match):
                    objective_ok = (str(extracted).strip() == str(ground_truth).strip())
                else:
                    objective_ok = bool(str(extracted).strip())

                # Final reward rules:
                # - If forward final answer is objectively correct:
                #     - backward says correct  => 1.0
                #     - backward says incorrect=> 0.5
                #   In both cases: force terminate immediately.
                # - If forward final answer is objectively wrong:
                #     - backward says correct  => terminate immediately with 0, and NO step reward for this turn
                #     - backward says incorrect=> allow at most ONE correction attempt; if still wrong again => 0.
                if objective_ok:
                    final_reward = 1.0 if bool(is_correct) else 0.5
                    episode_done = True
                    break

                # objective wrong
                if bool(is_correct):
                    # backward mistakenly accepts a wrong final answer -> immediate 0 and no step reward for this turn
                    final_reward = 0.0
                    turn_scores[-1] = 0.0
                    turn_scores_format[-1] = 0.0
                    turn_scores_schema[-1] = 0.0
                    episode_done = True
                    break

                # backward also says wrong -> correction attempt allowed at most once
                if final_checked_failed_once:
                    final_reward = 0.0
                    episode_done = True
                    break
                final_checked_failed_once = True

            # Trace step
            if bool(cfg.trace_enable):
                trace["steps"].append(
                    {
                        "round": int(round_idx),
                        "backward_raw": self._truncate_trace(backward_text),
                        "backward_action": action,
                        "known_clues": list(known_clues),
                        "pending_queue": list(pending_queue),
                        "asked_subproblems": list(asked_subproblems),
                        "forward_step_solutions_n": int(len(forward_step_solutions)),
                        "last_forward_solution": self._truncate_trace(last_forward_solution),
                        "forward_last_was_final": bool(forward_last_was_final),
                        "is_correct": bool(is_correct),
                        "request_final": request_final,
                        "final_checked_failed_once": bool(final_checked_failed_once),
                        "step_format_reward": float(step_format_reward),
                        "step_schema_reward": float(step_schema_reward),
                        "schema_ok": bool(schema_ok),
                    }
                )

            # Termination budgets
            if len(response_mask) >= self.response_length:
                break
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                break
            if self.max_user_turns and user_turns >= self.max_user_turns:
                break

            # 2) Decide what to ask the forward agent next (queue has priority).
            forward_request: Optional[str] = None
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
                # If backward gives nothing, force it to propose a subproblem to avoid deadlock.
                forward_request = "Propose the next smallest helpful subproblem to solve."
                forward_last_was_final = False

            # Execution-level guardrail: if queue is non-empty, we will prioritize it anyway.
            # So ignore a non-empty next_subproblem to keep behavior consistent with the prompt constraints.
            if pending_queue and not restate and not request_final:
                # next_sub is advisory only when queue is empty
                pass

            asked_subproblems.append(str(forward_request))

            # 3) Forward agent remote call
            forward_user = self._build_forward_user_prompt(
                problem_text=problem_text, known_clues=known_clues, request=str(forward_request)
            )
            forward_messages.append({"role": "user", "content": forward_user})

            t0 = time.time()
            forward_text = await self._call_forward_remote(
                base_url=str(cfg.forward_base_url),
                api_key=cfg.forward_api_key,
                model=str(cfg.forward_model),
                messages=forward_messages,
                max_tokens=int(cfg.forward_max_tokens),
                temperature=float(cfg.forward_temperature),
                top_p=float(cfg.forward_top_p),
                timeout_s=float(cfg.forward_timeout_s),
            )
            dt = time.time() - t0
            metrics.setdefault("forward_remote_s", 0.0)
            try:
                metrics["forward_remote_s"] = float(metrics["forward_remote_s"]) + float(dt)
            except Exception:
                metrics["forward_remote_s"] = float(dt)

            last_forward_solution = str(forward_text or "")
            forward_messages.append({"role": "assistant", "content": last_forward_solution})
            forward_step_solutions.append(last_forward_solution)

            # Note: we intentionally do NOT terminate immediately after forward final answer.
            # Backward must perform a final verification turn and output judge="correct" to end the episode.

        # -------------------------
        # Build AgentLoopOutput (BACKWARD tokens only)
        # -------------------------
        response_ids = prompt_ids[-len(response_mask) :] if response_mask else []
        prompt_only_ids = prompt_ids[: len(prompt_ids) - len(response_mask)] if response_mask else prompt_ids

        extra_fields: dict[str, Any] = {
            "turn_scores": turn_scores,
            "turn_scores_format": turn_scores_format,
            "turn_scores_schema": turn_scores_schema,
            "step_token_spans": step_token_spans,
            "final_reward": float(final_reward),
            # Debugging / analysis
            "assistant_turns": int(assistant_turns),
            "user_turns": int(user_turns),
        }
        if bool(cfg.trace_enable):
            extra_fields["backward_forward_trace"] = trace

        # -------------------------
        # Episode-level metrics (logged by trainer -> wandb)
        # -------------------------
        try:
            n_turns = int(len(turn_scores))
            total_resp_tokens = int(sum((e - s) for (s, e) in step_token_spans)) if step_token_spans else 0
            if n_turns > 0:
                # Per-sample reward stats (raw scalars) exported via reward_extra_info so PPO trainer can log them.
                # NOTE: AgentLoopManager currently discards per-sample `metrics` when aggregating timing, so this is
                # the most reliable way to surface step-reward statistics in training logs.
                extra_fields["reward_extra_info"] = {
                    "backward/step_reward": float(sum(turn_scores) / n_turns),
                    "backward/format_reward": float(sum(turn_scores_format) / n_turns),
                    "backward/schema_reward": float(sum(turn_scores_schema) / n_turns),
                    "backward/strict_json_rate": float(sum(strict_json_flags) / max(1, len(strict_json_flags))),
                    "backward/schema_ok_rate": float(sum(schema_ok_flags) / max(1, len(schema_ok_flags))),
                    "backward/turns": float(n_turns),
                    "backward/response_tokens": float(total_resp_tokens),
                    "backward/response_tokens_per_turn": float(total_resp_tokens / max(1, n_turns)),
                    "backward/final_reward": float(final_reward),
                }
                metrics["backward/step_reward/mean"] = float(sum(turn_scores) / n_turns)
                metrics["backward/step_reward/sum"] = float(sum(turn_scores))
                metrics["backward/format_reward/mean"] = float(sum(turn_scores_format) / n_turns)
                metrics["backward/format_reward/sum"] = float(sum(turn_scores_format))
                metrics["backward/schema_reward/mean"] = float(sum(turn_scores_schema) / n_turns)
                metrics["backward/schema_reward/sum"] = float(sum(turn_scores_schema))
                metrics["backward/strict_json_rate"] = float(sum(strict_json_flags) / max(1, len(strict_json_flags)))
                metrics["backward/schema_ok_rate"] = float(sum(schema_ok_flags) / max(1, len(schema_ok_flags)))
                metrics["backward/turns"] = float(n_turns)
                metrics["backward/strict_json_count"] = float(sum(strict_json_flags))
                metrics["backward/schema_ok_count"] = float(sum(schema_ok_flags))
                metrics["backward/response_tokens/sum"] = float(total_resp_tokens)
                metrics["backward/response_tokens/per_turn_mean"] = float(total_resp_tokens / max(1, n_turns))
            else:
                extra_fields.setdefault(
                    "reward_extra_info",
                    {
                        "backward/step_reward": 0.0,
                        "backward/format_reward": 0.0,
                        "backward/schema_reward": 0.0,
                        "backward/strict_json_rate": 0.0,
                        "backward/schema_ok_rate": 0.0,
                        "backward/turns": 0.0,
                        "backward/response_tokens": 0.0,
                        "backward/response_tokens_per_turn": 0.0,
                        "backward/final_reward": float(final_reward),
                    },
                )
                metrics.setdefault("backward/step_reward/mean", 0.0)
                metrics.setdefault("backward/format_reward/mean", 0.0)
                metrics.setdefault("backward/schema_reward/mean", 0.0)
                metrics.setdefault("backward/strict_json_rate", 0.0)
                metrics.setdefault("backward/schema_ok_rate", 0.0)
                metrics.setdefault("backward/turns", 0.0)
                metrics.setdefault("backward/strict_json_count", 0.0)
                metrics.setdefault("backward/schema_ok_count", 0.0)
                metrics.setdefault("backward/response_tokens/sum", 0.0)
                metrics.setdefault("backward/response_tokens/per_turn_mean", 0.0)
            metrics["backward/final_reward"] = float(final_reward)
        except Exception:
            # Never fail rollout due to logging.
            pass

        output = AgentLoopOutput(
            prompt_ids=prompt_only_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            multi_modal_data={},
            # Avoid async reward computation; rewards are derived from extra_fields by AgentLoopWorkerBase._postprocess()
            reward_score=0.0,
            num_turns=int(assistant_turns + user_turns + 1),
            metrics=metrics,
            extra_fields=extra_fields,
        )
        return output


