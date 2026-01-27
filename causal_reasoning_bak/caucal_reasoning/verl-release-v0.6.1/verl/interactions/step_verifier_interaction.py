"""
Step-wise verifier interaction.

Goal
-----
Provide a simple multi-turn "environment" that:
- feeds a sequence of sub-questions to the policy model (as user turns)
- after each assistant turn, verifies the assistant's last answer
- returns a per-step reward (for step-level credit assignment / analysis)

How it is used in verl
----------------------
In SGLang multi-turn rollout, after the model produces an assistant message, the rollout worker enters
`AsyncRolloutRequestStateEnum.INTERACTING` and calls:

    interaction.generate_response(request_id, messages, **interaction_kwargs)

The returned `interaction_responses` becomes the *next user message* appended to the conversation.
The returned `reward` is appended to `user_turn_rewards`, and then stored in `reward_scores` under
the key `"user_turn_rewards"`.

Note: by default, Verl reward managers still place a single scalar reward on the *last token* of the
response. If you want training to use step rewards directly, implement a custom reward function that
aggregates `extra_info["rollout_reward_scores"]["user_turn_rewards"]` into a scalar (or implement a
custom reward manager that maps step rewards to token-level positions).
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional
from uuid import uuid4

from verl.interactions.base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


VerifierFn = Callable[[str, Optional[Any], dict[str, Any]], float]


def _import_by_path(dotted_path: str) -> Any:
    """Import `a.b.c:obj` or `a.b.c.obj`."""
    if ":" in dotted_path:
        module_name, attr = dotted_path.split(":", 1)
    else:
        module_name, attr = dotted_path.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, attr)


def _default_exact_match_verifier(assistant_text: str, expected: Optional[Any], ctx: dict[str, Any]) -> float:
    """A conservative default verifier: 1 iff normalized strings match; else 0."""
    if expected is None:
        return 0.0
    a = str(assistant_text).strip()
    e = str(expected).strip()
    return 1.0 if a == e else 0.0


def _extract_last_assistant_text(messages: list[dict[str, Any]]) -> str:
    for item in reversed(messages):
        if item.get("role") == "assistant":
            return str(item.get("content", ""))
    return ""


@dataclass
class _SessionState:
    subproblems: list[str]
    expected_answers: list[Any] | None
    turn: int = 0
    assistant_steps: list[str] = None  # type: ignore[assignment]
    step_rewards: list[float] = None  # type: ignore[assignment]
    max_turns: int = 32
    stop_on_incorrect: bool = False
    feedback_mode: str = "none"  # none | binary

    def __post_init__(self) -> None:
        if self.assistant_steps is None:
            self.assistant_steps = []
        if self.step_rewards is None:
            self.step_rewards = []


class StepVerifierInteraction(BaseInteraction):
    """
    A minimal step-wise interaction.

    Expected runtime kwargs (passed via `interaction_kwargs`):
    - name: required by ToolAgentLoop; ignored by this class but kept for consistency
    - subproblems: list[str] (required)  # next user messages in order
    - expected_answers: list[Any] (optional)  # per-step expected answer, aligned with subproblems
    - max_turns: int (optional)
    - stop_on_incorrect: bool (optional)
    - feedback_mode: "none" | "binary" (optional)
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._sessions: dict[str, _SessionState] = {}

        verifier_path = (config or {}).get("verifier_fn", None)
        if verifier_path:
            try:
                fn = _import_by_path(verifier_path)
                if not callable(fn):
                    raise TypeError(f"verifier_fn must be callable, got {type(fn).__name__}")
                self._verifier: VerifierFn = fn  # type: ignore[assignment]
            except Exception as e:
                logger.warning("Failed to load verifier_fn=%r, fallback to exact match. Error: %s", verifier_path, e)
                self._verifier = _default_exact_match_verifier
        else:
            self._verifier = _default_exact_match_verifier

    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        subproblems = kwargs.get("subproblems")
        if not isinstance(subproblems, list) or not all(isinstance(x, str) for x in subproblems):
            raise ValueError("StepVerifierInteraction requires `subproblems: list[str]` in interaction_kwargs")

        expected_answers = kwargs.get("expected_answers", None)
        if expected_answers is not None and not isinstance(expected_answers, list):
            raise ValueError("`expected_answers` must be a list when provided")

        max_turns = int(kwargs.get("max_turns", self.config.get("max_turns", 32)))
        stop_on_incorrect = bool(kwargs.get("stop_on_incorrect", self.config.get("stop_on_incorrect", False)))
        feedback_mode = str(kwargs.get("feedback_mode", self.config.get("feedback_mode", "none"))).lower()
        if feedback_mode not in {"none", "binary"}:
            raise ValueError("`feedback_mode` must be one of: none, binary")

        self._sessions[instance_id] = _SessionState(
            subproblems=subproblems,
            expected_answers=expected_answers,
            max_turns=max_turns,
            stop_on_incorrect=stop_on_incorrect,
            feedback_mode=feedback_mode,
        )
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        state = self._sessions[instance_id]
        state.turn += 1

        # Step index equals how many assistant steps we have already verified.
        step_idx = len(state.assistant_steps)
        assistant_text = _extract_last_assistant_text(messages)
        state.assistant_steps.append(assistant_text)

        expected = None
        if state.expected_answers is not None and step_idx < len(state.expected_answers):
            expected = state.expected_answers[step_idx]

        ctx = {
            "instance_id": instance_id,
            "turn": state.turn,
            "step_idx": step_idx,
            "messages": messages,
        }
        reward = float(self._verifier(assistant_text, expected, ctx))
        # clamp to [0, 1] (you can relax this if your algorithm expects other ranges)
        reward = max(0.0, min(1.0, reward))
        state.step_rewards.append(reward)

        is_correct = reward > 0.0

        # Decide whether to terminate
        exhausted = step_idx + 1 >= len(state.subproblems)
        hit_max_turns = state.turn >= state.max_turns
        should_terminate = exhausted or hit_max_turns or (state.stop_on_incorrect and not is_correct)

        # Next user message
        if should_terminate:
            next_user = "Interaction finished."
        else:
            next_user = state.subproblems[step_idx + 1]
            if state.feedback_mode == "binary":
                verdict = "correct" if is_correct else "incorrect"
                next_user = f"[verifier] step {step_idx + 1}: {verdict}\n\n{next_user}"

        metrics = {
            "turn": state.turn,
            "step_idx": step_idx,
            "reward": reward,
            "is_correct": is_correct,
        }
        return should_terminate, next_user, reward, metrics

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        self._sessions.pop(instance_id, None)




