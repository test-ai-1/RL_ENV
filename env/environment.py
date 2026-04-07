"""Gym-like environment backed on a dataset with multi-step filter/sum interaction."""

from __future__ import annotations

import copy
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

from .action_validation import (
    PenaltyKind,
    apply_penalty,
    classify_invalid_for_state,
    classify_missing_column,
    classify_useless,
)
from .actions import Action as DataAction
from .observations import Observation
from .reward import answer_matches, confidence_adjusted_reward, shaped_reward
from .table_ops import (
    filter_working_set,
    max_working_set,
    parse_number,
    select_column_working,
    sum_working_set,
)

Answer = Union[int, float, str]

_DEFAULT_MAX_STEPS = 4


class DataEnv:
    """
    Multi-step RL loop: ``reset`` → repeated ``step`` until the gold answer is produced
    (on a ``sum``) or ``max_steps`` is reached.

    State:
        - ``_working``: rows after cumulative filters (observation ``data``).
        - ``_selected_column``: column from the latest successful action.
        - ``_intermediate_results``: append-only log of step outcomes.
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        question: str,
        *,
        correct_answer: Answer,
        max_steps: int = _DEFAULT_MAX_STEPS,
    ) -> None:
        if max_steps < 3:
            raise ValueError("max_steps must be at least 3 (typical horizon is 3 or 4).")
        self._data = [copy.deepcopy(r) for r in data]
        self.question = question
        self.correct_answer = correct_answer
        self.max_steps = max_steps

        self._working: List[Dict[str, Any]] = []
        self._selected_column: Optional[str] = None
        self._intermediate_results: List[Dict[str, Any]] = []
        self._episode_step: int = 0
        self._terminated: bool = False
        self._observation: Optional[Observation] = None
        self._select_credit: float = 0.0
        self._filter_credit: float = 0.0
        self._final_credit: float = 0.0

    def reset(self) -> Observation:
        """Start a new episode; working set is a deep copy of the original table."""
        self._terminated = False
        self._episode_step = 0
        self._working = [copy.deepcopy(r) for r in self._data]
        self._selected_column = None
        self._intermediate_results = []
        self._observation = self._build_observation()
        self._select_credit = 0.0
        self._filter_credit = 0.0
        self._final_credit = 0.0
        return self._observation.model_copy()

    def step(self, action: DataAction) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Apply one of ``select_column``, ``filter``, ``sum``, or ``max``.

        Bad actions yield negative rewards (see ``action_validation``); the environment
        does not raise for user/agent errors (only after ``done`` + ``step``).
        """
        if self._terminated:
            raise RuntimeError("Episode finished; call reset() before step().")

        working_before = copy.deepcopy(self._working)
        prev_selected = self._selected_column
        prev_progress_total = (
            self._select_credit + self._filter_credit + self._final_credit
        )

        self._episode_step += 1

        computed: Optional[float] = None
        op_error: Optional[str] = None
        base_reward = 0.0
        base_reason = "noop"
        penalty_kind: PenaltyKind = "none"
        penalty_detail = ""
        exception_trace: Optional[str] = None
        col_ok: Optional[str] = None

        try:
            if classify_missing_column(self._working, action):
                penalty_kind = "invalid"
                penalty_detail = "missing_column"
                base_reward, base_reason = 0.0, "validation_failed"
            else:
                if action.operation == "select_column":
                    col_ok, op_error = select_column_working(self._working, action)
                    base_reward, base_reason = 0.0, "awaiting_progress_credit"
                elif action.operation == "filter":
                    new_rows, op_error = filter_working_set(self._working, action)
                    if op_error is None:
                        self._working = new_rows
                    base_reward, base_reason = 0.0, "awaiting_progress_credit"
                elif action.operation == "sum":
                    computed, op_error = sum_working_set(self._working, action)
                    base_reward, base_reason = 0.0, "awaiting_progress_credit"
                elif action.operation == "max":
                    computed, op_error = max_working_set(self._working, action)
                    base_reward, base_reason = 0.0, "awaiting_progress_credit"
                else:
                    op_error = "unknown_operation"
                    base_reward, base_reason = 0.0, "unknown_operation"

                if penalty_kind == "none" and classify_invalid_for_state(
                    self._working, action, op_error
                ):
                    penalty_kind = "invalid"
                    penalty_detail = op_error or "invalid_for_state"

                if penalty_kind == "none" and classify_useless(
                    action,
                    working_before,
                    self._working,
                    op_error,
                    prev_selected,
                ):
                    penalty_kind = "useless"
                    penalty_detail = "no_effect"

        except Exception as exc:  # noqa: BLE001
            self._working = working_before
            self._selected_column = prev_selected
            op_error = f"exception:{type(exc).__name__}"
            exception_trace = traceback.format_exc()
            penalty_kind = "invalid"
            penalty_detail = "exception"
            base_reward, base_reason = 0.0, "exception"
            computed = None

        if penalty_kind == "invalid" and exception_trace is None:
            self._working = working_before
            self._selected_column = prev_selected

        if penalty_kind != "invalid" and exception_trace is None:
            if action.operation == "select_column" and op_error is None and col_ok:
                self._selected_column = col_ok
            elif (
                action.operation in ("filter", "sum", "max")
                and op_error is None
                and action.column
            ):
                self._selected_column = action.column.strip()

        if penalty_kind == "none" and op_error is None:
            if action.operation == "select_column" and col_ok:
                best_col_score = _best_column_score(self._working, self.correct_answer)
                selected_score = _column_score(self._working, col_ok, self.correct_answer)
                ratio = (
                    min(selected_score / best_col_score, 1.0)
                    if best_col_score > 0.0
                    else 0.0
                )
                self._select_credit = max(self._select_credit, 0.2 * ratio)
            elif action.operation == "filter":
                filter_target = 0.3 * _best_column_score(self._working, self.correct_answer)
                self._filter_credit = max(self._filter_credit, min(filter_target, 0.3))
            elif action.operation in ("sum", "max") and computed is not None:
                final_score, _ = shaped_reward(computed, self.correct_answer)
                exact = answer_matches(computed, self.correct_answer)
                calibrated_final, _ = confidence_adjusted_reward(
                    0.5 * final_score,
                    confidence=action.confidence,
                    is_final_answer_step=True,
                    is_exact=exact,
                )
                self._final_credit = max(self._final_credit, max(0.0, calibrated_final))

            progress_total = (
                self._select_credit + self._filter_credit + self._final_credit
            )
            base_reward = max(0.0, progress_total - prev_progress_total)
            base_reason = "trajectory_progress"

        reward, reward_reason = apply_penalty(
            base_reward, base_reason, penalty_kind, penalty_detail
        )
        confidence_reason = "none"
        if (
            penalty_kind == "none"
            and op_error is None
            and action.operation in ("sum", "max")
            and computed is not None
        ):
            # Apply confidence calibration at the step-reward level too, so
            # overconfident wrong final answers are directly penalized.
            exact = answer_matches(computed, self.correct_answer)
            reward, confidence_reason = confidence_adjusted_reward(
                reward,
                confidence=action.confidence,
                is_final_answer_step=True,
                is_exact=exact,
            )

        solved = (
            penalty_kind == "none"
            and op_error is None
            and computed is not None
            and answer_matches(computed, self.correct_answer)
            and action.operation in ("sum", "max")
        )
        done = solved or self._episode_step >= self.max_steps
        if done:
            self._terminated = True

        record: Dict[str, Any] = {
            "step": self._episode_step,
            "operation": action.operation,
            "column": action.column,
            "value": action.value,
            "confidence": action.confidence,
            "computed": computed,
            "rows_after": len(self._working),
            "error": op_error,
            "penalty_kind": penalty_kind,
            "reward_before_penalty": base_reward,
        }
        self._intermediate_results.append(record)

        self._observation = self._build_observation()

        info: Dict[str, Any] = {
            "action": action.model_dump(),
            "steps_taken": self._episode_step,
            "max_steps": self.max_steps,
            "computed_result": computed,
            "correct_answer": self.correct_answer,
            "operation_error": op_error,
            "reward_reason": reward_reason,
            "solved": solved,
            "working_row_count": len(self._working),
            "validation": penalty_kind,
            "penalty_applied": penalty_kind != "none",
            "reward_before_penalty": base_reward,
            "confidence": action.confidence,
            "confidence_reason": confidence_reason,
            "trajectory_progress": {
                "select_credit": self._select_credit,
                "filter_credit": self._filter_credit,
                "final_credit": self._final_credit,
                "total": self._select_credit + self._filter_credit + self._final_credit,
            },
        }
        if exception_trace:
            info["exception_trace"] = exception_trace
        return self._observation.model_copy(), reward, done, info

    def state(self) -> Optional[Observation]:
        """Return the latest observation, or None if ``reset`` has not been called."""
        if self._observation is None:
            return None
        return self._observation.model_copy()

    def _build_observation(self) -> Observation:
        return Observation(
            data=[copy.deepcopy(r) for r in self._working],
            question=self.question,
            step=self._episode_step,
            max_steps=self.max_steps,
            selected_column=self._selected_column,
            intermediate_results=list(self._intermediate_results),
        )


def _column_score(rows: List[Dict[str, Any]], column: str, expected: Answer) -> float:
    """How plausible a column is for reaching the final answer, in [0, 1]."""
    values: List[float] = []
    for row in rows:
        if column not in row:
            continue
        v = parse_number(row.get(column))
        if v is not None:
            values.append(v)
    if not values:
        return 0.0

    sum_reward, _ = shaped_reward(sum(values), expected)
    max_reward, _ = shaped_reward(max(values), expected)
    return max(sum_reward, max_reward)


def _best_column_score(rows: List[Dict[str, Any]], expected: Answer) -> float:
    """Best achievable column-level score over current rows, in [0, 1]."""
    columns = {k for row in rows for k in row.keys()}
    if not columns:
        return 0.0
    best = 0.0
    for c in columns:
        best = max(best, _column_score(rows, c, expected))
    return best
