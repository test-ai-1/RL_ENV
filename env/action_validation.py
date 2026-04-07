"""Action validation: classify mistakes and apply fixed penalties."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from .actions import Action as DataAction

PenaltyKind = Literal["none", "invalid", "useless"]

PENALTY_INVALID = -0.3
PENALTY_USELESS = -0.1


def column_in_schema(rows: List[Dict[str, Any]], column: str) -> bool:
    """True if ``column`` appears as a key in at least one row."""
    if not column:
        return False
    return any(column in r for r in rows)


def classify_missing_column(
    working_rows: List[Dict[str, Any]], action: DataAction
) -> bool:
    """Invalid: required column is absent from the current working table schema."""
    if not action.column:
        return False
    col = action.column.strip()
    if not col:
        return True
    # Empty table: no column can exist
    if not working_rows:
        return True
    return not column_in_schema(working_rows, col)


def classify_invalid_for_state(
    working_rows: List[Dict[str, Any]],
    action: DataAction,
    op_error: Optional[str],
) -> bool:
    """
    Invalid operation for current state (beyond missing column, handled separately).

    Uses table-op error codes and row counts.
    """
    if op_error:
        if op_error in {
            "empty_table",
            "column_missing",
            "column_not_numeric",
            "empty_column",
            "filter_requires_value",
            "filter_empty_value",
            "expected_filter_operation",
            "expected_sum_operation",
            "expected_max_operation",
            "expected_select_column_operation",
        }:
            return True
        if op_error.startswith("expected_"):
            return True
    if action.operation in ("sum", "max") and not working_rows:
        return True
    return False


def classify_useless(
    action: DataAction,
    working_before: List[Dict[str, Any]],
    working_after: List[Dict[str, Any]],
    op_error: Optional[str],
    prev_selected_column: Optional[str],
) -> bool:
    """
    Action had no meaningful effect (small penalty only if not already invalid).

    - filter: row count unchanged (predicate matched all remaining rows)
    - select_column: same column as already selected
    """
    if op_error is not None:
        return False

    if action.operation == "filter":
        return len(working_before) == len(working_after)

    if action.operation == "select_column":
        if not action.column:
            return False
        return action.column.strip() == (prev_selected_column or "").strip() and bool(
            prev_selected_column
        )

    return False


def apply_penalty(
    base_reward: float,
    base_reason: str,
    penalty: PenaltyKind,
    detail: str,
) -> Tuple[float, str]:
    """Replace base reward when a penalty tier applies."""
    if penalty == "invalid":
        return PENALTY_INVALID, f"penalty_invalid:{detail}"
    if penalty == "useless":
        return PENALTY_USELESS, f"penalty_useless:{detail}"
    return base_reward, base_reason
