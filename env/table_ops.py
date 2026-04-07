"""Tabular operations used by :class:`DataEnv` (sum / filter + aggregate)."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from .actions import Action as DataAction


def filter_working_set(
    rows: List[Dict[str, Any]], action: DataAction
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Narrow the working table (multi-step ``filter`` only; no aggregation).

    Returns:
        (new_rows, None) or ([], error_code).
    """
    if action.operation != "filter":
        return list(rows), "expected_filter_operation"
    if not action.column:
        return list(rows), "empty_column"
    col = action.column.strip()
    if not col:
        return list(rows), "empty_column"
    raw = action.value
    if raw is None:
        return list(rows), "filter_requires_value"
    value = raw.strip()
    if not value:
        return list(rows), "filter_empty_value"
    return _filter_rows(rows, col, value), None


def sum_working_set(
    rows: List[Dict[str, Any]], action: DataAction
) -> Tuple[Optional[float], Optional[str]]:
    """Sum one numeric column over the current working table (multi-step ``sum``)."""
    if action.operation != "sum":
        return None, "expected_sum_operation"
    if not action.column:
        return None, "empty_column"
    col = action.column.strip()
    if not rows:
        return None, "empty_table"
    return _sum_column(rows, col)


def max_working_set(
    rows: List[Dict[str, Any]], action: DataAction
) -> Tuple[Optional[float], Optional[str]]:
    """Maximum of one numeric column over the current working table."""
    if action.operation != "max":
        return None, "expected_max_operation"
    if not action.column:
        return None, "empty_column"
    if not rows:
        return None, "empty_table"
    return _max_column(rows, action.column.strip())


def select_column_working(
    rows: List[Dict[str, Any]], action: DataAction
) -> Tuple[Optional[str], Optional[str]]:
    """
    Validate that ``column`` exists in at least one row (or allow empty table).

    Does not mutate rows; returns the normalized column name on success.
    """
    if action.operation != "select_column":
        return None, "expected_select_column_operation"
    if not action.column:
        return None, "empty_column"
    col = action.column.strip()
    if rows and not any(col in r for r in rows):
        return None, "column_missing"
    return col, None


def _max_column(
    rows: List[Dict[str, Any]], column: str
) -> Tuple[Optional[float], Optional[str]]:
    best: Optional[float] = None
    saw_key = False
    any_numeric = False
    for row in rows:
        if column not in row:
            continue
        saw_key = True
        v = _to_float(row.get(column))
        if v is not None:
            any_numeric = True
            best = v if best is None else max(best, v)
    if not saw_key:
        return None, "column_missing"
    if not any_numeric:
        return None, "column_not_numeric"
    return best, None


def apply_action(
    data: List[Dict[str, Any]], action: DataAction
) -> Tuple[Optional[float], Optional[str]]:
    """
    Legacy single-shot: ``sum`` over ``data``, or ``filter`` then sum inferred metric.

    Prefer :func:`filter_working_set` + :func:`sum_working_set` for multi-step control.
    """
    if action.operation not in ("sum", "filter"):
        return None, "legacy_only_sum_or_filter"

    if not action.column:
        return None, "empty_column"
    col = action.column.strip()
    if not col:
        return None, "empty_column"

    if action.operation == "sum":
        if not data:
            return 0.0, None
        return _sum_column(data, col)

    if action.operation == "filter":
        raw = action.value
        if raw is None:
            return None, "filter_requires_value"
        value = raw.strip()
        if not value:
            return None, "filter_empty_value"
        rows = _filter_rows(data, col, value)
        metric = _infer_metric_column(rows, col, data)
        if metric is None:
            return None, "no_numeric_metric_column"
        if not rows:
            return 0.0, None
        return _sum_column(rows, metric)

    return None, "unknown_operation"


def _sum_column(rows: List[Dict[str, Any]], column: str) -> Tuple[Optional[float], Optional[str]]:
    if not rows:
        return 0.0, None

    total = 0.0
    any_numeric = False
    saw_key = False
    for row in rows:
        if column not in row:
            continue
        saw_key = True
        v = _to_float(row.get(column))
        if v is not None:
            total += v
            any_numeric = True

    if not saw_key:
        return None, "column_missing"
    if not any_numeric:
        return None, "column_not_numeric"
    return total, None


def _to_float(x: Any) -> Optional[float]:
    """Parse ints/floats plus messy analyst exports: currency, thousands suffixes, NULL text."""
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        upper = s.upper()
        if upper in {"NULL", "NONE", "N/A", "NA", "#N/A", "NAN", "-"}:
            return None

        s = s.replace(",", "")
        s = re.sub(r"^\$+", "", s).strip()
        if not s:
            return None

        mult = 1.0
        if len(s) > 1 and s[-1].lower() == "k":
            mult = 1000.0
            s = s[:-1].strip()
        elif len(s) > 1 and s[-1].lower() == "m":
            mult = 1_000_000.0
            s = s[:-1].strip()
        if not s:
            return None
        try:
            return float(s) * mult
        except ValueError:
            return None
    return None


def parse_number(x: Any) -> Optional[float]:
    """Public alias for :func:`_to_float` (messy currency / k / NULL handling)."""
    return _to_float(x)


def _row_matches(row: Dict[str, Any], column: str, value: str) -> bool:
    if column not in row:
        return False
    cell = row[column]
    target = value.strip()
    if isinstance(cell, bool):
        return str(cell).lower() == target.lower()
    if isinstance(cell, (int, float)):
        tf = _to_float(target)
        if tf is not None:
            return abs(float(cell) - tf) < 1e-9
        return str(cell) == target
    return str(cell).strip().lower() == target.lower()


def _filter_rows(
    rows: List[Dict[str, Any]], column: str, value: str
) -> List[Dict[str, Any]]:
    return [r for r in rows if _row_matches(r, column, value)]


def _numeric_columns_for(rows: List[Dict[str, Any]]) -> List[str]:
    if not rows:
        return []
    keys: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    out: List[str] = []
    for k in keys:
        if any(_to_float(r.get(k)) is not None for r in rows):
            out.append(k)
    return out


def _infer_metric_column(
    filtered_rows: List[Dict[str, Any]],
    filter_column: str,
    schema_rows: List[Dict[str, Any]],
) -> Optional[str]:
    """
    Choose which numeric column to sum after a filter.

    If the filter leaves no rows, infer numeric columns from ``schema_rows`` (full table).
    """
    base = filtered_rows if filtered_rows else schema_rows
    if not base:
        return None

    numeric_keys = _numeric_columns_for(base)
    if not numeric_keys:
        return None

    if filter_column in numeric_keys and len(numeric_keys) == 1:
        return filter_column
    for k in numeric_keys:
        if k != filter_column:
            return k
    return numeric_keys[0]
