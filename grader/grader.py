"""Compare model outputs to ground truth with discrete scores."""

from __future__ import annotations

import math
from typing import Any, Optional, Union

from env.table_ops import parse_number

Prediction = Union[int, float, str, None]
GroundTruth = Union[int, float, str]

# Relative tolerance for treating two floats as identical.
_EXACT_RTOL = 1e-9
# Max relative error still counted as a partial match (exclusive of exact).
_PARTIAL_MAX_REL = 0.10

# Validator requires scores strictly in (0, 1), not 0.0 nor 1.0.
_SCORE_WRONG = 0.01
_SCORE_PARTIAL = 0.50
_SCORE_EXACT = 0.99


def _to_float(value: Any) -> Optional[float]:
    """Delegate to env parsing so grader matches messy task literals."""
    return parse_number(value)


def grade(prediction: Prediction, ground_truth: GroundTruth) -> float:
    """
    Score a prediction against the reference answer.

    Returns:
        Values strictly in ``(0, 1)`` (never ``0.0`` nor ``1.0``), per submission validators:
        ``_SCORE_EXACT`` — exact match (numeric within tight tolerance; strings equal after strip).
        ``_SCORE_PARTIAL`` — partial match (close numeric; or strings equal ignoring case).
        ``_SCORE_WRONG`` — no match or empty prediction.
    """
    if prediction is None:
        return _SCORE_WRONG

    pred_f = _to_float(prediction)
    truth_f = _to_float(ground_truth)

    if pred_f is not None and truth_f is not None:
        return _grade_numeric(pred_f, truth_f)

    pred_s = str(prediction).strip()
    truth_s = str(ground_truth).strip()
    if pred_s == truth_s:
        return _SCORE_EXACT
    if pred_s.casefold() == truth_s.casefold():
        return _SCORE_PARTIAL
    return _SCORE_WRONG


def _grade_numeric(pred: float, truth: float) -> float:
    scale = max(abs(truth), 1e-12)
    err = abs(pred - truth)
    if err <= _EXACT_RTOL * max(1.0, abs(truth)):
        return _SCORE_EXACT
    if truth == 0.0:
        if err <= 1e-9:
            return _SCORE_EXACT
        if err <= 1e-3:
            return _SCORE_PARTIAL
        return _SCORE_WRONG
    rel_err = err / scale
    if rel_err <= _PARTIAL_MAX_REL:
        return _SCORE_PARTIAL
    return _SCORE_WRONG
