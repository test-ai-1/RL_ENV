"""Reward shaping: compare computed scalars to gold answers in [0, 1]."""

from __future__ import annotations

import math
from typing import Optional, Union

Answer = Union[int, float, str]

# Exact match: relative or absolute tolerance (handles expected == 0).
_EXACT_REL_TOL = 1e-9
_EXACT_ABS_TOL = 1e-9
# Piecewise relative-error bands (for non-zero expected).
_BAND_EXCELLENT = 0.01
_BAND_GOOD = 0.05
_BAND_FAIR = 0.15
_BAND_WEAK = 0.40


def answer_matches(computed: Optional[float], expected: Answer) -> bool:
    """True when ``computed`` matches gold with exact-tier tolerance."""
    if computed is None:
        return False
    _r, reason = shaped_reward(computed, expected)
    return reason == "exact"


def coerce_expected(expected: Answer) -> Optional[float]:
    """Parse gold answer to float; invalid/missing values → None."""
    return _to_float(expected)


def _to_float(value: object) -> Optional[float]:
    """
    Robust numeric preprocessing for messy table values.

    Handles:
    - "1k", "2.5m"
    - "$1,200"
    - None / NULL-like placeholders
    """
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return float(value)

    if not isinstance(value, str):
        return None

    s = value.strip()
    if not s:
        return None
    if s.upper() in {"NULL", "NONE", "N/A", "NA", "#N/A", "NAN", "-"}:
        return None

    s = s.replace(",", "")
    s = s.replace("$", "").strip()
    if not s:
        return None

    multiplier = 1.0
    suffix = s[-1].lower()
    if suffix == "k":
        multiplier = 1_000.0
        s = s[:-1].strip()
    elif suffix == "m":
        multiplier = 1_000_000.0
        s = s[:-1].strip()

    if not s:
        return None

    try:
        return float(s) * multiplier
    except ValueError:
        return None


def shaped_reward(
    computed: Optional[Union[int, float, str]], expected: Answer
) -> tuple[float, str]:
    """
    Map ``computed`` vs numeric ``expected`` to a reward in [0, 1] with interpretable tiers.

    Returns:
        (reward, short_reason_code)

    Tiers (for non-zero expected, by relative error):
        - exact → 1.0
        - excellent (≤1%) → ~0.9–1.0
        - good (≤5%) → ~0.65–0.9
        - fair (≤15%) → ~0.35–0.65
        - weak (≤40%) → ~0.1–0.35
        - else → smooth tail toward 0

    For ``expected == 0``, uses absolute error with a dedicated band.
    """
    exp = coerce_expected(expected)
    if exp is None:
        return 0.0, "invalid_expected_answer"

    comp = _to_float(computed)
    if comp is None:
        return 0.0, "invalid_action"
    err = abs(comp - exp)

    if exp == 0.0:
        return _reward_expected_zero(err)

    scale = abs(exp)
    rel = err / scale

    if err <= _EXACT_REL_TOL * max(1.0, scale):
        return 1.0, "exact"

    if rel <= _BAND_EXCELLENT:
        t = rel / _BAND_EXCELLENT
        r = 0.90 + 0.10 * (1.0 - t)
        return float(r), "excellent"

    if rel <= _BAND_GOOD:
        t = (rel - _BAND_EXCELLENT) / (_BAND_GOOD - _BAND_EXCELLENT)
        r = 0.65 + 0.25 * (1.0 - t)
        return float(r), "good"

    if rel <= _BAND_FAIR:
        t = (rel - _BAND_GOOD) / (_BAND_FAIR - _BAND_GOOD)
        r = 0.35 + 0.30 * (1.0 - t)
        return float(r), "fair"

    if rel <= _BAND_WEAK:
        t = (rel - _BAND_FAIR) / (_BAND_WEAK - _BAND_FAIR)
        r = 0.10 + 0.25 * (1.0 - t)
        return float(r), "weak"

    # Long tail: still give tiny credit if not absurdly far (stays in [0, 0.1]).
    tail = max(0.0, 0.10 * (1.0 - min((rel - _BAND_WEAK) / (1.0 - _BAND_WEAK), 1.0)))
    return float(tail), "distant"


def _reward_expected_zero(err: float) -> tuple[float, str]:
    if err <= _EXACT_ABS_TOL:
        return 1.0, "exact"
    if err <= 1e-6:
        return 0.85 + 0.15 * (1.0 - min(err / 1e-6, 1.0)), "excellent"
    if err <= 1e-3:
        t = (err - 1e-6) / (1e-3 - 1e-6)
        return float(0.45 + 0.40 * (1.0 - t)), "good"
    if err <= 0.05:
        t = (err - 1e-3) / (0.05 - 1e-3)
        return float(0.15 + 0.30 * (1.0 - t)), "fair"
    tail = max(0.0, 0.15 * math.exp(-err / 0.5))
    return float(tail), "distant"


def confidence_adjusted_reward(
    base_reward: float,
    *,
    confidence: float,
    is_final_answer_step: bool,
    is_exact: bool,
) -> tuple[float, str]:
    """
    Calibrate final-answer reward by confidence.

    - Correct answer: slight boost with higher confidence.
    - Wrong answer: high confidence incurs stronger penalty.
    """
    c = min(max(confidence, 0.0), 1.0)
    if not is_final_answer_step:
        return float(base_reward), "no_confidence_adjustment"

    if is_exact:
        boosted = min(1.0, base_reward * (0.8 + 0.2 * c))
        return float(boosted), "confidence_boost_correct"

    # Penalize overconfidence when wrong; low confidence keeps a softer penalty.
    penalty = 0.5 * c * c
    adjusted = max(-0.5, base_reward - penalty)
    return float(adjusted), "confidence_penalty_wrong"
