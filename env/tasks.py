"""Curated adversarial analyst tasks (easy → hard).
The environment is **multi-step**: ``filter`` narrows the working table; ``sum`` aggregates
the current table. Gold numbers are reachable within a few steps (often: filter → sum).

Numeric parsing (see ``table_ops.parse_number``): ``$``, commas, ``k``/``m`` suffixes,
``NULL`` / ``N/A``-like strings, and ``None`` are handled; inconsistent row shapes are allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Union


Answer = Union[int, float, str]


@dataclass(frozen=True)
class DataAnalystTask:
    """One benchmark: rows, natural-language question, and scalar gold answer."""

    difficulty: Literal["easy", "medium", "hard"]
    dataset: list[dict[str, Any]]
    question: str
    correct_answer: Answer


# --- Easy: one metric column, intentionally dirty literals + junk rows.
# Sum qty_sold: 12 + 8 + 5 = 25 (None / "NULL" skipped; "units_order" is a distractor).
TASK_EASY = DataAnalystTask(
    difficulty="easy",
    dataset=[
        {"sku": "W-01", "qty_sold": "12"},
        {"sku": "W-02", "qty_sold": "$8"},
        {"sku": "G-10", "qty_sold": "5.00"},
        {"sku": "X-99", "qty_sold": None},
        {"sku": "Y-00", "qty_sold": "NULL"},
        {"sku": "Z-11", "units_order": "999"},  # wrong field name on purpose
    ],
    question=(
        "CRM export with mixed types. What is the sum of `qty_sold` across rows where that "
        "field carries a usable number? Treat currency-looking strings as numbers, skip NULL-like "
        "sentinels and blank cells, and do **not** use other columns (e.g. `units_order`) as a substitute."
    ),
    correct_answer=25,
)

# --- Medium: filter by region + messy money strings + extra West row.
# West totals: $4,200 + 1.8k + $100 = 6,100 (case-insensitive region match).
TASK_MEDIUM = DataAnalystTask(
    difficulty="medium",
    dataset=[
        {"rep": "A. Lee", "region": "West", "deal_value": "$4,200"},
        {"rep": "A. Lee", "region": "west", "deal_value": "1.8k"},
        {"rep": "M. Rao", "region": "East", "deal_value": "3,100.00"},
        {"rep": "M. Rao", "region": "East", "deal_value": None},
        {"rep": "K. Ng", "region": "East", "deal_value": "NULL"},
        {"rep": "Adj", "region": "West", "deal_value": "$100"},
    ],
    question=(
        "Pipeline spreadsheet with noisy cells. Restrict to rows whose `region` is West "
        "(match case-insensitively). What is the total `deal_value` for that subset? "
        "Parse currency and `k` suffixes; ignore NULL-like values inside matching rows."
    ),
    correct_answer=6100,
)

# --- Hard: compare totals across groups, then return the larger group's total.
# north total: 2.4k + $1,100 + NULL = 3,500
# south total: 900 + 800 + $50 = 1,750
# answer: 3,500
TASK_HARD = DataAnalystTask(
    difficulty="hard",
    dataset=[
        {"region": "north", "rep": "A", "sales": "2.4k"},
        {"region": "north", "rep": "B", "sales": "$1,100"},
        {"region": "north", "rep": "C", "sales": "NULL"},
        {"region": "south", "rep": "D", "sales": "900"},
        {"region": "south", "rep": "E", "sales": 800},
        {"region": "south", "rep": "F", "sales": "$50"},
    ],
    question=(
        "RegionalSales sheet has two groups: `north` and `south`. Compare **total `sales` by region** "
        "(parse `$` and `k`, skip NULL-like cells). Which region has the larger total? "
        "Return only that region's total as a single number."
    ),
    correct_answer=3500,
)

ALL_TASKS: tuple[DataAnalystTask, ...] = (TASK_EASY, TASK_MEDIUM, TASK_HARD)
