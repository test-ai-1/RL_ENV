"""Observation schema for the data environment."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class Observation(BaseModel):
    """Per-step observation: working table, task text, and interaction progress."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    data: list[dict[str, Any]] = Field(description="Current working rows (after any filters)")
    question: str = Field(min_length=1, description="Natural-language task question")
    step: int = Field(ge=0, description="Number of environment steps completed so far")
    max_steps: int = Field(ge=1, description="Episode horizon")
    selected_column: Optional[str] = Field(
        default=None, description="Column from the most recent action"
    )
    intermediate_results: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Append-only log of prior step outcomes (ops, scalars, row counts)",
    )
