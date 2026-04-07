"""Action schema for multi-step dataset operations."""

from __future__ import annotations

from typing import Literal, Optional, assert_never

from pydantic import BaseModel, ConfigDict, Field, model_validator

OperationName = Literal["select_column", "filter", "sum", "max"]


class Action(BaseModel):
    """
    One environment step: choose an operation and optional column / filter value.

    Constraints depend on ``operation`` (enforced by :meth:`validate_operation_fields`).
    """

    model_config = ConfigDict(extra="forbid", frozen=False)

    operation: OperationName = Field(description="Which operation to run")
    column: Optional[str] = Field(
        default=None,
        description="Column name; required for select_column, filter, sum, and max",
    )
    value: Optional[str] = Field(
        default=None,
        description="Filter predicate; required only when operation is filter",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Agent confidence in this action/result, in [0, 1]",
    )

    @model_validator(mode="after")
    def validate_operation_fields(self) -> Action:
        col = self.column.strip() if self.column is not None else None
        val = self.value.strip() if self.value is not None else None

        if self.operation == "filter":
            if not col:
                raise ValueError("column is required when operation is 'filter'")
            if val is None or val == "":
                raise ValueError("value is required when operation is 'filter'")
            self.column = col
            self.value = val
            return self

        if self.operation in ("select_column", "sum", "max"):
            if not col:
                raise ValueError(
                    f"column is required when operation is {self.operation!r}"
                )
            self.column = col
            self.value = None
            return self

        assert_never(self.operation)
