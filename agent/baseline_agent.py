"""Baseline agent: maps observations to actions via the OpenAI Chat Completions API."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from openai import APIError, OpenAI, RateLimitError

from env.actions import Action
from env.observations import Observation

# Official OpenEnv / HF Inference sample default (overridden by MODEL_NAME)
_DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
_DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"

_SYSTEM_PROMPT = """You are a multi-step data analyst agent. Decide ONE next action per turn.

You receive:
- `data`: current working rows after prior filters
- `question`: task to solve
- `step`, `max_steps`: current step and budget
- `intermediate_results`: prior actions and outcomes

Always reason as a sequence of steps:
1) First choose the target metric column with `select_column`.
2) Then apply `filter` only if needed to isolate the correct subset.
3) Then compute the final result with `sum` or `max`.

Do not jump to aggregation before column selection. Prefer minimal, purposeful actions.
Use `intermediate_results` to avoid repeating useless actions.

Output format (MANDATORY):
- Return JSON only, no markdown, no prose.
- Return exactly one object with exactly these keys:
  - "operation"
  - "column"
  - "value"
  - "confidence"

Schema:
- "operation": one of "select_column", "filter", "sum", "max"
- "column": non-empty string; use an exact key from current `data` whenever possible
- "value": string for `filter`; for non-filter operations use null
- "confidence": number in [0, 1] indicating certainty of this next action

Operation behavior:
- "select_column": validates the metric column; does not change rows
- "filter": keeps rows where `column` equals `value` (case-insensitive text match)
- "sum": sums numeric values in `column` over CURRENT rows
- "max": returns maximum numeric value in `column` over CURRENT rows

Data can be messy (`$`, commas, `k`/`m`, NULL-like values); environment parsing handles this.
Your job is to choose the correct NEXT step in the sequence."""


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse JSON from model output; strip optional ```json fences."""
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


class BaselineAgent:
    """
    Calls the LLM via the OpenAI Python client (OpenAI-compatible servers).

    **Official sample (HF Inference Router):**

    - ``HF_TOKEN`` or ``API_KEY`` — required
    - ``API_BASE_URL`` — optional; defaults to ``https://router.huggingface.co/v1``
    - ``MODEL_NAME`` — optional; defaults to ``Qwen/Qwen2.5-72B-Instruct``

    Other providers: set ``API_BASE_URL`` and ``MODEL_NAME`` accordingly.
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        client: Optional[OpenAI] = None,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> None:
        hf_or_api = (
            os.environ.get("HF_TOKEN", "").strip()
            or os.environ.get("API_KEY", "").strip()
        )
        openai_only = os.environ.get("OPENAI_API_KEY", "").strip()
        api_key = hf_or_api or openai_only

        if os.environ.get("API_BASE_URL", "").strip():
            api_base = os.environ["API_BASE_URL"].strip()
        elif hf_or_api:
            api_base = _DEFAULT_API_BASE_URL
        elif openai_only:
            api_base = "https://api.openai.com/v1"
        else:
            api_base = _DEFAULT_API_BASE_URL

        if model:
            self._model = model
        elif os.environ.get("MODEL_NAME", "").strip():
            self._model = os.environ["MODEL_NAME"].strip()
        elif os.environ.get("OPENAI_MODEL", "").strip():
            self._model = os.environ["OPENAI_MODEL"].strip()
        else:
            self._model = _DEFAULT_MODEL_NAME
        self._temperature = temperature
        self._seed = seed

        if client is not None:
            self._client = client
            return

        if not api_key:
            raise ValueError(
                "Set HF_TOKEN or API_KEY (Hugging Face token), or OPENAI_API_KEY for other providers."
            )

        self._client = OpenAI(
            api_key=api_key,
            base_url=api_base.rstrip("/"),
        )

    def act(self, observation: Observation) -> Action:
        """Return an action for the given observation."""
        user_payload = {
            "question": observation.question,
            "data": observation.data,
            "step": observation.step,
            "max_steps": observation.max_steps,
            "intermediate_results": observation.intermediate_results,
        }
        user_message = json.dumps(user_payload, ensure_ascii=False, indent=2)

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        def _create(**extra: Any) -> Any:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "temperature": self._temperature,
                "messages": messages,
            }
            kwargs.update(extra)
            return self._client.chat.completions.create(**kwargs)

        try:
            completion = _create(
                seed=self._seed,
                response_format={"type": "json_object"},
            )
        except RateLimitError as exc:
            raise RuntimeError(
                "OpenAI returned 429 (rate limit). Retry later or check your account limits."
            ) from exc
        except APIError as exc:
            # Some models/endpoints reject json_mode and/or seed; retry with minimal kwargs.
            if getattr(exc, "status_code", None) == 400:
                try:
                    completion = _create(
                        seed=self._seed,
                    )
                except APIError:
                    completion = _create()
            else:
                raise

        raw = completion.choices[0].message.content
        if not raw:
            raise RuntimeError("OpenAI returned empty message content")

        data = _extract_json_object(raw)
        return Action.model_validate(data)
