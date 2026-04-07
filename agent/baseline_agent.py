"""Baseline agent: maps observations to actions via the Groq Chat Completions API (OpenAI-compatible)."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from openai import APIError, OpenAI, RateLimitError

from env.actions import Action
from env.observations import Observation

# Groq exposes an OpenAI-compatible API; see https://console.groq.com/docs/openai
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Override with env GROQ_MODEL if this id is unavailable in your account.
_DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

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
    Calls Groq with the observation and returns a validated :class:`Action`.

    Requires ``GROQ_API_KEY``. Optional ``GROQ_MODEL`` (defaults to
    ``openai/gpt-oss-120b`` or the constructor ``model`` argument).
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        client: Optional[OpenAI] = None,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> None:
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        groq_key = os.environ.get("GROQ_API_KEY", "").strip()
        if model:
            self._model = model
        elif os.environ.get("GROQ_MODEL", "").strip():
            self._model = os.environ["GROQ_MODEL"].strip()
        elif os.environ.get("OPENAI_MODEL", "").strip():
            self._model = os.environ["OPENAI_MODEL"].strip()
        else:
            self._model = _DEFAULT_OPENAI_MODEL if openai_key else _DEFAULT_GROQ_MODEL
        self._temperature = temperature
        self._seed = seed

        if client is not None:
            self._client = client
            return

        if not openai_key and not groq_key:
            raise ValueError(
                "Neither OPENAI_API_KEY nor GROQ_API_KEY is set. Add one to your "
                "environment or .env file."
            )

        # If both keys are available, default to OpenAI unless model looks Groq-specific.
        use_groq = False
        if groq_key and not openai_key:
            use_groq = True
        elif groq_key and openai_key:
            m = self._model.lower()
            use_groq = m.startswith("openai/") or "llama" in m or "mixtral" in m

        if use_groq:
            self._client = OpenAI(base_url=_GROQ_BASE_URL, api_key=groq_key)
        else:
            self._client = OpenAI(api_key=openai_key)

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
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                seed=self._seed,
                response_format={"type": "json_object"},
                messages=messages,
            )
        except RateLimitError as exc:
            raise RuntimeError(
                "Groq returned 429 (rate limit). Retry later or check limits at "
                "https://console.groq.com/"
            ) from exc
        except APIError as exc:
            # Some models may not support JSON mode; retry without it.
            if getattr(exc, "status_code", None) == 400:
                completion = self._client.chat.completions.create(
                    model=self._model,
                    temperature=self._temperature,
                    seed=self._seed,
                    messages=messages,
                )
            else:
                raise

        raw = completion.choices[0].message.content
        if not raw:
            raise RuntimeError("Groq returned empty message content")

        data = _extract_json_object(raw)
        return Action.model_validate(data)
