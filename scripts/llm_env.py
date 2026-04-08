"""Shared env normalization for OpenAI client (HF_TOKEN / API_BASE_URL / MODEL_NAME)."""

from __future__ import annotations

import os


def ensure_llm_env_defaults() -> None:
    if os.environ.get("HF_TOKEN", "").strip() and not os.environ.get(
        "OPENAI_API_KEY", ""
    ).strip():
        os.environ["OPENAI_API_KEY"] = os.environ["HF_TOKEN"].strip()
    if not os.environ.get("API_BASE_URL", "").strip():
        os.environ["API_BASE_URL"] = "https://api.openai.com/v1"
    if os.environ.get("MODEL_NAME", "").strip() and not os.environ.get(
        "OPENAI_MODEL", ""
    ).strip():
        os.environ["OPENAI_MODEL"] = os.environ["MODEL_NAME"].strip()


def has_api_key() -> bool:
    return bool(
        os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )
