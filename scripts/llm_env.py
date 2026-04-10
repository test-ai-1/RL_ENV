"""Shared env normalization for OpenAI client (API_KEY + API_BASE_URL for LiteLLM proxy)."""

from __future__ import annotations

import os


def ensure_llm_env_defaults() -> None:
    # Platform proxy: sync key into names the client accepts
    if os.environ.get("API_KEY", "").strip():
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            os.environ["OPENAI_API_KEY"] = os.environ["API_KEY"].strip()
    elif os.environ.get("HF_TOKEN", "").strip() and not os.environ.get(
        "OPENAI_API_KEY", ""
    ).strip():
        os.environ["OPENAI_API_KEY"] = os.environ["HF_TOKEN"].strip()
    # Never send traffic to api.openai.com when the platform set API_KEY (proxy required)
    if not os.environ.get("API_BASE_URL", "").strip():
        if not os.environ.get("API_KEY", "").strip():
            os.environ["API_BASE_URL"] = "https://api.openai.com/v1"
    if os.environ.get("MODEL_NAME", "").strip() and not os.environ.get(
        "OPENAI_MODEL", ""
    ).strip():
        os.environ["OPENAI_MODEL"] = os.environ["MODEL_NAME"].strip()


def has_api_key() -> bool:
    return bool(
        os.environ.get("API_KEY", "").strip()
        or os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )
