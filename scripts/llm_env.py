"""Shared env normalization for OpenAI-compatible client (HF Inference Router defaults)."""

from __future__ import annotations

import os

_DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
_DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"


def ensure_llm_env_defaults() -> None:
    if os.environ.get("API_KEY", "").strip():
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            os.environ["OPENAI_API_KEY"] = os.environ["API_KEY"].strip()
    elif os.environ.get("HF_TOKEN", "").strip() and not os.environ.get(
        "OPENAI_API_KEY", ""
    ).strip():
        os.environ["OPENAI_API_KEY"] = os.environ["HF_TOKEN"].strip()
    if not os.environ.get("API_BASE_URL", "").strip():
        if (
            os.environ.get("HF_TOKEN", "").strip()
            or os.environ.get("API_KEY", "").strip()
        ):
            os.environ["API_BASE_URL"] = _DEFAULT_API_BASE_URL
        elif os.environ.get("OPENAI_API_KEY", "").strip():
            os.environ["API_BASE_URL"] = "https://api.openai.com/v1"
        else:
            os.environ["API_BASE_URL"] = _DEFAULT_API_BASE_URL
    if not os.environ.get("MODEL_NAME", "").strip():
        os.environ["MODEL_NAME"] = _DEFAULT_MODEL_NAME
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
