"""Hugging Face Spaces entrypoint: FastAPI + Gradio (health + reset for validators)."""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout

import gradio as gr
from fastapi import FastAPI

from scripts.eval_baseline import main as eval_main

fastapi_app = FastAPI(title="Tabular Analyst OpenEnv")


@fastapi_app.get("/health")
def health() -> dict[str, str]:
    """Space liveness: must return 200."""
    return {"status": "ok"}


@fastapi_app.post("/reset")
def reset() -> dict[str, str]:
    """Validator hook: acknowledge reset (no persistent server-side env state in this Space)."""
    return {"status": "ok", "reset": True}


def run_eval(seed: int, model: str, api_key: str) -> str:
    os.environ["BASELINE_SEED"] = str(int(seed))
    if model.strip():
        os.environ["MODEL_NAME"] = model.strip()
        os.environ["OPENAI_MODEL"] = model.strip()
    if api_key.strip():
        os.environ["HF_TOKEN"] = api_key.strip()
        os.environ["OPENAI_API_KEY"] = api_key.strip()
    api_base = os.environ.get("API_BASE_URL", "").strip()
    if not api_base:
        os.environ["API_BASE_URL"] = "https://api.openai.com/v1"

    output = io.StringIO()
    try:
        with redirect_stdout(output):
            eval_main()
        return output.getvalue()
    except Exception as exc:  # noqa: BLE001
        return f"Evaluation failed: {type(exc).__name__}: {exc}"


demo = gr.Interface(
    fn=run_eval,
    inputs=[
        gr.Number(label="BASELINE_SEED", value=42, precision=0),
        gr.Textbox(label="MODEL_NAME (optional)", value="gpt-4o-mini"),
        gr.Textbox(label="HF_TOKEN / API key", type="password"),
    ],
    outputs=gr.Textbox(label="Run output", lines=20),
    title="Tabular Analyst OpenEnv Baseline",
    description="Runs evaluation. Set Space secrets: API_BASE_URL, MODEL_NAME, HF_TOKEN.",
)

try:
    from gradio import mount_gradio_app
except ImportError:  # pragma: no cover
    mount_gradio_app = getattr(gr, "mount_gradio_app", None)

if mount_gradio_app is None:  # pragma: no cover
    raise RuntimeError("gradio>=4.44 required (mount_gradio_app). pip install -r requirements.txt")

app = mount_gradio_app(fastapi_app, demo, path="/")
