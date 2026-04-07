"""Hugging Face Spaces entrypoint for running baseline evaluation."""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout

import gradio as gr

from scripts.eval_baseline import main as eval_main


def run_eval(seed: int, model: str, api_key: str) -> str:
    os.environ["BASELINE_SEED"] = str(seed)
    if model.strip():
        os.environ["GROQ_MODEL"] = model.strip()
    if api_key.strip():
        os.environ["GROQ_API_KEY"] = api_key.strip()

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
        gr.Textbox(label="GROQ_MODEL (optional)", value="openai/gpt-oss-120b"),
        gr.Textbox(label="GROQ_API_KEY", type="password"),
    ],
    outputs=gr.Textbox(label="Run output", lines=20),
    title="Tabular Analyst OpenEnv Baseline",
    description="Runs the baseline agent on easy/medium/hard tasks and reports 0..1 scores.",
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
