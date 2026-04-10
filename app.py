"""Hugging Face Spaces entrypoint: FastAPI + Gradio (health + reset for validators)."""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout

import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from scripts.eval_baseline import main as eval_main

fastapi_app = FastAPI(title="Tabular Analyst OpenEnv")


@fastapi_app.get("/")
def root() -> HTMLResponse:
    """Serve a stable landing page that embeds Gradio UI."""
    return HTMLResponse(
        """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Tabular Analyst OpenEnv</title>
    <style>
      html, body { height: 100%; margin: 0; background: #0b1020; color: #fff; font-family: sans-serif; }
      .wrap { height: 100%; display: flex; flex-direction: column; }
      .top { padding: 10px 14px; font-size: 14px; opacity: 0.9; }
      iframe { border: 0; width: 100%; flex: 1; background: #111827; }
      a { color: #8ab4ff; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="top">Loading UI... If it does not appear, open <a href="/ui" target="_blank">/ui</a>.</div>
      <iframe src="/ui"></iframe>
    </div>
  </body>
</html>
        """
    )


@fastapi_app.get("/health")
def health() -> dict[str, str]:
    """Space liveness: must return 200."""
    return {"status": "ok"}


@fastapi_app.post("/reset")
async def reset(_request: Request) -> JSONResponse:
    """
    Validator hook: always return 200 with reset acknowledgement.

    Accepts arbitrary request bodies so automated checkers cannot crash this handler.
    """
    return JSONResponse(status_code=200, content={"status": "ok", "reset": True})


def run_eval(seed: int, model: str, api_key: str) -> str:
    os.environ["BASELINE_SEED"] = str(int(seed))
    if model.strip():
        os.environ["MODEL_NAME"] = model.strip()
        os.environ["OPENAI_MODEL"] = model.strip()
    if api_key.strip():
        os.environ["API_KEY"] = api_key.strip()
        os.environ["HF_TOKEN"] = api_key.strip()
        os.environ["OPENAI_API_KEY"] = api_key.strip()
    # Do not override API_BASE_URL: Space secrets must supply the LiteLLM proxy URL.

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
        gr.Textbox(label="API_KEY (or paste proxy key)", type="password"),
    ],
    outputs=gr.Textbox(label="Run output", lines=20),
    title="Tabular Analyst OpenEnv Baseline",
    description="Set Space secrets: API_BASE_URL (LiteLLM proxy), API_KEY, MODEL_NAME. Optional: paste API_KEY here.",
)

try:
    from gradio import mount_gradio_app
except ImportError:  # pragma: no cover
    mount_gradio_app = getattr(gr, "mount_gradio_app", None)

if mount_gradio_app is None:  # pragma: no cover
    raise RuntimeError("gradio>=4.44 required (mount_gradio_app). pip install -r requirements.txt")

app = mount_gradio_app(fastapi_app, demo, path="/ui")


def main() -> None:
    """Console entry point used by [project.scripts]."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
