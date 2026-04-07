# Tabular Analyst OpenEnv

A multi-step, real-world-style data analysis environment where an agent solves messy spreadsheet tasks using structured actions.

## Why this is a real-world task

This environment simulates analyst workflows on CRM/pipeline/sales exports:
- inconsistent numeric formats (`$4,200`, `1.8k`, `NULL`)
- multi-step reasoning (choose metric, filter subset, aggregate)
- imperfect decisions and partial progress rewards

It is not a game/toy benchmark; it models practical tabular reasoning behavior.

## OpenEnv compliance

This project implements typed environment interfaces and metadata:

- typed models:
  - `env.actions.Action`
  - `env.observations.Observation`
- environment API:
  - `DataEnv.reset()`
  - `DataEnv.step(action)`
  - `DataEnv.state()`
- OpenEnv manifest:
  - `openenv.yaml`

## Environment description

`DataEnv` maintains:
- a working table (`_working`) that changes after filtering
- selected metric column
- intermediate action log
- trajectory credits for progress:
  - select credit up to `0.2`
  - filter credit up to `0.3`
  - final answer credit up to `0.5`

## Action space

Defined in `env/actions.py`:

```json
{
  "operation": "select_column | filter | sum | max",
  "column": "string",
  "value": "string | null",
  "confidence": "float in [0, 1]"
}
```

Rules:
- `filter` requires `column` and `value`
- `select_column`, `sum`, `max` require `column`
- non-filter ops use `value = null`

## Observation space

Defined in `env/observations.py`:
- `data`: current rows
- `question`: natural-language task
- `step`: current step index
- `max_steps`: horizon
- `selected_column`: last selected column
- `intermediate_results`: prior step outcomes

## Tasks and graders (easy -> medium -> hard)

Tasks are defined in `env/tasks.py` with at least 3 levels:
- `easy`
- `medium`
- `hard`

Agent grading is in `grader/grader.py`:
- `grade(prediction, ground_truth) -> float`
- score range: `0.0` to `1.0`

Evaluation script:
- `scripts/eval_baseline.py`
- reports per-task and overall scores in `[0, 1]`

## Reward design

Reward is meaningful and trajectory-aware:
- partial progress signals (select/filter/final)
- smooth numeric closeness shaping (`env/reward.py`)
- penalties:
  - invalid action: `-0.3`
  - useless repeated action: `-0.1`
- confidence calibration:
  - overconfident wrong final answers penalized more

## Baseline inference with reproducible scores

Baseline agent:
- `agent/baseline_agent.py`
- deterministic settings:
  - `temperature=0.0`
  - `seed=BASELINE_SEED` (default `42`)

Run baseline:

```bash
python scripts/run_baseline.py
```

Run score-based evaluation:

```bash
python scripts/eval_baseline.py
```

Environment variables:
- `GROQ_API_KEY` (required)
- `GROQ_MODEL` (optional)
- `BASELINE_SEED` (optional, default `42`)

## Setup

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

Set key:

```powershell
$env:GROQ_API_KEY="gsk_..."
```

## Hugging Face Spaces deployment (Docker)

This repo includes:
- `Dockerfile`
- `app.py` (Gradio app, port `7860`)
- `requirements.txt`

Steps:
1. Create a new **Docker Space** on Hugging Face.
2. Push this repository contents.
3. In Space secrets, set `GROQ_API_KEY` (or provide it in app input).
4. Launch Space; app runs `python app.py`.

## Project structure

- `env/` - environment, tasks, rewards, validation, table ops
- `agent/` - baseline LLM policy
- `grader/` - score function in `[0, 1]`
- `scripts/` - run/eval scripts
- `openenv.yaml` - OpenEnv manifest
- `app.py` - HF Spaces app
- `Dockerfile` - container definition

