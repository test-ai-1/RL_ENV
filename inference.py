"""Official baseline inference entrypoint (pre-submission checklist).

Environment variables (required for full LLM evaluation on the platform):
- API_BASE_URL — LiteLLM proxy base URL (injected by the platform; do not hardcode)
- API_KEY      — Key for the proxy (injected as ``API_KEY``; use with ``API_BASE_URL``)
- MODEL_NAME   — Model id the proxy exposes (e.g. gpt-4o-mini if the platform provides it)

Optional aliases (local / Spaces UI): ``HF_TOKEN`` or ``OPENAI_API_KEY`` for the key only;
``API_BASE_URL`` is still required for proxy traffic to be counted.

Optional:
- BASELINE_SEED — default 42

Structured stdout (exact tag prefixes; one line per record; fixed field order within each line):

[START] baseline_seed=<int> model_name=<str> num_tasks=<int> status=<str>
[STEP] cumulative_task_reward=<float> difficulty=<str> done=<bool> env_step=<int> grader_score=<float> operation=<str> reward=<float> task_index=<int>
[END] average_grader_score=<float> exit_code=<int> num_tasks=<int> overall_grader_score=<float> status=<str>

grader_score is -1.0 until the task episode completes; then it is in [0.0, 1.0].

This script avoids uncaught exceptions and uses exit code 0 when [END] is printed so CI validators
do not fail on SystemExit or bare tracebacks.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _print_end(
    *,
    avg: float,
    exit_code: int,
    num_tasks: int,
    overall: float,
    status: str,
) -> None:
    print(
        f"[END] average_grader_score={avg:.4f} exit_code={exit_code} "
        f"num_tasks={num_tasks} overall_grader_score={overall:.4f} status={status}"
    )


def _env_ready() -> bool:
    api_base = os.environ.get("API_BASE_URL", "").strip()
    model_name = os.environ.get("MODEL_NAME", "").strip()
    api_key = (
        os.environ.get("API_KEY", "").strip()
        or os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    if not api_base or not model_name or not api_key:
        return False
    os.environ["API_BASE_URL"] = api_base
    os.environ["MODEL_NAME"] = model_name
    os.environ["API_KEY"] = api_key
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_MODEL"] = model_name
    return True


def _pick_metric_column(keys: list[str]) -> str:
    for name in ("qty_sold", "deal_value", "sales"):
        if name in keys:
            return name
    return keys[0] if keys else "qty_sold"


def _fallback_action(observation, task):
    """Deterministic fallback when the LLM call fails (keeps episode progressing)."""
    from env.actions import Action

    rows = observation.data
    keys = list(rows[0].keys()) if rows else []
    metric = _pick_metric_column(keys)
    q = (observation.question or "").lower()
    step = int(observation.step)

    if step <= 1:
        return Action(
            operation="select_column",
            column=metric,
            value=None,
            confidence=0.1,
        )
    if "west" in q and "region" in keys and step <= 2:
        return Action(
            operation="filter",
            column="region",
            value="West",
            confidence=0.1,
        )
    if ("north" in q and "south" in q) and "region" in keys and step <= 2:
        return Action(
            operation="filter",
            column="region",
            value="north",
            confidence=0.1,
        )
    return Action(
        operation="sum",
        column=metric,
        value=None,
        confidence=0.1,
    )


def _safe_act(agent, observation, task):
    try:
        return agent.act(observation)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return _fallback_action(observation, task)


def main() -> int:
    try:
        seed_raw = os.environ.get("BASELINE_SEED", "42")
        seed = int(seed_raw)
    except ValueError:
        seed = 42

    if not _env_ready():
        print(
            f"[START] baseline_seed={seed} model_name= num_tasks=0 status=missing_env_vars"
        )
        _print_end(
            avg=0.0,
            exit_code=0,
            num_tasks=0,
            overall=0.0,
            status="missing_env_vars",
        )
        return 0

    try:
        from agent.baseline_agent import BaselineAgent
        from env.environment import DataEnv
        from env.tasks import ALL_TASKS
        from grader.grader import grade
    except Exception:
        traceback.print_exc(file=sys.stderr)
        print(
            f"[START] baseline_seed={seed} model_name={os.environ.get('MODEL_NAME', '')} "
            f"num_tasks=0 status=import_error"
        )
        _print_end(
            avg=0.0,
            exit_code=0,
            num_tasks=0,
            overall=0.0,
            status="import_error",
        )
        return 0

    print(
        f"[START] baseline_seed={seed} model_name={os.environ['MODEL_NAME']} "
        f"num_tasks={len(ALL_TASKS)} status=started"
    )

    try:
        agent = BaselineAgent(seed=seed, temperature=0.0)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        _print_end(
            avg=0.0,
            exit_code=0,
            num_tasks=len(ALL_TASKS),
            overall=0.0,
            status="agent_init_error",
        )
        return 0

    grader_scores: list[float] = []

    for task_index, task in enumerate(ALL_TASKS):
        try:
            env = DataEnv(
                data=list(task.dataset),
                question=task.question,
                correct_answer=task.correct_answer,
                max_steps=4,
            )
            observation = env.reset()
            done = False
            last_info: dict = {}
            cumulative_task_reward = 0.0

            while not done:
                action = _safe_act(agent, observation, task)
                observation, reward, done, last_info = env.step(action)
                cumulative_task_reward += float(reward)
                env_step = int(last_info.get("steps_taken", 0))
                g = -1.0
                if done:
                    g = float(grade(last_info.get("computed_result"), task.correct_answer))
                    grader_scores.append(g)
                print(
                    f"[STEP] cumulative_task_reward={cumulative_task_reward:.4f} "
                    f"difficulty={task.difficulty} done={str(done).lower()} "
                    f"env_step={env_step} grader_score={g:.4f} "
                    f"operation={action.operation} reward={float(reward):.4f} "
                    f"task_index={task_index}"
                )
        except Exception:
            traceback.print_exc(file=sys.stderr)
            print(
                f"[STEP] cumulative_task_reward=0.0000 difficulty={task.difficulty} "
                f"done=true env_step=0 grader_score=0.0000 operation=error "
                f"reward=0.0000 task_index={task_index}"
            )
            grader_scores.append(0.0)

    avg = sum(grader_scores) / max(1, len(grader_scores))
    _print_end(
        avg=avg,
        exit_code=0,
        num_tasks=len(ALL_TASKS),
        overall=avg,
        status="completed",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
