"""Official baseline inference entrypoint (pre-submission checklist).

Environment variables (required for evaluation):
- API_BASE_URL  — Chat Completions base URL (e.g. https://api.openai.com/v1)
- MODEL_NAME    — Model id for inference
- HF_TOKEN      — API key (OpenAI-compatible; passed to OpenAI client)

Optional:
- BASELINE_SEED — default 42

Structured stdout (exact tag prefixes; one line per record; fixed field order within each line):

[START] baseline_seed=<int> model_name=<str> num_tasks=<int> status=<str>
[STEP] cumulative_task_reward=<float> difficulty=<str> done=<bool> env_step=<int> grader_score=<float> operation=<str> reward=<float> task_index=<int>
[END] average_grader_score=<float> exit_code=<int> num_tasks=<int> overall_grader_score=<float> status=<str>

grader_score is -1.0 until the task episode completes; then it is in [0.0, 1.0].
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _sync_agent_env() -> None:
    """Map competition env vars into values BaselineAgent understands."""
    api_base = os.environ.get("API_BASE_URL", "").strip()
    model_name = os.environ.get("MODEL_NAME", "").strip()
    hf_token = os.environ.get("HF_TOKEN", "").strip()

    if not api_base or not model_name or not hf_token:
        print(
            "[END] average_grader_score=0.0000 exit_code=2 num_tasks=0 "
            "overall_grader_score=0.0000 status=missing_env_vars"
        )
        sys.exit(2)

    os.environ["API_BASE_URL"] = api_base
    os.environ["MODEL_NAME"] = model_name
    os.environ["HF_TOKEN"] = hf_token
    # Back-compat with internal defaults
    os.environ["OPENAI_API_KEY"] = hf_token
    os.environ["OPENAI_MODEL"] = model_name


def main() -> None:
    _sync_agent_env()

    # Import after env is wired (agent reads os.environ at init)
    from agent.baseline_agent import BaselineAgent
    from env.environment import DataEnv
    from env.tasks import ALL_TASKS
    from grader.grader import grade

    seed = int(os.environ.get("BASELINE_SEED", "42"))
    print(
        f"[START] baseline_seed={seed} model_name={os.environ['MODEL_NAME']} "
        f"num_tasks={len(ALL_TASKS)} status=started"
    )

    agent = BaselineAgent(seed=seed, temperature=0.0)
    grader_scores: list[float] = []

    for task_index, task in enumerate(ALL_TASKS):
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
            action = agent.act(observation)
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

    avg = sum(grader_scores) / max(1, len(grader_scores))
    print(
        f"[END] average_grader_score={avg:.4f} exit_code=0 num_tasks={len(ALL_TASKS)} "
        f"overall_grader_score={avg:.4f} status=completed"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(
            "[END] average_grader_score=0.0000 exit_code=1 num_tasks=0 "
            f"overall_grader_score=0.0000 status=error_{type(exc).__name__}"
        )
        sys.exit(1)
