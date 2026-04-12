"""Evaluate baseline agent with 0..1 task scores using grader."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent.baseline_agent import BaselineAgent
from env.environment import DataEnv
from env.tasks import ALL_TASKS
from grader.grader import grade
from scripts.llm_env import ensure_llm_env_defaults, has_api_key
from scripts.run_baseline import _load_env_file


def main() -> None:
    _load_env_file(_PROJECT_ROOT / ".env")
    ensure_llm_env_defaults()
    if not has_api_key():
        print(
            "HF_TOKEN or API_KEY (Hugging Face token) is required. "
            "Optional: OPENAI_API_KEY for non-HF providers.",
            file=sys.stderr,
        )
        sys.exit(1)

    seed = int(os.environ.get("BASELINE_SEED", "42"))
    agent = BaselineAgent(seed=seed, temperature=0.0)
    print(f"baseline_seed: {seed}")

    scores: list[float] = []
    for task in ALL_TASKS:
        env = DataEnv(
            data=list(task.dataset),
            question=task.question,
            correct_answer=task.correct_answer,
            max_steps=4,
        )
        obs = env.reset()
        done = False
        info: dict = {}
        while not done:
            action = agent.act(obs)
            obs, _reward, done, info = env.step(action)

        task_score = grade(info.get("computed_result"), task.correct_answer)
        scores.append(task_score)
        print(f"{task.difficulty}: score={task_score:.2f}")

    avg = sum(scores) / max(1, len(scores))
    print(f"overall_score={avg:.3f}")


if __name__ == "__main__":
    main()
