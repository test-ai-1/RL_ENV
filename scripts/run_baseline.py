"""Run the baseline agent on every benchmark task and print scores."""

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
from scripts.llm_env import ensure_llm_env_defaults, has_api_key


def _diagnose_failures(intermediate_results: list[dict]) -> list[str]:
    """Classify common failure causes from step records."""
    reasons: list[str] = []
    saw_incorrect_column = False
    saw_wrong_operation = False
    saw_bad_filtering = False

    for step in intermediate_results:
        op = str(step.get("operation") or "")
        err = str(step.get("error") or "")

        if err in {"column_missing", "empty_column", "column_not_numeric", "missing_column"}:
            saw_incorrect_column = True

        if err.startswith("expected_") or err == "unknown_operation":
            saw_wrong_operation = True

        if op == "filter" and (
            err in {"filter_requires_value", "filter_empty_value"}
            or step.get("penalty_kind") == "useless"
        ):
            saw_bad_filtering = True

    if saw_incorrect_column:
        reasons.append("incorrect column")
    if saw_wrong_operation:
        reasons.append("wrong operation")
    if saw_bad_filtering:
        reasons.append("bad filtering")
    if not reasons:
        reasons.append("no clear failure category detected")
    return reasons


def _load_env_file(path: Path) -> None:
    """Set missing keys from a simple KEY=VALUE .env file (no python-dotenv required)."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> None:
    _load_env_file(_PROJECT_ROOT / ".env")
    ensure_llm_env_defaults()
    if not has_api_key():
        print(
            "Set HF_TOKEN or API_KEY (Hugging Face read token), or OPENAI_API_KEY for OpenAI-only.\n"
            f"  • Local .env: {_PROJECT_ROOT / '.env'}\n"
            "  • HF defaults: API_BASE_URL=https://router.huggingface.co/v1, MODEL_NAME=Qwen/Qwen2.5-72B-Instruct",
            file=sys.stderr,
        )
        sys.exit(1)

    seed = int(os.environ.get("BASELINE_SEED", "42"))
    agent = BaselineAgent(seed=seed, temperature=0.0)
    print(f"baseline_seed: {seed}")

    solved_count = 0
    reward_sum = 0.0
    for task in ALL_TASKS:
        env = DataEnv(
            data=list(task.dataset),
            question=task.question,
            correct_answer=task.correct_answer,
            max_steps=4,
        )
        observation = env.reset()
        total_reward = 0.0
        done = False
        last_info: dict = {}

        while not done:
            action = agent.act(observation)
            observation, reward, done, last_info = env.step(action)
            total_reward += reward

        steps_taken = int(last_info.get("steps_taken", 0))
        print(f"{task.difficulty}:")
        print(f"  total reward: {total_reward:.4f}")
        print(f"  number of steps: {steps_taken}")
        print(f"  solved: {last_info.get('solved')}")
        print(f"  last computed: {last_info.get('computed_result')}")
        print(f"  expected: {task.correct_answer}")
        if last_info.get("operation_error"):
            print(f"  last operation_error: {last_info['operation_error']}")
        if not last_info.get("solved"):
            failure_reasons = _diagnose_failures(observation.intermediate_results)
            print(f"  failure reasons: {', '.join(failure_reasons)}")
        if last_info.get("solved"):
            solved_count += 1
        reward_sum += total_reward

    print("summary:")
    print(f"  tasks: {len(ALL_TASKS)}")
    print(f"  solved: {solved_count}/{len(ALL_TASKS)}")
    print(f"  average total reward: {reward_sum / max(1, len(ALL_TASKS)):.4f}")


if __name__ == "__main__":
    main()
