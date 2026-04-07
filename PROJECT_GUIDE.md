# Project Guide (Beginner-Friendly)

## 1) What This Project Is

This project is a **mini reinforcement-learning style environment** where an LLM agent learns to solve spreadsheet-like questions in multiple steps.

Think of it like this:
- You have a table (rows + columns).
- You ask a question like: "What is total sales in West?"
- The agent must do actions step-by-step:
  1. choose a column,
  2. filter rows (if needed),
  3. compute final result (`sum` or `max`).

The environment gives a reward based on:
- progress made,
- correctness of final answer,
- whether the action was invalid/useless,
- confidence calibration.

---

## 2) High-Level Architecture

Main parts:

- `env/` -> the simulation environment (state, actions, rewards, tasks)
- `agent/` -> the LLM-based policy that picks the next action
- `scripts/run_baseline.py` -> runner script to execute tasks end-to-end
- `grader/` -> standalone evaluator utility

Core runtime loop:
1. Load a task (dataset + question + answer)
2. `env.reset()` -> get first observation
3. Repeat:
   - send observation to agent
   - agent returns action JSON
   - env applies action with `step()`
   - env returns next observation + reward + done
4. Stop when done

---

## 3) File-by-File Explanation

## `env/actions.py`

Defines what one action looks like.

Action schema:
- `operation`: `"select_column" | "filter" | "sum" | "max"`
- `column`: target column name
- `value`: used only for `filter`
- `confidence`: float in `[0, 1]`

Validation rules:
- For `filter`, both `column` and `value` are required.
- For `select_column`, `sum`, `max`, `column` is required and `value` is set to `None`.

Why this matters:
- Guarantees the agent always outputs structured, valid actions.

---

## `env/observations.py`

Defines what the agent sees each step:
- `data`: current working rows
- `question`: NL question
- `step`: current step index
- `max_steps`: limit
- `selected_column`: last selected column
- `intermediate_results`: log of previous action outcomes

Why this matters:
- The agent is not blind; it can reason from prior mistakes and current filtered table.

---

## `env/table_ops.py`

Implements low-level table operations and number parsing.

Important functions:
- `filter_working_set(...)`
- `sum_working_set(...)`
- `max_working_set(...)`
- `select_column_working(...)`
- `parse_number(...)`

Messy data handling:
- Parses `$`, commas, `k`, `m`
- Handles `NULL`, `N/A`, `None`
- Ignores non-numeric cells in numeric aggregation

Why this matters:
- Real-world tables are messy. This file makes operations robust.

---

## `env/action_validation.py`

Classifies bad actions and applies fixed penalties:
- invalid action -> `-0.3`
- useless repeated action -> `-0.1`

Key functions:
- `classify_missing_column(...)`
- `classify_invalid_for_state(...)`
- `classify_useless(...)`
- `apply_penalty(...)`

Why this matters:
- Prevents random action spam and rewards meaningful trajectories.

---

## `env/reward.py`

Contains reward shaping logic.

Main ideas:
1. `shaped_reward(computed, expected)` returns score in `[0, 1]` by numeric closeness.
2. Robust numeric preprocessing converts values like `"1k"` or `"$1,200"` to numbers.
3. `confidence_adjusted_reward(...)` calibrates reward with confidence:
   - high confidence + wrong final answer -> stronger penalty
   - correct final answer can get slight confidence-based boost

Why this matters:
- Encourages both correctness and calibrated confidence.

---

## `env/environment.py`

This is the heart of the project (`DataEnv` class).

Responsibilities:
- maintain environment state
- process one action at a time
- compute reward
- terminate episode when solved or step limit reached

State inside env:
- `_working`: currently active rows
- `_selected_column`: selected metric column
- `_intermediate_results`: per-step logs
- `_select_credit`, `_filter_credit`, `_final_credit`: trajectory progress credits

How `step(action)` works:
1. Validate and apply operation.
2. Detect invalid/useless actions.
3. Update progress credits:
   - select credit up to `0.2`
   - filter credit up to `0.3`
   - final-answer credit up to `0.5`
4. Compute incremental reward = new_total_progress - previous_total_progress.
5. Override with penalties when needed.
6. Add confidence calibration for final compute steps.
7. Return `(next_observation, reward, done, info)`.

Done conditions:
- solved by exact `sum`/`max` match
- or reached `max_steps`

Why this matters:
- This creates trajectory-aware reward, not only end-of-episode reward.

---

## `env/tasks.py`

Defines benchmark tasks using `DataAnalystTask` dataclass.

Each task has:
- `difficulty`
- `dataset` (list of rows)
- `question`
- `correct_answer`

Current tasks:
- Easy: direct sum with noisy values
- Medium: filter + sum
- Hard: compare group totals (north vs south), return larger total

Why this matters:
- Provides controlled training/evaluation scenarios.

---

## `agent/baseline_agent.py`

LLM policy adapter using Groq OpenAI-compatible API.

What it does:
1. Builds a strict system prompt with operation rules and JSON schema.
2. Sends observation to model.
3. Parses JSON response.
4. Validates it using `Action.model_validate(...)`.

Prompt policy:
- Think step-by-step (select -> filter -> compute)
- Output JSON only with keys:
  - `operation`
  - `column`
  - `value`
  - `confidence`

Why this matters:
- Converts natural language reasoning into environment-compatible actions.

---

## `scripts/run_baseline.py`

Main executable script.

What it does:
- loads `.env` for API key/model
- creates `BaselineAgent`
- loops over all tasks:
  - reset env
  - while not done:
    - get action from agent
    - apply env step
    - accumulate reward
- prints:
  - total reward
  - number of steps
  - solved status
  - expected/computed
  - failure reasons (if unsolved)

Failure diagnostics include:
- incorrect column
- wrong operation
- bad filtering

Why this matters:
- Gives you quick feedback for debugging agent behavior.

---

## `grader/grader.py`

A separate utility that grades a prediction vs truth:
- `1.0` exact
- `0.5` partial
- `0.0` wrong

It uses shared number parsing so grading is consistent with environment conventions.

Why this matters:
- Useful for offline scoring outside the full RL loop.

---

## `__init__.py` files

Mostly package markers and tiny descriptions so Python treats folders as importable modules.

---

## 4) End-to-End Example (Concrete)

Suppose task asks:
"Total sales for West region?"

Possible trajectory:
1. Agent action: `{"operation":"select_column","column":"sales","value":null,"confidence":0.78}`
2. Agent action: `{"operation":"filter","column":"region","value":"West","confidence":0.70}`
3. Agent action: `{"operation":"sum","column":"sales","value":null,"confidence":0.84}`

Env then:
- awards select/filter/final progress increments,
- penalizes invalid/useless actions if any,
- checks if final sum matches expected answer.

---

## 5) How to Run

From project root:

```bash
python scripts/run_baseline.py
```

Requirements:
- set `GROQ_API_KEY`
- optional `GROQ_MODEL`

PowerShell example:

```powershell
$env:GROQ_API_KEY = "gsk_..."
python scripts/run_baseline.py
```

---

## 6) Beginner Concepts You Learn From This Project

- data modeling with Pydantic (`Action`, `Observation`)
- environment design (`reset`, `step`, state transitions)
- reward shaping and penalties
- robust parsing for messy real-world data
- strict LLM output contracts (JSON schema)
- episode loops and trajectory accumulation

---

## 7) Current Limitations (Good to Know)

- filter supports equality matching only (no >, <, contains)
- very short horizon (`max_steps=4`)
- hard tasks are still constrained by simple operation space
- confidence calibration currently focuses on final compute steps

---

## 8) Suggested Next Improvements

- add richer filter operators (`gt`, `lt`, `in`, date ranges)
- add grouped operations (`group_sum`, `argmax_group`)
- track confidence calibration metrics over many runs
- add unit tests per module (`table_ops`, reward, env transitions)

---

## 9) Quick Mental Model

If you remember one thing:

This project is a **small training playground** where an LLM agent learns to solve tabular analysis questions through **structured multi-step actions**, with rewards that encourage correct, efficient, and calibrated behavior.

