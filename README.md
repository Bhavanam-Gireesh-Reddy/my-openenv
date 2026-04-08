---
title: Enterprise Code Review OpenEnv
emoji: 🛠️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - code-review
  - hackathon
---

# Enterprise Code Review OpenEnv

`Enterprise Code Review` is a production-style OpenEnv environment for RL post-training on realistic software review workflows. The agent operates like a reviewer on an internal pull request: it opens changed files, runs safe analysis commands, leaves line comments, and submits a final review decision.

## Motivation

Code review is high-signal RL data because strong trajectories are not just about producing the right final answer. Good reviewers gather context efficiently, use tools selectively, identify risk precisely, and communicate actionable feedback. This environment rewards that full process.

The tasks cover three practical review patterns:

1. `easy_mutable_default`: catch a state leak caused by a mutable default argument.
2. `medium_quadratic_reporting`: identify a quadratic aggregation routine that should be reduced to linear work.
3. `hard_path_traversal`: find a cross-file path traversal vulnerability in an artifact download flow.

## OpenEnv API

The environment implements the standard OpenEnv lifecycle:

1. `reset(seed=None, episode_id=None, task_id=None, difficulty=None) -> CodeReviewObservation`
2. `step(CodeReviewAction) -> CodeReviewObservation`
3. `state -> EnterpriseCodeReviewState`

The FastAPI app is exposed from [env.py](/Users/hp/Downloads/Hackathon/env.py) via `env:app`, matching the `openenv.yaml` manifest in [openenv.yaml](/Users/hp/Downloads/Hackathon/openenv.yaml).

## Action Space

The action model is `CodeReviewAction` in [env.py](/Users/hp/Downloads/Hackathon/env.py). It supports exactly four operations:

1. `view_file(file_path)`
2. `run_linter(command)`
3. `add_comment(line_number, comment_text)`
4. `submit_review(decision)`

`add_comment` attaches to the file most recently opened with `view_file`, which mirrors the common review UI pattern where reviewers comment on the file currently in focus.

## Observation Space

Each `CodeReviewObservation` contains:

1. `task_id` and `task_title`
2. `message` describing the latest environment response
3. `available_files` in the pull request
4. `current_file` with full contents and line counts
5. `linter_output` containing the latest structured safe-analysis result
6. `pull_request_status` with changed files, accumulated review comments, submitted decision, and final deterministic grader score
7. `steps_remaining` so policies can manage the trajectory budget

The observation is intentionally typed and information-dense so policies can learn both tool use and review judgment.

## Reward Shaping

The reward function provides trajectory-level signal instead of only terminal grading:

1. Positive reward for opening relevant files for the first time.
2. Positive reward for safe linter usage, with larger reward when the command reveals the core issue.
3. Positive reward for comments that land on the right file and line and include the right failure mode and fix direction.
4. Strong terminal reward for matching the deterministic grader on `submit_review`.
5. Penalties for hallucinated files, unsupported or destructive commands, duplicate comments, repeated looping behavior, and exhausting the step budget without submitting a review.

## Task Design And Grading

The full task registry and graders live in [tasks.py](/Users/hp/Downloads/Hackathon/tasks.py).

Each task is fully deterministic:

1. Source files are embedded directly in the task definition.
2. Safe linter outputs are simulated from fixed findings per task.
3. Final scores are floats in `(0.0, 1.0)`.
4. Graders combine:
   - file correctness
   - line proximity
   - issue identification keywords
   - rationale and remediation keywords
   - final review decision correctness

This makes evaluation stable across repeated runs while still requiring realistic reviewer behavior.

## Baseline Inference

The baseline agent lives in [inference.py](/Users/hp/Downloads/Hackathon/inference.py).

Behavior:

1. Resets the environment for each task.
2. Views every changed file.
3. Runs three safe analysis commands:
   - `python -m ruff check .`
   - `python -m semgrep --config perf .`
   - `python -m bandit -r .`
4. Uses lightweight heuristics to infer the likely issue.
5. Uses the `OpenAI` Python client for LLM-based comment drafting when `API_BASE_URL` and `API_KEY` are present, using `MODEL_NAME` when set and otherwise defaulting to `openai/gpt-4.1-mini`.
6. Falls back to deterministic comments locally when those variables are missing.

The script emits strict stdout markers only:

1. `[START] ...`
2. `[STEP] Action: {...} | Reward: ...`
3. `[END] ...`

## Baseline Scores

Local reproducible baseline run from [inference.py](/Users/hp/Downloads/Hackathon/inference.py):

1. `easy_mutable_default`: `0.9990`
2. `medium_quadratic_reporting`: `0.8687`
3. `hard_path_traversal`: `0.9062`

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn env:app --host 0.0.0.0 --port 8000
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn env:app --host 0.0.0.0 --port 8000
```

## Running The Baseline

Set the required environment variables:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export API_KEY="your_proxy_api_key_here"
export MODEL_NAME="openai/gpt-4.1-mini"
python inference.py
```

Windows PowerShell:

```powershell
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:API_KEY = "your_proxy_api_key_here"
$env:MODEL_NAME = "openai/gpt-4.1-mini"
python inference.py
```

## Hugging Face Space Secrets

When you create the Docker Space, add these repository secrets or variables:

1. `API_BASE_URL`
2. `API_KEY`
3. `MODEL_NAME` (optional, defaults to `openai/gpt-4.1-mini`)

An example local template is available in [.env.example](/Users/hp/Downloads/Hackathon/.env.example).

## Docker

Build and run the environment container:

```bash
docker build -t enterprise-code-review-openenv .
docker run --rm -p 8000:8000 enterprise-code-review-openenv
```

For validator-style local checks:

```bash
openenv validate .
openenv validate --url http://127.0.0.1:8000
```

## File Guide

1. [env.py](/Users/hp/Downloads/Hackathon/env.py): typed environment models, reward shaping, and FastAPI app export
2. [tasks.py](/Users/hp/Downloads/Hackathon/tasks.py): tasks, simulated linter outputs, and deterministic graders
3. [inference.py](/Users/hp/Downloads/Hackathon/inference.py): baseline agent using the `OpenAI` client
4. [openenv.yaml](/Users/hp/Downloads/Hackathon/openenv.yaml): OpenEnv manifest
5. [Dockerfile](/Users/hp/Downloads/Hackathon/Dockerfile): container image definition
6. [requirements.txt](/Users/hp/Downloads/Hackathon/requirements.txt): Python dependencies

## Expected Utility

This environment is useful for:

1. RL training on multi-step reviewer behavior rather than single-turn classification.
2. Evaluating tool use, precision, and restraint under a step budget.
3. Benchmarking secure, production-style code review for enterprise engineering assistants.
