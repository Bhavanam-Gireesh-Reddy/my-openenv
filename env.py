from __future__ import annotations

import json
import random
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator
from fastapi import Body, FastAPI

from tasks import (
    Difficulty,
    ReviewCommentRecord,
    ReviewDecision,
    ReviewSubmission,
    TaskDefinition,
    clamp_open_score,
    extract_line,
    get_task,
    grade_single_comment,
    grade_submission,
    list_tasks,
    run_simulated_linter,
    summarize_task_catalog,
)

try:
    from openenv.core.env_server.http_server import create_app as openenv_create_app
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import Action, EnvironmentMetadata, Observation, State
except ImportError:
    from fastapi import FastAPI

    class Action(BaseModel):
        model_config = ConfigDict(extra="forbid", validate_assignment=True)
        metadata: dict[str, Any] = Field(default_factory=dict)

    class Observation(BaseModel):
        model_config = ConfigDict(extra="forbid", validate_assignment=True)
        done: bool = False
        reward: float | None = None
        metadata: dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):
        model_config = ConfigDict(extra="allow", validate_assignment=True)
        episode_id: str | None = None
        step_count: int = 0

    class EnvironmentMetadata(BaseModel):
        name: str
        description: str
        version: str | None = None
        author: str | None = None

    class Environment:
        SUPPORTS_CONCURRENT_SESSIONS = True

        def reset(
            self,
            seed: int | None = None,
            episode_id: str | None = None,
            **kwargs: Any,
        ) -> Observation:
            raise NotImplementedError

        def step(
            self,
            action: Action,
            timeout_s: float | None = None,
            **kwargs: Any,
        ) -> Observation:
            raise NotImplementedError

        @property
        def state(self) -> State:
            raise NotImplementedError

        def get_metadata(self) -> EnvironmentMetadata:
            return EnvironmentMetadata(
                name=self.__class__.__name__,
                description=f"{self.__class__.__name__} environment",
                version="1.0.0",
            )

        def close(self) -> None:
            pass

    def openenv_create_app(
        env: type[Environment],
        action_cls: type[Action],
        observation_cls: type[Observation],
        env_name: str | None = None,
        max_concurrent_envs: int | None = None,
    ) -> FastAPI:
        app = FastAPI(title=env_name or env.__name__)
        env_instance = env()

        @app.post("/reset")
        async def reset_endpoint(payload: dict[str, Any] | None = None) -> dict[str, Any]:
            observation = env_instance.reset(**(payload or {}))
            return {
                "observation": observation.model_dump(),
                "reward": observation.reward,
                "done": observation.done,
            }

        @app.post("/step")
        async def step_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
            action = action_cls(**payload["action"])
            observation = env_instance.step(action)
            return {
                "observation": observation.model_dump(),
                "reward": observation.reward,
                "done": observation.done,
            }

        @app.get("/state")
        async def state_endpoint() -> dict[str, Any]:
            return env_instance.state.model_dump()

        return app


class ActionType(str, Enum):
    VIEW_FILE = "view_file"
    RUN_LINTER = "run_linter"
    ADD_COMMENT = "add_comment"
    SUBMIT_REVIEW = "submit_review"


class FileSnapshot(BaseModel):
    file_path: str = Field(..., description="Repository-relative file path.")
    content: str = Field(..., description="Full file content currently visible to the agent.")
    total_lines: int = Field(..., ge=0, description="Total number of lines in the file.")
    highlighted_line: int | None = Field(
        default=None,
        description="Optional line number that the agent most recently focused on.",
    )


class LinterFinding(BaseModel):
    file_path: str
    line_number: int = Field(..., ge=1)
    rule_id: str
    severity: str
    message: str


class LinterRun(BaseModel):
    command: str
    success: bool
    output: str
    findings: list[LinterFinding] = Field(default_factory=list)


class ReviewComment(BaseModel):
    file_path: str
    line_number: int = Field(..., ge=1)
    comment_text: str
    quality_score: float = Field(..., gt=0.0, lt=1.0)


class PullRequestStatus(BaseModel):
    pr_title: str
    pr_description: str
    changed_files: list[str]
    task_id: str
    difficulty: str
    comments: list[ReviewComment] = Field(default_factory=list)
    submitted_decision: ReviewDecision | None = None
    review_completed: bool = False
    grader_score: float | None = Field(default=None, gt=0.0, lt=1.0)


class CodeReviewAction(Action):
    action_type: ActionType = Field(..., description="The code review operation to execute.")
    file_path: str | None = Field(
        default=None,
        description="File path to open. Required for view_file.",
    )
    command: str | None = Field(
        default=None,
        description="Linter command to execute. Required for run_linter.",
    )
    line_number: int | None = Field(
        default=None,
        ge=1,
        description="Line number on the currently opened file for review comments.",
    )
    comment_text: str | None = Field(
        default=None,
        description="Comment body to attach to the current file and line.",
    )
    decision: ReviewDecision | None = Field(
        default=None,
        description="Final review decision. Required for submit_review.",
    )

    @model_validator(mode="after")
    def validate_shape(self) -> "CodeReviewAction":
        if self.action_type == ActionType.VIEW_FILE and not self.file_path:
            raise ValueError("file_path is required for view_file actions.")
        if self.action_type == ActionType.RUN_LINTER and not self.command:
            raise ValueError("command is required for run_linter actions.")
        if self.action_type == ActionType.ADD_COMMENT:
            if self.line_number is None or not self.comment_text:
                raise ValueError("line_number and comment_text are required for add_comment actions.")
        if self.action_type == ActionType.SUBMIT_REVIEW and self.decision is None:
            raise ValueError("decision is required for submit_review actions.")
        return self


class CodeReviewObservation(Observation):
    task_id: str = Field(..., description="Identifier of the active code review task.")
    task_title: str = Field(..., description="Human-readable task title.")
    message: str = Field(..., description="Status message after the latest action.")
    available_files: list[str] = Field(
        default_factory=list,
        description="Files included in the simulated pull request.",
    )
    current_file: FileSnapshot | None = Field(
        default=None,
        description="Currently visible file content, if any.",
    )
    linter_output: LinterRun | None = Field(
        default=None,
        description="Most recent linter execution result.",
    )
    pull_request_status: PullRequestStatus = Field(
        ...,
        description="Current review status, accumulated comments, and final grading output.",
    )
    steps_remaining: int = Field(
        ...,
        ge=0,
        description="Remaining step budget before the episode auto-terminates.",
    )


class EnterpriseCodeReviewState(State):
    task_id: str | None = None
    current_file_path: str | None = None
    viewed_files: list[str] = Field(default_factory=list)
    executed_commands: list[str] = Field(default_factory=list)
    comments: list[ReviewComment] = Field(default_factory=list)
    submitted_decision: ReviewDecision | None = None
    review_completed: bool = False
    final_score: float | None = None


class EnterpriseCodeReviewEnv(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._catalog = list_tasks()
        self._task: TaskDefinition | None = None
        self._state = EnterpriseCodeReviewState(episode_id=None, step_count=0)
        self._latest_linter_run: LinterRun | None = None
        self._action_counts: dict[str, int] = {}
        self._rewarded_files: set[str] = set()
        self._rewarded_commands: set[str] = set()
        self._comment_keys: set[tuple[str, int, str]] = set()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str | None = None,
        difficulty: str | None = None,
        **kwargs: Any,
    ) -> CodeReviewObservation:
        task = self._select_task(seed=seed, task_id=task_id, difficulty=difficulty)
        self._task = task
        self._state = EnterpriseCodeReviewState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task.task_id,
            current_file_path=None,
            viewed_files=[],
            executed_commands=[],
            comments=[],
            submitted_decision=None,
            review_completed=False,
            final_score=None,
        )
        self._latest_linter_run = None
        self._action_counts = {}
        self._rewarded_files = set()
        self._rewarded_commands = set()
        self._comment_keys = set()

        return self._build_observation(
            message=(
                "Review session initialized. Inspect the changed files, run safe analysis "
                "commands, leave line comments, and submit a final review."
            ),
            reward=clamp_open_score(0.0),
            done=False,
        )

    def step(
        self,
        action: CodeReviewAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CodeReviewObservation:
        task = self._require_task()

        if self._state.review_completed:
            return self._build_observation(
                message="This review is already complete. Call reset() to start a new task.",
                reward=clamp_open_score(-0.1),
                done=True,
            )

        self._state.step_count += 1

        signature = json.dumps(action.model_dump(mode="json"), sort_keys=True)
        self._action_counts[signature] = self._action_counts.get(signature, 0) + 1
        repeat_penalty = -0.05 if self._action_counts[signature] >= 3 else 0.0
        base_penalty = -0.01

        if action.action_type == ActionType.VIEW_FILE:
            message, reward = self._handle_view_file(action.file_path or "")
        elif action.action_type == ActionType.RUN_LINTER:
            message, reward = self._handle_run_linter(action.command or "")
        elif action.action_type == ActionType.ADD_COMMENT:
            message, reward = self._handle_add_comment(
                line_number=action.line_number or 1,
                comment_text=action.comment_text or "",
            )
        else:
            message, reward = self._handle_submit_review(action.decision)

        reward = clamp_open_score(reward + base_penalty + repeat_penalty)
        done = self._state.review_completed

        if not done and self._state.step_count >= task.max_steps:
            self._state.review_completed = True
            self._state.final_score = clamp_open_score(0.0)
            message = (
                f"{message} Step budget exhausted before a final review was submitted."
            )
            reward = clamp_open_score(reward - 0.35)
            done = True

        return self._build_observation(message=message, reward=reward, done=done)

    @property
    def state(self) -> EnterpriseCodeReviewState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="enterprise_code_review",
            description=(
                "Enterprise code review environment with typed PR observations, "
                "safe linter actions, deterministic graders, and trajectory-aware rewards."
            ),
            version="1.0.0",
            author="OpenAI Codex",
        )

    def _select_task(
        self,
        seed: int | None,
        task_id: str | None,
        difficulty: str | None,
    ) -> TaskDefinition:
        if task_id:
            return get_task(task_id)

        if difficulty:
            normalized = difficulty.strip().lower()
            matching = [task for task in self._catalog if task.difficulty.value == normalized]
            if not matching:
                available = sorted({task.difficulty.value for task in self._catalog})
                raise ValueError(f"Unknown difficulty '{difficulty}'. Available values: {available}")
            return matching[0]

        rng = random.Random(seed)
        return rng.choice(list(self._catalog))

    def _require_task(self) -> TaskDefinition:
        if self._task is None:
            raise RuntimeError("Environment not initialized. Call reset() before step().")
        return self._task

    def _handle_view_file(self, file_path: str) -> tuple[str, float]:
        task = self._require_task()
        if file_path not in task.files:
            return (
                f"File '{file_path}' is not part of this pull request. Avoid hallucinating files.",
                -0.2,
            )

        self._state.current_file_path = file_path
        if file_path not in self._state.viewed_files:
            self._state.viewed_files.append(file_path)

        reward = 0.03
        if file_path in task.reward_target_files and file_path not in self._rewarded_files:
            reward += 0.17
            self._rewarded_files.add(file_path)
        elif self._state.viewed_files.count(file_path) > 1:
            reward -= 0.02

        snippet = extract_line(task.files[file_path], self._task.primary_target_line)
        message = (
            f"Opened {file_path}. Review the implementation carefully."
            if snippet
            else f"Opened {file_path}."
        )
        return message, reward

    def _handle_run_linter(self, command: str) -> tuple[str, float]:
        task = self._require_task()
        output, findings, success = run_simulated_linter(task, command)
        self._latest_linter_run = LinterRun(
            command=command,
            success=success,
            output=output,
            findings=[
                LinterFinding(
                    file_path=finding.file_path,
                    line_number=finding.line_number,
                    rule_id=finding.rule_id,
                    severity=finding.severity,
                    message=finding.message,
                )
                for finding in findings
            ],
        )
        self._state.executed_commands.append(command)

        lowered = command.lower()
        if "remove-item" in lowered or " rm " in f" {lowered} " or "del " in lowered:
            return ("Blocked destructive analysis command.", -0.8)
        if not success:
            return ("Unsupported analysis command.", -0.15)

        reward = 0.05
        if command not in self._rewarded_commands:
            reward += 0.04
            self._rewarded_commands.add(command)

        if any(finding.file_path == task.primary_target_file for finding in findings):
            reward += 0.14
            message = "Linter surfaced a meaningful signal in the changed files."
        elif findings:
            reward += 0.05
            message = "Linter produced findings, but they may not be the primary issue."
        else:
            message = "Linter completed without findings."
        return message, reward

    def _handle_add_comment(self, line_number: int, comment_text: str) -> tuple[str, float]:
        task = self._require_task()
        if not self._state.current_file_path:
            return ("Open a file before placing a line comment.", -0.15)

        current_file = self._state.current_file_path
        normalized_comment = " ".join(comment_text.lower().split())
        dedupe_key = (current_file, line_number, normalized_comment)
        if dedupe_key in self._comment_keys:
            return ("Duplicate comment detected.", -0.08)

        record = ReviewCommentRecord(
            file_path=current_file,
            line_number=line_number,
            comment_text=comment_text,
        )
        graded = grade_single_comment(task, record)
        self._comment_keys.add(dedupe_key)
        self._state.comments.append(
            ReviewComment(
                file_path=current_file,
                line_number=line_number,
                comment_text=comment_text,
                quality_score=graded.score,
            )
        )

        if graded.score >= 0.85:
            reward = 0.34
        elif graded.score >= 0.6:
            reward = 0.2
        elif graded.score >= 0.35:
            reward = 0.08
        else:
            reward = -0.04

        message = (
            f"Comment added on {current_file}:{line_number}. "
            f"Quality score={graded.score:.4f}."
        )
        return message, reward

    def _handle_submit_review(self, decision: ReviewDecision | None) -> tuple[str, float]:
        task = self._require_task()
        submission = ReviewSubmission(
            comments=tuple(
                ReviewCommentRecord(
                    file_path=comment.file_path,
                    line_number=comment.line_number,
                    comment_text=comment.comment_text,
                )
                for comment in self._state.comments
            ),
            decision=decision,
        )
        final_score = grade_submission(task, submission)
        self._state.submitted_decision = decision
        self._state.review_completed = True
        self._state.final_score = final_score

        if decision == task.expected_decision:
            reward = (2.0 * final_score) - 0.25
        else:
            reward = (1.5 * final_score) - 0.55

        reward = clamp_open_score(reward)

        message = (
            f"Review submitted with decision '{decision.value if decision else 'unknown'}'. "
            f"Deterministic grader score={final_score:.4f}."
        )
        return message, reward

    def _build_observation(
        self,
        message: str,
        reward: float,
        done: bool,
    ) -> CodeReviewObservation:
        task = self._require_task()
        current_file = None
        if self._state.current_file_path:
            content = task.files[self._state.current_file_path]
            current_file = FileSnapshot(
                file_path=self._state.current_file_path,
                content=content,
                total_lines=len(content.splitlines()),
                highlighted_line=self._task.primary_target_line
                if self._state.current_file_path == task.primary_target_file
                else None,
            )

        pull_request_status = PullRequestStatus(
            pr_title=task.pr_title,
            pr_description=task.pr_description,
            changed_files=list(task.files.keys()),
            task_id=task.task_id,
            difficulty=task.difficulty.value,
            comments=list(self._state.comments),
            submitted_decision=self._state.submitted_decision,
            review_completed=self._state.review_completed,
            grader_score=self._state.final_score,
        )

        return CodeReviewObservation(
            done=done,
            reward=reward,
            metadata={
                "episode_id": self._state.episode_id,
                "task_summary": task.summary,
                "task_catalog": summarize_task_catalog(),
            },
            task_id=task.task_id,
            task_title=task.title,
            message=message,
            available_files=list(task.files.keys()),
            current_file=current_file,
            linter_output=self._latest_linter_run,
            pull_request_status=pull_request_status,
            steps_remaining=max(0, task.max_steps - self._state.step_count),
        )


class ResetPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    seed: int | None = None
    episode_id: str | None = None
    task_id: str | None = None
    difficulty: str | None = None


class StepPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    action: CodeReviewAction
    timeout_s: float | None = None


class ResetStepEnvelope(BaseModel):
    observation: CodeReviewObservation
    reward: float | None = None
    done: bool


class SchemaEnvelope(BaseModel):
    action: dict[str, Any]
    observation: dict[str, Any]
    state: dict[str, Any]


class JsonRpcErrorEnvelope(BaseModel):
    code: int
    message: str


class JsonRpcEnvelope(BaseModel):
    jsonrpc: str = "2.0"
    id: str | int | None = None
    result: dict[str, Any] | None = None
    error: JsonRpcErrorEnvelope | None = None


def build_app() -> FastAPI:
    app = FastAPI(
        title="Enterprise Code Review OpenEnv API",
        version="1.0.0",
        description=(
            "Persistent HTTP API for the Enterprise Code Review OpenEnv task. "
            "Use /reset, /step, and /state to run full review trajectories."
        ),
    )

    app.state.env = EnterpriseCodeReviewEnv()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        app.state.env.close()

    @app.get("/")
    def root() -> dict[str, Any]:
        env = app.state.env
        return {
            "status": "ok",
            "name": "enterprise_code_review",
            "description": env.get_metadata().description,
            "docs_url": "/docs",
            "endpoints": ["/reset", "/step", "/state", "/schema", "/metadata", "/health"],
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/metadata")
    def metadata() -> EnvironmentMetadata:
        return app.state.env.get_metadata()

    @app.get("/schema")
    def schema() -> SchemaEnvelope:
        return SchemaEnvelope(
            action=CodeReviewAction.model_json_schema(),
            observation=CodeReviewObservation.model_json_schema(),
            state=EnterpriseCodeReviewState.model_json_schema(),
        )

    @app.post("/mcp")
    def mcp(request: dict[str, Any] | None = Body(default=None)) -> JsonRpcEnvelope:
        request_id = None if not isinstance(request, dict) else request.get("id")
        return JsonRpcEnvelope(
            id=request_id,
            error=JsonRpcErrorEnvelope(
                code=-32601,
                message=(
                    "MCP methods are not implemented for this HTTP surface. "
                    "Use the simulation endpoints /reset, /step, and /state."
                ),
            ),
        )

    @app.get("/state")
    def state() -> EnterpriseCodeReviewState:
        return app.state.env.state

    @app.post("/reset")
    def reset(request: ResetPayload = Body(default_factory=ResetPayload)) -> ResetStepEnvelope:
        observation = app.state.env.reset(**request.model_dump(exclude_unset=True))
        return ResetStepEnvelope(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
        )

    @app.post("/step")
    def step(request: StepPayload) -> ResetStepEnvelope:
        observation = app.state.env.step(
            request.action,
            timeout_s=request.timeout_s,
        )
        return ResetStepEnvelope(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
        )

    return app


app = build_app()


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
