from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Iterable, Sequence


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ReviewDecision(str, Enum):
    APPROVE = "approve"
    COMMENT = "comment"
    REQUEST_CHANGES = "request_changes"


@dataclass(frozen=True)
class ReviewCommentRecord:
    file_path: str
    line_number: int
    comment_text: str


@dataclass(frozen=True)
class ReviewSubmission:
    comments: tuple[ReviewCommentRecord, ...]
    decision: ReviewDecision | None


@dataclass(frozen=True)
class LinterFindingSpec:
    file_path: str
    line_number: int
    rule_id: str
    severity: str
    message: str


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    difficulty: Difficulty
    title: str
    summary: str
    pr_title: str
    pr_description: str
    files: dict[str, str]
    expected_decision: ReviewDecision
    primary_target_file: str
    primary_target_line: int
    related_files: tuple[str, ...]
    linter_findings: dict[str, tuple[LinterFindingSpec, ...]]
    issue_keyword_groups: tuple[tuple[str, ...], ...]
    rationale_keyword_groups: tuple[tuple[str, ...], ...]
    reward_target_files: tuple[str, ...]
    max_steps: int = 14


@dataclass(frozen=True)
class GradedComment:
    score: float
    file_score: float
    line_score: float
    issue_score: float
    rationale_score: float


OPEN_SCORE_EPSILON = 0.0001


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _matches_any(text: str, phrases: Sequence[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _score_keyword_groups(text: str, groups: Sequence[Sequence[str]]) -> float:
    if not groups:
        return 0.99
    matches = sum(1 for group in groups if _matches_any(text, group))
    return clamp_open_score(matches / len(groups))


def _score_file_match(file_path: str, task: TaskDefinition) -> float:
    if file_path == task.primary_target_file:
        return 0.99
    if file_path in task.related_files:
        return 0.65
    return 0.01


def _score_line_match(line_number: int, task: TaskDefinition) -> float:
    if line_number == task.primary_target_line:
        return 0.99
    if abs(line_number - task.primary_target_line) <= 5:
        return 0.75
    if abs(line_number - task.primary_target_line) <= 15:
        return 0.45
    return 0.01


def clamp_open_score(score: float | None) -> float:
    numeric_score = 0.0 if score is None else float(score)
    # Ensure the score is strictly between 0 and 1 by using an epsilon.
    # We use 0.01 to leave a clear gap from the boundaries.
    epsilon = 0.01
    bounded_score = max(epsilon, min(1.0 - epsilon, numeric_score))
    return round(bounded_score, 4)


def grade_single_comment(task: TaskDefinition, comment: ReviewCommentRecord) -> GradedComment:
    normalized = _normalize_text(comment.comment_text)
    file_score = _score_file_match(comment.file_path, task)
    line_score = _score_line_match(comment.line_number, task)
    issue_score = _score_keyword_groups(normalized, task.issue_keyword_groups)
    rationale_score = _score_keyword_groups(normalized, task.rationale_keyword_groups)

    score = (
        0.25 * file_score
        + 0.15 * line_score
        + 0.35 * issue_score
        + 0.25 * rationale_score
    )
    return GradedComment(
        score=clamp_open_score(score),
        file_score=clamp_open_score(file_score),
        line_score=clamp_open_score(line_score),
        issue_score=clamp_open_score(issue_score),
        rationale_score=clamp_open_score(rationale_score),
    )


def grade_submission(task: TaskDefinition, submission: ReviewSubmission) -> float:
    best_comment_score = 0.0
    if submission.comments:
        best_comment_score = max(
            grade_single_comment(task, comment).score for comment in submission.comments
        )

    decision_score = 1.0 if submission.decision == task.expected_decision else 0.0

    has_supporting_comment = any(
        comment.file_path in task.related_files and comment.file_path != task.primary_target_file
        for comment in submission.comments
    )
    multi_file_bonus = 0.05 if task.difficulty == Difficulty.HARD and has_supporting_comment else 0.0

    score = (0.75 * best_comment_score) + (0.25 * decision_score) + multi_file_bonus
    return clamp_open_score(score)


def _render_linter_output(command: str, findings: Sequence[LinterFindingSpec]) -> str:
    if not findings:
        return f"$ {command}\nNo issues found."

    lines = [f"$ {command}"]
    for finding in findings:
        lines.append(
            f"{finding.file_path}:{finding.line_number}: {finding.severity} "
            f"{finding.rule_id} {finding.message}"
        )
    lines.append(f"Found {len(findings)} issue(s).")
    return "\n".join(lines)


def classify_linter_command(command: str) -> str | None:
    normalized = _normalize_text(command)
    if any(token in normalized for token in ("rm ", "del ", "remove-item", "shutdown", "format c:", "mkfs")):
        return "destructive"
    if "ruff" in normalized or "flake8" in normalized:
        return "ruff"
    if "bandit" in normalized:
        return "bandit"
    if "semgrep" in normalized or "perf" in normalized:
        return "perf"
    if "py_compile" in normalized or "compileall" in normalized:
        return "compile"
    return None


def run_simulated_linter(task: TaskDefinition, command: str) -> tuple[str, tuple[LinterFindingSpec, ...], bool]:
    profile = classify_linter_command(command)
    if profile == "destructive":
        return (
            f"$ {command}\nBlocked: destructive commands are forbidden in this environment.",
            tuple(),
            False,
        )

    if profile is None:
        return (
            f"$ {command}\nCommand not recognized. Use a linter-style command such as "
            "ruff, bandit, py_compile, or a perf-focused semgrep profile.",
            tuple(),
            False,
        )

    findings = task.linter_findings.get(profile, tuple())
    return _render_linter_output(command, findings), findings, True


TASKS: dict[str, TaskDefinition] = {
    "easy_mutable_default": TaskDefinition(
        task_id="easy_mutable_default",
        difficulty=Difficulty.EASY,
        title="Mutable default argument in digest builder",
        summary=(
            "A helper used by the notification pipeline caches counts in a default "
            "dictionary, causing state to leak across requests."
        ),
        pr_title="Cache digest counts for notification rollups",
        pr_description=(
            "This PR adds a utility for building digest counters. Reviewers should make "
            "sure request-scoped state is isolated."
        ),
        files={
            "services/digest.py": """from collections.abc import Iterable


def build_digest(records: Iterable[str], seen: dict[str, int] = {}) -> dict[str, int]:
    for record in records:
        seen[record] = seen.get(record, 0) + 1
    return seen
""",
            "services/pipeline.py": """from services.digest import build_digest


def collect_rollup(records: list[str]) -> dict[str, int]:
    return build_digest(records)
""",
        },
        expected_decision=ReviewDecision.REQUEST_CHANGES,
        primary_target_file="services/digest.py",
        primary_target_line=4,
        related_files=("services/pipeline.py",),
        linter_findings={
            "ruff": (
                LinterFindingSpec(
                    file_path="services/digest.py",
                    line_number=4,
                    rule_id="B006",
                    severity="warning",
                    message="Do not use mutable data structures for argument defaults.",
                ),
            ),
            "compile": tuple(),
            "bandit": tuple(),
            "perf": tuple(),
        },
        issue_keyword_groups=(
            ("mutable default", "mutable argument default", "default dict", "default {}"),
            ("shared across calls", "persists across calls", "leaks between requests", "reused between calls"),
        ),
        rationale_keyword_groups=(
            ("default to none", "use none", "create the dict inside", "initialize inside the function"),
        ),
        reward_target_files=("services/digest.py",),
    ),
    "medium_quadratic_reporting": TaskDefinition(
        task_id="medium_quadratic_reporting",
        difficulty=Difficulty.MEDIUM,
        title="Quadratic customer total aggregation",
        summary=(
            "The reporting job rescans the full order list for every customer. It works "
            "for small payloads but becomes a bottleneck on enterprise-sized datasets."
        ),
        pr_title="Add account-level revenue rollups to finance reports",
        pr_description=(
            "This change computes a total order amount per customer so finance can add it "
            "to the weekly review packet."
        ),
        files={
            "src/reporting.py": """from collections.abc import Iterable


def map_customer_totals(customer_ids: list[str], orders: list[dict[str, object]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for customer_id in customer_ids:
        total = 0.0
        for order in orders:
            if order["customer_id"] == customer_id:
                total += float(order["amount"])
        totals[customer_id] = total
    return totals
""",
            "src/jobs/generate_weekly_packet.py": """from src.reporting import map_customer_totals


def generate_packet(customer_ids: list[str], orders: list[dict[str, object]]) -> dict[str, float]:
    return map_customer_totals(customer_ids, orders)
""",
        },
        expected_decision=ReviewDecision.REQUEST_CHANGES,
        primary_target_file="src/reporting.py",
        primary_target_line=7,
        related_files=("src/jobs/generate_weekly_packet.py",),
        linter_findings={
            "perf": (
                LinterFindingSpec(
                    file_path="src/reporting.py",
                    line_number=7,
                    rule_id="PERF102",
                    severity="warning",
                    message=(
                        "Nested iteration rescans every order for each customer_id; "
                        "build a dictionary in one pass to avoid O(N^2) behavior."
                    ),
                ),
            ),
            "ruff": tuple(),
            "compile": tuple(),
            "bandit": tuple(),
        },
        issue_keyword_groups=(
            ("o(n^2)", "quadratic", "nested loop", "double loop"),
            ("scan orders for every customer", "rescans the full order list", "for each customer"),
        ),
        rationale_keyword_groups=(
            ("dictionary", "dict", "hash map", "index orders"),
            ("single pass", "linear", "o(n)", "pre-aggregate"),
        ),
        reward_target_files=("src/reporting.py",),
    ),
    "hard_path_traversal": TaskDefinition(
        task_id="hard_path_traversal",
        difficulty=Difficulty.HARD,
        title="Artifact download path traversal",
        summary=(
            "The artifact download flow stitches a request parameter into a filesystem path "
            "without validating that it stays inside the tenant's artifact directory."
        ),
        pr_title="Support direct artifact downloads from the reviewer workspace",
        pr_description=(
            "This PR wires the API route to the artifact service so users can download "
            "build outputs directly from the web app."
        ),
        files={
            "api/routes/artifacts.py": """from services.artifact_service import fetch_artifact_bytes


def download_artifact(request, tenant_id: str, artifact_path: str) -> bytes:
    actor = request.headers.get("X-Actor", "unknown")
    return fetch_artifact_bytes(
        tenant_id=tenant_id,
        artifact_path=artifact_path,
        actor=actor,
    )
""",
            "services/artifact_service.py": """from storage.disk import tenant_artifact_path


def audit_download(actor: str, tenant_id: str, artifact_path: str) -> None:
    print(f"{actor}:{tenant_id}:{artifact_path}")


def fetch_artifact_bytes(tenant_id: str, artifact_path: str, actor: str) -> bytes:
    resolved_path = tenant_artifact_path(tenant_id, artifact_path)
    with open(resolved_path, "rb") as artifact_file:
        payload = artifact_file.read()

    audit_download(actor=actor, tenant_id=tenant_id, artifact_path=artifact_path)
    return payload
""",
            "storage/disk.py": """from pathlib import Path


DATA_ROOT = Path("/srv/artifacts")


def tenant_artifact_path(tenant_id: str, artifact_path: str) -> Path:
    return DATA_ROOT / tenant_id / artifact_path
""",
        },
        expected_decision=ReviewDecision.REQUEST_CHANGES,
        primary_target_file="storage/disk.py",
        primary_target_line=7,
        related_files=("api/routes/artifacts.py", "services/artifact_service.py"),
        linter_findings={
            "bandit": (
                LinterFindingSpec(
                    file_path="storage/disk.py",
                    line_number=7,
                    rule_id="B702",
                    severity="high",
                    message=(
                        "Potential path traversal: user-controlled artifact_path is joined "
                        "into a filesystem path without normalization or boundary checks."
                    ),
                ),
                LinterFindingSpec(
                    file_path="services/artifact_service.py",
                    line_number=9,
                    rule_id="B703",
                    severity="medium",
                    message=(
                        "File is opened using a path derived from request-controlled input. "
                        "Validate the resolved path stays under the tenant directory."
                    ),
                ),
            ),
            "ruff": tuple(),
            "compile": tuple(),
            "perf": tuple(),
        },
        issue_keyword_groups=(
            ("path traversal", "directory traversal"),
            ("artifact_path", "user-controlled path", "request parameter"),
            ("escape the tenant directory", "outside the tenant root", "read arbitrary files", "traverse out"),
        ),
        rationale_keyword_groups=(
            ("resolve", "normalize", "canonicalize"),
            ("commonpath", "relative_to", "check the tenant root", "enforce the base directory"),
        ),
        reward_target_files=("api/routes/artifacts.py", "storage/disk.py"),
        max_steps=16,
    ),
}


def get_task(task_id: str) -> TaskDefinition:
    if task_id not in TASKS:
        # Fallback to a generic task instead of crashing
        return TASKS["easy_mutable_default"]
    return TASKS[task_id]


def list_tasks() -> tuple[TaskDefinition, ...]:
    return tuple(TASKS.values())


def grade_easy(submission: ReviewSubmission) -> float:
    return grade_submission(get_task("easy_mutable_default"), submission)


def grade_medium(submission: ReviewSubmission) -> float:
    return grade_submission(get_task("medium_quadratic_reporting"), submission)


def grade_hard(submission: ReviewSubmission) -> float:
    return grade_submission(get_task("hard_path_traversal"), submission)


def extract_line(content: str, line_number: int) -> str:
    lines = content.splitlines()
    if 1 <= line_number <= len(lines):
        return lines[line_number - 1]
    return ""


def summarize_task_catalog(tasks: Iterable[TaskDefinition] | None = None) -> list[dict[str, str]]:
    catalog = tasks if tasks is not None else list_tasks()
    return [
        {
            "task_id": task.task_id,
            "difficulty": task.difficulty.value,
            "title": task.title,
            "summary": task.summary,
        }
        for task in catalog
    ]
