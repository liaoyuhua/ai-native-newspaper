"""统一修订循环使用的状态与诊断协议。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class QualityIssue:
    code: str
    dimension: str
    message: str
    repair_type: str
    severity: str = "blocking"
    target_section: str = ""
    deficit: float = 0.0

    @property
    def signature(self) -> str:
        return f"{self.dimension}:{self.code}:{self.target_section}"


@dataclass
class ValidationResult:
    validator: str
    issues: list[QualityIssue] = field(default_factory=list)
    metrics: dict[str, float | int | str] = field(default_factory=dict)

    @property
    def blocking_issues(self) -> list[QualityIssue]:
        return [issue for issue in self.issues if issue.severity == "blocking"]

    def to_dict(self) -> dict:
        return {
            "validator": self.validator,
            "issues": [asdict(issue) for issue in self.issues],
            "metrics": self.metrics,
        }


@dataclass
class ArticleState:
    title: str
    subtitle: str
    thesis: str
    sections: list[dict]
    outline_sections: list[dict] = field(default_factory=list)
    mechanism_cards: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def clone(self) -> "ArticleState":
        return deepcopy(self)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RevisionResult:
    status: str
    state: ArticleState
    validations: list[ValidationResult]
    rounds: list[dict]
    stop_reason: str

