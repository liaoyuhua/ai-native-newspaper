"""Weekly Deep Dive 的稳定数据契约。

这些对象刻意不依赖具体 LLM 供应商。所有模型输出先经过这里归一化，
避免 prompt 字段漂移直接污染后续研究、写作和发布阶段。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Any


QUALITY_DIMENSIONS = (
    "importance",
    "timeliness",
    "technical_delta",
    "reader_payoff",
    "attractiveness",
    "independent_confirmation",
    "non_redundancy",
)

FEASIBILITY_DIMENSIONS = (
    "primary_sources",
    "mechanism_clarity",
    "comparison_value",
    "thesis_potential",
    "weekly_scope_fit",
)


def _score(value: Any) -> float:
    try:
        return max(1.0, min(5.0, float(value)))
    except (TypeError, ValueError):
        return 1.0


@dataclass
class CandidateQuestion:
    id: str
    question: str
    scope: str
    why_now: str
    technical_delta: str
    reader_takeaway: str
    lens: str
    item_ids: list[str] = field(default_factory=list)
    memory_relation: str = "new"
    related_topic_id: str = ""
    signal_score: float = 0.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any], index: int) -> "CandidateQuestion":
        question = str(raw.get("question") or raw.get("name") or "").strip()
        return cls(
            id=str(raw.get("id") or f"question-{index + 1}"),
            question=question,
            scope=str(raw.get("scope") or raw.get("description") or "").strip(),
            why_now=str(raw.get("why_now") or "").strip(),
            technical_delta=str(raw.get("technical_delta") or "").strip(),
            reader_takeaway=str(raw.get("reader_takeaway") or "").strip(),
            lens=str(raw.get("lens") or "other").strip(),
            item_ids=[str(x) for x in raw.get("item_ids", []) if x],
            memory_relation=str(raw.get("memory_relation") or "new"),
            related_topic_id=str(raw.get("related_topic_id") or ""),
            signal_score=float(raw.get("signal_score") or 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EditorialJudgment:
    perspective: str
    scores: dict[str, float]
    fatal_flaws: list[str]
    reasoning: str

    @classmethod
    def from_dict(cls, perspective: str, raw: dict[str, Any]) -> "EditorialJudgment":
        scores = {name: _score((raw.get("scores") or {}).get(name)) for name in QUALITY_DIMENSIONS}
        return cls(
            perspective=perspective,
            scores=scores,
            fatal_flaws=[str(x) for x in raw.get("fatal_flaws", []) if str(x).strip()],
            reasoning=str(raw.get("reasoning") or "").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeasibilityProbe:
    scores: dict[str, float]
    proposed_thesis: str
    mechanism_targets: list[str]
    missing_evidence: list[str]
    recommendation: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "FeasibilityProbe":
        return cls(
            scores={name: _score((raw.get("scores") or {}).get(name)) for name in FEASIBILITY_DIMENSIONS},
            proposed_thesis=str(raw.get("proposed_thesis") or "").strip(),
            mechanism_targets=[str(x) for x in raw.get("mechanism_targets", []) if str(x).strip()][:2],
            missing_evidence=[str(x) for x in raw.get("missing_evidence", []) if str(x).strip()],
            recommendation=str(raw.get("recommendation") or "reject").lower(),
        )

    @property
    def average(self) -> float:
        return round(sum(self.scores.values()) / len(self.scores), 3)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "average": self.average}


def aggregate_judgments(judgments: list[EditorialJudgment]) -> dict[str, Any]:
    """用中位数减少某一个评委过度兴奋或过度保守造成的偏差。"""
    if not judgments:
        return {"scores": {d: 1.0 for d in QUALITY_DIMENSIONS}, "average": 1.0, "fatal_flaws": ["no_judgments"]}
    scores = {
        dim: round(median(j.scores[dim] for j in judgments), 3)
        for dim in QUALITY_DIMENSIONS
    }
    flaws = sorted({flaw for j in judgments for flaw in j.fatal_flaws})
    return {
        "scores": scores,
        "average": round(sum(scores.values()) / len(scores), 3),
        "fatal_flaws": flaws,
    }


def mechanism_is_publishable(
    *, model_confidence: str, steps_ok: bool, state_ok: bool, evidence_ok: bool, thin_source: bool
) -> bool:
    """纯规则质量闸门，供运行代码和离线测试共同使用。"""
    return bool(
        model_confidence == "high"
        and steps_ok
        and state_ok
        and evidence_ok
        and not thin_source
    )
