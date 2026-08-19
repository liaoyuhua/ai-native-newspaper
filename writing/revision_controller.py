"""有预算、可回滚的文章修订控制器。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol

from writing.revision_state import ArticleState, QualityIssue, RevisionResult, ValidationResult

logger = logging.getLogger(__name__)


class Validator(Protocol):
    def validate(self, state: ArticleState) -> ValidationResult: ...


class Repairer(Protocol):
    def repair(self, state: ArticleState, issues: list[QualityIssue]) -> ArticleState: ...


class RevisionController:
    def __init__(
        self,
        validators: list[Validator],
        repairers: dict[str, Repairer],
        *,
        max_rounds: int = 3,
        max_repairs_per_round: int = 2,
        snapshot_dir: Path | None = None,
    ) -> None:
        self.validators = validators
        self.repairers = repairers
        self.max_rounds = max(0, max_rounds)
        self.max_repairs_per_round = max(1, max_repairs_per_round)
        self.snapshot_dir = snapshot_dir

    def run(self, initial_state: ArticleState) -> RevisionResult:
        current = initial_state.clone()
        validations = self._validate(current)
        best_state, best_validations = current.clone(), validations
        best_score = self._score(validations)
        rounds: list[dict] = []
        if best_score[0] == 0:
            return RevisionResult("passed", current, validations, rounds, "already_passed")

        stop_reason = "round_budget_exhausted"
        for round_number in range(1, self.max_rounds + 1):
            issues = self._blocking(validations)
            selected = self._select_repairs(issues)
            if not selected:
                stop_reason = "no_registered_repairer"
                break

            candidate = current.clone()
            applied: list[str] = []
            for repair_type, repair_issues in selected:
                candidate = self.repairers[repair_type].repair(candidate, repair_issues)
                applied.append(repair_type)

            candidate_validations = self._validate(candidate)
            candidate_score = self._score(candidate_validations)
            accepted = candidate_score < best_score
            record = {
                "round": round_number,
                "input_score": list(self._score(validations)),
                "candidate_score": list(candidate_score),
                "accepted": accepted,
                "repairs": applied,
                "issues_before": [issue.signature for issue in issues],
                "validations_after": [result.to_dict() for result in candidate_validations],
                "state": candidate.to_dict(),
            }
            rounds.append(record)
            self._write_snapshot(round_number, record)

            if accepted:
                current, validations = candidate, candidate_validations
                best_state, best_validations, best_score = candidate.clone(), candidate_validations, candidate_score
                logger.info("修订控制器第 %d 轮改善质量: score=%s", round_number, candidate_score)
                if candidate_score[0] == 0:
                    return RevisionResult("passed", best_state, best_validations, rounds, "all_validators_passed")
            else:
                stop_reason = "no_measurable_improvement"
                logger.warning("修订控制器第 %d 轮未改善，回滚到最佳版本", round_number)
                break

        return RevisionResult("blocked", best_state, best_validations, rounds, stop_reason)

    def _validate(self, state: ArticleState) -> list[ValidationResult]:
        return [validator.validate(state) for validator in self.validators]

    @staticmethod
    def _blocking(results: list[ValidationResult]) -> list[QualityIssue]:
        return [issue for result in results for issue in result.blocking_issues]

    @staticmethod
    def _score(results: list[ValidationResult]) -> tuple[int, float]:
        issues = RevisionController._blocking(results)
        return len(issues), round(sum(max(0.0, issue.deficit) for issue in issues), 6)

    def _select_repairs(self, issues: list[QualityIssue]) -> list[tuple[str, list[QualityIssue]]]:
        grouped: dict[str, list[QualityIssue]] = {}
        for issue in issues:
            if issue.repair_type in self.repairers:
                grouped.setdefault(issue.repair_type, []).append(issue)
        return list(grouped.items())[: self.max_repairs_per_round]

    def _write_snapshot(self, round_number: int, payload: dict) -> None:
        if self.snapshot_dir is None:
            return
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = self.snapshot_dir / f"round-{round_number:02d}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

