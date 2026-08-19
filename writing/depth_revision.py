"""文章深度 Validator 与基于证据的主角机制修复器。"""

from __future__ import annotations

import config
from research.evidence import EvidenceStore
from writing.editorial_review import review_and_revise_section
from writing.factcheck import verify_and_revise_section
from writing.full_draft import expand_primary_after_factcheck
from writing.quality_gate import plain_char_count
from writing.revision_state import ArticleState, QualityIssue, ValidationResult


class ArticleDepthValidator:
    name = "article_depth"

    def validate(self, state: ArticleState) -> ValidationResult:
        primary_indexes = {
            index for index, card in enumerate(state.mechanism_cards)
            if card.get("method_role") == "primary_subject"
        }
        counts = [plain_char_count(section.get("text", "")) for section in state.sections]
        body_chars = sum(counts)
        mechanism_chars = sum(
            count for count, section in zip(counts, state.sections)
            if section.get("role") == "mechanism"
        )
        primary_chars = sum(
            count for count, section in zip(counts, state.sections)
            if section.get("role") == "mechanism"
            and section.get("card_index") in primary_indexes
        )
        ratio = mechanism_chars / body_chars if body_chars else 0.0
        issues: list[QualityIssue] = []
        repair_type = "expand_primary_mechanism"
        if body_chars < config.ARTICLE_MIN_BODY_CHARS:
            issues.append(QualityIssue(
                "body_too_short", "depth", f"正文仅 {body_chars} 字", repair_type,
                deficit=(config.ARTICLE_MIN_BODY_CHARS - body_chars) / config.ARTICLE_MIN_BODY_CHARS,
            ))
        if ratio < config.ARTICLE_MIN_MECHANISM_RATIO:
            issues.append(QualityIssue(
                "mechanism_ratio_low", "depth", f"机制占比仅 {ratio:.1%}", repair_type,
                deficit=(config.ARTICLE_MIN_MECHANISM_RATIO - ratio) / config.ARTICLE_MIN_MECHANISM_RATIO,
            ))
        if primary_chars < config.ARTICLE_MIN_PRIMARY_MECHANISM_CHARS:
            issues.append(QualityIssue(
                "primary_mechanism_thin", "depth", f"主角机制仅 {primary_chars} 字", repair_type,
                target_section="primary_subject",
                deficit=(config.ARTICLE_MIN_PRIMARY_MECHANISM_CHARS - primary_chars)
                / config.ARTICLE_MIN_PRIMARY_MECHANISM_CHARS,
            ))
        return ValidationResult(self.name, issues, {
            "body_chars": body_chars,
            "mechanism_chars": mechanism_chars,
            "mechanism_ratio": round(ratio, 4),
            "primary_mechanism_chars": primary_chars,
        })


class PrimaryMechanismRepairer:
    def __init__(self, evidence_store: EvidenceStore) -> None:
        self.evidence_store = evidence_store

    def repair(self, state: ArticleState, issues: list[QualityIssue]) -> ArticleState:
        candidate = state.clone()
        primary_index = next(
            (i for i, card in enumerate(candidate.mechanism_cards)
             if card.get("method_role") == "primary_subject"),
            None,
        )
        target = next((section for section in candidate.sections
                       if section.get("role") == "mechanism"
                       and section.get("card_index") == primary_index), None)
        outline_section = next((section for section in candidate.outline_sections
                                if section.get("role") == "mechanism"
                                and section.get("card_index") == primary_index), None)
        if primary_index is None or target is None or outline_section is None:
            return candidate
        messages = [issue.message for issue in issues]
        expanded, model_used = expand_primary_after_factcheck(
            target.get("text", ""), candidate.mechanism_cards[primary_index],
            self.evidence_store, messages,
        )
        expanded = review_and_revise_section(
            expanded, outline_section, thesis=candidate.thesis,
            mechanism_card=candidate.mechanism_cards[primary_index],
        )
        target["text"] = verify_and_revise_section(expanded, self.evidence_store)
        candidate.metadata["draft_model"] = model_used
        return candidate
