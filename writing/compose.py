"""
短深文成文：大纲 -> 撰写 -> 编辑审稿 -> 事实核查 -> 引用编号。
"""

from __future__ import annotations

import logging
import json
import re
from pathlib import Path

import config
from research.evidence import EvidenceStore
from writing.editorial_review import enforce_title_body_numbers, review_and_revise_section
from writing.factcheck import verify_and_revise_section
from writing.full_draft import generate_full_draft
from writing.outline import generate_outline
from writing.quality_gate import run_article_quality_gate
from writing.reader_review import (
    ensure_reader_context, reader_review_passed, repair_context_for_reader, review_for_cold_reader,
)
from writing.contracts import validate_concept_dependencies

logger = logging.getLogger(__name__)

MARKER_RE = re.compile(r"【EV:([^】]+)】")


def compose_article(
    topic_name: str,
    dossier: dict,
    evidence_store: EvidenceStore,
    *,
    draft_checkpoint: Path | None = None,
    resume_draft: bool = False,
) -> dict:
    publish_mode = dossier.get("publish_mode") or "deep_dive"
    high_cards = list(dossier.get("high_mechanism_cards") or [])
    saved_draft = None
    if resume_draft and draft_checkpoint and draft_checkpoint.exists():
        saved_draft = json.loads(draft_checkpoint.read_text(encoding="utf-8"))
    reader_context = ensure_reader_context(topic_name, dossier)
    outline = saved_draft["outline"] if saved_draft else generate_outline(topic_name, dossier)
    dependency_issues = validate_concept_dependencies(
        outline.get("sections", []), reader_context
    )
    if dependency_issues:
        raise ValueError("大纲概念依赖不完整: " + "；".join(dependency_issues))
    article_title = outline.get("title", topic_name)
    thesis = outline.get("thesis", "")
    hook = outline.get("hook", "")
    beginner_context = dossier.get("beginner_context") or ""
    core_intuition = dossier.get("core_intuition") or ""
    running_example = dossier.get("running_example") or ""

    logger.info("撰写模式: %s | high 卡 %d 张", publish_mode, len(high_cards))
    if thesis:
        logger.info("全文主结论: %s", thesis)

    draft_model = (saved_draft or {}).get("draft_model", config.MODEL_WRITING)
    section_texts = list((saved_draft or {}).get("sections", []))
    if not saved_draft:
        raw_sections, draft_model = generate_full_draft(topic_name, outline, dossier, evidence_store)
        section_texts = []
    for section, raw_section in ([] if saved_draft else zip(outline.get("sections", []), raw_sections, strict=True)):
        card = None
        if (section.get("role") or "").lower() == "mechanism":
            idx = section.get("card_index")
            if isinstance(idx, int) and 0 <= idx < len(high_cards):
                card = high_cards[idx]

        logger.info("撰写小节: %s (role=%s)", section.get("heading"), section.get("role", ""))
        edited = review_and_revise_section(
            raw_section["text"],
            section,
            thesis=thesis,
            mechanism_card=card,
        )
        checked = verify_and_revise_section(edited, evidence_store)
        section_texts.append(
            {
                "heading": raw_section.get("heading") or section["heading"],
                "text": checked,
                "role": section.get("role", ""),
                "card_index": section.get("card_index"),
                "method_role": section.get("method_role"),
            }
        )

    if draft_checkpoint and not saved_draft:
        draft_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        draft_checkpoint.write_text(
            json.dumps({"outline": outline, "draft_model": draft_model, "sections": section_texts}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 冷启动审稿只看成稿，不允许依据研究档案替正文脑补。失败后只定点修复背景节。
    reader_reviews = []
    for round_index in range(2):
        reader_review = review_for_cold_reader(article_title, section_texts)
        reader_reviews.append(reader_review)
        if reader_review_passed(reader_review):
            break
        if round_index == 1:
            from writing.quality_gate import ArticleQualityError
            raise ArticleQualityError("reader_background_failed", {"reader_review": reader_review, "rounds": reader_reviews})
        context = next((s for s in section_texts if s.get("role") == "context"), None)
        if context is None:
            from writing.quality_gate import ArticleQualityError
            raise ArticleQualityError("reader_context_section_missing", {"reader_review": reader_review})
        context["text"] = repair_context_for_reader(
            context.get("text", ""), reader_review, reader_context
        )
        context["text"] = verify_and_revise_section(context["text"], evidence_store)

    # 在把内部 evidence marker 转成脚注编号之前做整篇 claim 审计，
    # 否则终审无法把正文论断映射回证据片段。
    quality = run_article_quality_gate(
        article_title, thesis, section_texts, evidence_store,
        reader_review=reader_reviews[-1], mechanism_cards=high_cards,
    )

    citation_order: list[str] = []
    citation_number: dict[str, int] = {}

    def _renumber(match: re.Match) -> str:
        eids = [e.strip() for e in match.group(1).split(",") if e.strip()]
        numbers = []
        for eid in eids:
            if eid not in citation_number:
                citation_number[eid] = len(citation_order) + 1
                citation_order.append(eid)
            numbers.append(citation_number[eid])
        return ",".join(f"[{n}]" for n in sorted(set(numbers)))

    for section in section_texts:
        section["text"] = MARKER_RE.sub(_renumber, section["text"])

    body_join = "\n".join(s["text"] for s in section_texts)
    allowed_claims = []
    for card in high_cards:
        for ev in card.get("evidence") or []:
            if ev.get("claim"):
                allowed_claims.append(str(ev["claim"]))
    article_title = enforce_title_body_numbers(article_title, body_join, allowed_claims)

    bibliography = []
    for eid in citation_order:
        ev = evidence_store.get(eid)
        if ev:
            bibliography.append(
                {
                    "number": citation_number[eid],
                    "title": ev.title,
                    "url": ev.url,
                    "source_type": ev.source_type,
                }
            )

    featured_works = _build_featured_works(dossier.get("key_works", []), citation_number, evidence_store)
    open_questions = list(dossier.get("open_questions", []) or [])[:3]

    return {
        "title": article_title,
        "subtitle": outline.get("subtitle", ""),
        "topic_name": topic_name,
        "publish_mode": publish_mode,
        "thesis": thesis,
        "hook": hook,
        "beginner_context": beginner_context,
        "core_intuition": core_intuition,
        "running_example": running_example,
        "mechanism_cards": dossier.get("mechanism_cards") or [],
        "high_mechanism_cards": high_cards,
        "sections": section_texts,
        "bibliography": bibliography,
        "featured_works": featured_works[:5],
        "open_questions": open_questions,
        "quality_report": quality.to_dict(),
        "claim_ledger": evidence_store.claim_list(),
        "generation": {
            "outline_model": outline.get("_model_used", config.MODEL_WRITING),
            "frame_revision_model": outline.get("_frame_revision_model", ""),
            "draft_model": draft_model,
            "fallback_used": draft_model != config.MODEL_WRITING or outline.get("_model_used") != config.MODEL_WRITING,
        },
    }


def _build_featured_works(key_works: list[dict], citation_number: dict, evidence_store: EvidenceStore) -> list[dict]:
    out = []
    for w in key_works:
        eid = w.get("evidence_id", "")
        ev = evidence_store.get(eid)
        out.append(
            {
                "title": w.get("title") or (ev.title if ev else ""),
                "url": w.get("url") or (ev.url if ev else ""),
                "why_relevant": w.get("why_relevant", ""),
                "citation_number": citation_number.get(eid),
            }
        )
    out.sort(key=lambda w: (w["citation_number"] is None, w["citation_number"] or 0))
    return out
