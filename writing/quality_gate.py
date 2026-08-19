"""文章级 claim 审计与独立终审。

与旧的逐节检查不同，这里看完整文章，并采用 fail-closed：无法证明通过就不发布。
"""

from __future__ import annotations

import json
import re
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import config
from processing.llm_client import chat_json, chat_json_with_fallback, chat_text
from research.evidence import EvidenceStore
from writing.reader_review import reader_review_passed
from writing.review_checkpoint import write_review_checkpoint

logger = logging.getLogger(__name__)

MARKER_RE = re.compile(r"【EV:([^】]+)】")
FINAL_SCORE_DIMENSIONS = (
    "thesis_focus",
    "structural_coherence",
    "mechanism_depth",
    "editorial_voice",
    "weekly_scope",
    "evidence_honesty",
)


class ArticleQualityError(RuntimeError):
    def __init__(self, reason: str, report: dict):
        super().__init__(reason)
        self.reason = reason
        self.report = report


CLAIM_AUDIT_PROMPT = """\
你是技术文章的事实边界审计员。只找出文章中可由外部资料核实的陈述，不能只检查带引用的句子。
分类：fact（机制/架构/时间等事实）、reported_result（实验或性能结果）、inference（作者从材料推出的解释）、
author_judgment（价值判断）。

规则：
- fact/reported_result 必须带有效 evidence_id，且 excerpt 真正支持完整陈述；否则 unsupported 或 partial。
- inference 可以没有直接证据。作者可以自然地综合、解释和命名一个模式，不要求每句话重复“可能/可以推测”。
- inference 只有在使用“证明、必然、唯一原因、完全取代”等强断言，虚构因果，或把有限实验外推到所有模型时，
  才需要作为潜在问题列出；普通的“这说明、关键区别是、这些结果的含义是”属于正常作者分析。
- author_judgment 不要求引用。
- 不要把段落总结、行文连接句、类比、文章结构判断或没有新增外部事实的概念框架枚举为 claim。
- 目标是检查“事实可靠、推断诚实”，不是要求文章每句话都能在来源中找到逐字对应。
- 注意表格、标题、图注、伪代码中的陈述。

只返回 JSON：{"claims":[{"text":"文章中的准确原句或紧凑转述","kind":"fact|reported_result|inference|author_judgment",
"evidence_ids":["ev_x"],"support":"supported|partial|unsupported","reason":"..."}]}
"""

FINAL_REVIEW_PROMPT = """\
你是 Weekly Deep Dive 的独立终审。目标是每周 8–15 分钟技术文章，不要求完整领域综述。
检查整篇而非局部文采。按 1–5 分评价：
- thesis_focus：是否只回答一个中心问题；
- structural_coherence：段落是否推进而非重复；
- mechanism_depth：读者能否复述 1–2 个关键机制；
- editorial_voice：是否有克制的人类判断，而非模板化金句和强行比喻；
- weekly_scope：是否达到周刊深度且没有摊成综述；
- evidence_honesty：数字、方法和实验范围是否可靠；读者能否分辨来源事实与作者分析。
  不要求正常的作者综合判断逐句带免责声明，避免把文章改成充满“可能/可以推测”的审计报告。

fatal_issues 包含任何必须阻止发布的问题：thesis_missing、mechanism_unclear、major_repetition、
forced_connection、unsupported_core_claim、ai_template_prose、scope_failure。
blocking_revisions 必须列出虽可修但在修复前不能发布的问题，category 只能是：
central_thesis_overclaim、primary_method_underexplained、evidence_missing_for_core_claim、
mechanism_result_disconnect、metaphor_density。普通措辞偏好只放 revision_priorities。
按绝对标准评估，不要因为文章已经写完而倾向通过。
只返回 JSON：{"scores":{...},"fatal_issues":[],"blocking_revisions":[{"category":"...","instruction":"..."}],
"strengths":[],"revision_priorities":[],"overall_pass":false}
"""

CLAIM_REPAIR_PROMPT = """\
你是技术文章的事实编辑。根据文章级 claim 审计反馈，只修复当前小节中被点名的问题。
- partial：收窄或软化到证据实际支持的范围。
- unsupported：若下方证据明确支持修正后的表述，可改写并在句末加对应【EV:evidence_id】；否则删除。
- overstated_inference：只收窄“正是、证明、只剩、必然、彻底、唯一原因”等强断言。
  普通作者综合不需要每句话添加“可能/可以理解为”，不要制造免责声明堆积。
- 保留已有有效【EV:evidence_id】。只能使用下方真实提供的 evidence_id，禁止虚构引用。
- 不要新增证据没有的数字、方法步骤或论文结论。
- 不重写无关段落，不改变小节职责，不增加篇幅。
直接返回修复后的完整 Markdown 小节，不要解释。
"""


@dataclass
class QualityGateResult:
    claim_audit: dict
    final_review: dict
    reader_review: dict | None = None

    def to_dict(self) -> dict:
        return {"claim_audit": self.claim_audit, "final_review": self.final_review,
                "reader_review": self.reader_review or {}}


def run_article_quality_gate(
    article_title: str,
    thesis: str,
    sections: list[dict],
    evidence_store: EvidenceStore,
    reader_review: dict | None = None,
    mechanism_cards: list[dict] | None = None,
    checkpoint: dict | None = None,
    checkpoint_path: Path | None = None,
) -> QualityGateResult:
    checkpoint = checkpoint if checkpoint is not None else {}
    if not reader_review or not reader_review_passed(reader_review):
        raise ArticleQualityError("reader_background_not_proven", {"reader_review": reader_review or {}})
    contract = _validate_article_contract(article_title, thesis, sections, mechanism_cards or [])
    if not contract["pass"]:
        raise ArticleQualityError("article_depth_contract_failed", {"depth_contract": contract})
    body = "\n\n".join(f"## {s.get('heading', '')}\n{s.get('text', '')}" for s in sections)
    audit_rounds = list(checkpoint.get("claim_audit_rounds") or [])
    audit = dict(checkpoint.get("claim_audit") or {})
    # 如果常规预算最后只剩推断措辞问题，用确定性降调修复并允许一次受限重验。
    if (
        not checkpoint.get("claim_audit_complete")
        and len(audit_rounds) >= config.ARTICLE_CLAIM_AUDIT_ROUNDS
        and _only_inference_language_issues(audit.get("blocking_claims") or [])
        and not checkpoint.get("inference_language_recheck")
    ):
        changed = _soften_inference_language(sections, audit.get("blocking_claims") or [])
        if changed:
            checkpoint.update({
                "sections": sections,
                "inference_language_recheck": True,
                "claim_audit_complete": False,
            })
            write_review_checkpoint(checkpoint_path, checkpoint)
            logger.info("常规审计预算后仅剩推断措辞问题，确定性降调 %d 处并追加一次重验", changed)
    if (
        not checkpoint.get("claim_audit_complete")
        and len(audit_rounds) >= config.ARTICLE_CLAIM_AUDIT_ROUNDS
        and 0 < len(audit.get("blocking_claims") or []) <= 2
        and not checkpoint.get("terminal_claim_recheck")
        and not checkpoint.get("inference_language_recheck")
    ):
        _repair_blocking_sections(sections, audit.get("blocking_claims") or [], evidence_store)
        checkpoint.update({
            "sections": sections,
            "terminal_claim_recheck": True,
            "claim_audit_complete": False,
        })
        write_review_checkpoint(checkpoint_path, checkpoint)
        logger.info("常规审计预算后仅剩 %d 条事实问题，修复后追加一次收尾重验", len(audit.get("blocking_claims") or []))
    extra_recheck = checkpoint.get("inference_language_recheck") or checkpoint.get("terminal_claim_recheck")
    audit_limit = config.ARTICLE_CLAIM_AUDIT_ROUNDS + (1 if extra_recheck else 0)
    audit_start = audit_limit if checkpoint.get("claim_audit_complete") else len(audit_rounds)
    for round_index in range(audit_start, audit_limit):
        body = "\n\n".join(f"## {s.get('heading', '')}\n{s.get('text', '')}" for s in sections)
        evidence_store.clear_claims()
        audit = _audit_claims_by_section(
            sections, evidence_store, round_index + 1, checkpoint, checkpoint_path,
        )
        blocking = _register_and_find_blocking_claims(audit, evidence_store)
        if not audit.get("claims"):
            blocking.append({"reason": "claim_auditor_returned_no_claims"})
        audit["blocking_claims"] = blocking
        audit["pass"] = not blocking
        audit["round"] = round_index + 1
        audit_rounds.append(audit)
        checkpoint.update({
            "claim_audit": audit,
            "claim_audit_rounds": audit_rounds,
            "sections": sections,
        })
        checkpoint.pop("claim_audit_partial", None)
        write_review_checkpoint(checkpoint_path, checkpoint)
        if not blocking:
            checkpoint["claim_audit_complete"] = True
            write_review_checkpoint(checkpoint_path, checkpoint)
            break
        if round_index + 1 >= audit_limit:
            raise ArticleQualityError("claim_audit_failed", {"claim_audit": audit, "claim_audit_rounds": audit_rounds})
        _repair_blocking_sections(sections, blocking, evidence_store)
        checkpoint.update({"sections": sections, "claim_audit_complete": False})
        write_review_checkpoint(checkpoint_path, checkpoint)

    # 从 checkpoint 恢复通过的审计时，重建 claim ledger。
    if checkpoint.get("claim_audit_complete") and audit:
        evidence_store.clear_claims()
        restored_blocking = _register_and_find_blocking_claims(audit, evidence_store)
        if restored_blocking:
            raise ArticleQualityError("claim_audit_checkpoint_invalid", {"blocking_claims": restored_blocking})

    body = "\n\n".join(f"## {s.get('heading', '')}\n{s.get('text', '')}" for s in sections)
    review = dict(checkpoint.get("final_review") or {})
    if not review:
        review = _final_review(article_title, thesis, body)
        checkpoint["final_review"] = review
        write_review_checkpoint(checkpoint_path, checkpoint)
    scores = review.get("scores") or {}
    normalized = {key: _bounded_score(scores.get(key)) for key in FINAL_SCORE_DIMENSIONS}
    review["scores"] = normalized
    review["average"] = round(sum(normalized.values()) / len(normalized), 3) if normalized else 0.0
    fatal = [str(x) for x in review.get("fatal_issues", []) if str(x).strip()]
    blocking_revisions = [x for x in review.get("blocking_revisions", []) if x]
    review["blocking_revisions"] = blocking_revisions
    passed = (
        review.get("overall_pass") is True and not fatal and not blocking_revisions
        and review["average"] >= config.ARTICLE_MIN_FINAL_SCORE
    )
    review["pass"] = passed
    if not passed:
        raise ArticleQualityError("editorial_final_review_failed", {"claim_audit": audit, "final_review": review})
    audit["audit_rounds"] = len(audit_rounds)
    checkpoint["quality_complete"] = True
    checkpoint["sections"] = sections
    write_review_checkpoint(checkpoint_path, checkpoint)
    return QualityGateResult(audit, review, reader_review)


def _audit_claims(body: str, evidence_store: EvidenceStore) -> dict:
    referenced = []
    for marker in MARKER_RE.findall(body):
        referenced.extend(x.strip() for x in marker.split(",") if x.strip())
    evidence = []
    for eid in dict.fromkeys(referenced):
        ev = evidence_store.get(eid)
        if ev:
            evidence.append({"id": eid, "title": ev.title, "url": ev.url, "excerpt": ev.excerpt[:2400]})
    result, _ = chat_json_with_fallback(
        model=config.MODEL_FACTCHECK,
        fallback_model=config.MODEL_FACTCHECK_FALLBACK,
        system_prompt=CLAIM_AUDIT_PROMPT,
        user_prompt="文章：\n" + body + "\n\n可用证据：\n" + json.dumps(evidence, ensure_ascii=False),
        temperature=0.0,
    )
    return result


def _audit_claims_by_section(
    sections: list[dict],
    evidence_store: EvidenceStore,
    round_number: int,
    checkpoint: dict,
    checkpoint_path: Path | None,
) -> dict:
    """逐节审计并逐节落盘，限制输出规模且支持失败后续跑。"""
    digest = hashlib.sha256(
        json.dumps(sections, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    partial = checkpoint.get("claim_audit_partial") or {}
    if partial.get("round") != round_number or partial.get("sections_digest") != digest:
        partial = {"round": round_number, "sections_digest": digest, "results": {}}
    results = partial.setdefault("results", {})
    for index, section in enumerate(sections):
        key = str(index)
        if key in results:
            continue
        logger.info(
            "文章 claim audit round=%d section=%d/%d heading=%s",
            round_number, index + 1, len(sections), section.get("heading", ""),
        )
        body = f"## {section.get('heading', '')}\n{section.get('text', '')}"
        results[key] = _audit_claims(body, evidence_store)
        checkpoint["claim_audit_partial"] = partial
        write_review_checkpoint(checkpoint_path, checkpoint)
    claims = []
    for index in range(len(sections)):
        claims.extend((results.get(str(index)) or {}).get("claims") or [])
    return {"claims": claims, "section_audits": results}


def _register_and_find_blocking_claims(audit: dict, evidence_store: EvidenceStore) -> list[dict]:
    blocking = []
    for raw in audit.get("claims", []):
        text = str(raw.get("text") or "").strip()
        kind = str(raw.get("kind") or "fact")
        if kind not in {"fact", "reported_result", "inference", "author_judgment"}:
            kind = "fact"
        eids = [str(x) for x in raw.get("evidence_ids", []) if evidence_store.get(str(x))]
        support = str(raw.get("support") or "unsupported")
        evidence_store.add_claim(text, kind, eids, support=support, notes=str(raw.get("reason") or ""))
        if kind in {"fact", "reported_result"} and (not eids or support != "supported"):
            blocking.append({"text": text, "kind": kind, "evidence_ids": eids,
                             "support": support, "reason": raw.get("reason", "")})
        if kind == "inference":
            issue = _inference_language_issue(text, eids, support)
            if issue:
                blocking.append({"text": text, "kind": kind, "evidence_ids": eids,
                                 "support": issue, "reason": "推断语气强于证据"})
    return blocking


def _only_inference_language_issues(blocking: list[dict]) -> bool:
    return bool(blocking) and all(
        problem.get("kind") == "inference"
        and problem.get("support") in {"unmarked_inference", "overstated_inference"}
        for problem in blocking
    )


def _soften_inference_language(sections: list[dict], blocking: list[dict]) -> int:
    """只处理审计已定位的原句，不新增事实或改动引用。"""
    changed = 0
    for problem in blocking:
        claim = str(problem.get("text") or "").strip()
        if not claim:
            continue
        softened = claim.replace("证明了", "提示了").replace("证明", "提示")
        if not _HEDGE_RE.search(softened):
            softened = "一种可能的解释是，" + softened
        for section in sections:
            text = str(section.get("text") or "")
            if claim in text:
                section["text"] = text.replace(claim, softened, 1)
                changed += 1
                break
    return changed


def _final_review(title: str, thesis: str, body: str) -> dict:
    return chat_json(
        model=config.MODEL_EDITORIAL,
        system_prompt=FINAL_REVIEW_PROMPT,
        user_prompt=f"标题：{title}\n主结论：{thesis}\n\n全文：\n{body}",
        temperature=0.1,
    )


def _repair_blocking_sections(
    sections: list[dict], blocking: list[dict], evidence_store: EvidenceStore
) -> None:
    assignments: dict[int, list[dict]] = {}
    pending = []
    for problem in blocking:
        claim = str(problem.get("text") or "")
        exact = [i for i, section in enumerate(sections) if claim and (claim in section.get("text", "") or claim[:24] in section.get("text", ""))]
        candidates = exact
        if not candidates:
            eids = [str(x) for x in problem.get("evidence_ids", []) if x]
            candidates = [
                i for i, section in enumerate(sections)
                if any(f"【EV:{eid}" in str(section.get("text") or "") for eid in eids)
            ]
        if not candidates and claim:
            scored = [(_keyword_overlap(claim, str(section.get("text") or "")), i) for i, section in enumerate(sections)]
            best_score, best_index = max(scored, default=(0, -1))
            if best_score > 0:
                candidates = [best_index]
        if not candidates:
            pending.append(problem)
            continue
        # 同一证据可能出现在多节；优先选择与 claim 共享最多显著词的小节。
        target = max(candidates, key=lambda i: _keyword_overlap(claim, str(sections[i].get("text") or "")))
        assignments.setdefault(target, []).append(problem)

    for index, relevant in assignments.items():
        section = sections[index]
        text = str(section.get("text") or "")
        candidate_eids = []
        for marker in MARKER_RE.findall(text):
            candidate_eids.extend(x.strip() for x in marker.split(",") if x.strip())
        for problem in relevant:
            claim_lower = str(problem.get("text") or "").lower()
            for ev in evidence_store.all():
                title_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", ev.title.lower())
                if any(token in claim_lower for token in title_tokens):
                    candidate_eids.append(ev.id)
        evidence = []
        for eid in dict.fromkeys(candidate_eids):
            ev = evidence_store.get(eid)
            if ev:
                evidence.append({"id": eid, "title": ev.title, "excerpt": ev.excerpt[:2200]})
            if len(evidence) >= 4:
                break
        candidate_text = chat_text(
            model=config.MODEL_EDITORIAL,
            system_prompt=CLAIM_REPAIR_PROMPT,
            user_prompt=(
                "审计问题：\n" + json.dumps(relevant, ensure_ascii=False, indent=2)
                + "\n\n可用于修复的证据：\n" + json.dumps(evidence, ensure_ascii=False, indent=2)
                + "\n\n当前小节：\n" + text
            ),
            temperature=0.2,
        )
        original_chars = plain_char_count(text)
        candidate_chars = plain_char_count(candidate_text)
        if original_chars and candidate_chars < original_chars * 0.75:
            logger.warning(
                "claim repair 导致章节过度缩水，拒绝候选并保留原文: heading=%s %d->%d",
                section.get("heading", ""), original_chars, candidate_chars,
            )
        else:
            section["text"] = candidate_text
    if pending:
        raise ArticleQualityError("claim_repair_target_not_found", {"blocking_claims": pending})


def _keyword_overlap(claim: str, text: str) -> int:
    latin = set(re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", claim.lower()))
    chinese = {claim[i : i + 2] for i in range(max(0, len(claim) - 1)) if "\u4e00" <= claim[i] <= "\u9fff"}
    return sum(4 for word in latin if word in text.lower()) + sum(1 for pair in chinese if pair in text)


def _bounded_score(value) -> float:
    try:
        return max(1.0, min(5.0, float(value)))
    except (TypeError, ValueError):
        return 1.0


_HEDGE_RE = re.compile(r"可能|可以推测|可推测|可以理解为|更合理的理解|一种解释|似乎|或许|未必|倾向于|更多承担|这意味着|从设计意图看|大致")
_OVERCLAIM_RE = re.compile(r"证明了?|只剩|必然|正是|彻底|完全取代|根治|无疑")


def _inference_language_issue(text: str, evidence_ids: list[str], support: str) -> str:
    if _OVERCLAIM_RE.search(text) and not (evidence_ids and support == "supported"):
        return "overstated_inference"
    return ""


def plain_char_count(text: str) -> int:
    text = MARKER_RE.sub("", str(text or ""))
    text = re.sub(r"[`#>*_|\[\](){}\-]", "", text)
    return len(re.sub(r"\s+", "", text))


def _validate_article_contract(
    title: str, thesis: str, sections: list[dict], mechanism_cards: list[dict]
) -> dict:
    body_chars = sum(plain_char_count(s.get("text", "")) for s in sections)
    mechanism_sections = [s for s in sections if s.get("role") == "mechanism"]
    mechanism_chars = sum(plain_char_count(s.get("text", "")) for s in mechanism_sections)
    primary_indexes = {
        i for i, card in enumerate(mechanism_cards) if card.get("method_role") == "primary_subject"
    }
    primary_sections = [s for s in mechanism_sections if s.get("card_index") in primary_indexes]
    primary_chars = sum(plain_char_count(s.get("text", "")) for s in primary_sections)
    ratio = mechanism_chars / body_chars if body_chars else 0.0
    issues = []
    if body_chars < config.ARTICLE_MIN_BODY_CHARS:
        issues.append(f"正文仅 {body_chars} 字，低于 {config.ARTICLE_MIN_BODY_CHARS} 字深度下限")
    if ratio < config.ARTICLE_MIN_MECHANISM_RATIO:
        issues.append(f"机制内容占比 {ratio:.1%}，低于 {config.ARTICLE_MIN_MECHANISM_RATIO:.0%}")
    if not primary_indexes or not primary_sections:
        issues.append("没有直接解释本期主角方法的机制节")
    elif primary_chars < config.ARTICLE_MIN_PRIMARY_MECHANISM_CHARS:
        issues.append(
            f"主角机制仅 {primary_chars} 字，低于 {config.ARTICLE_MIN_PRIMARY_MECHANISM_CHARS} 字"
        )
    if _OVERCLAIM_RE.search(str(title) + "\n" + str(thesis)):
        issues.append("标题或 thesis 含未经限定的绝对结论")
    return {
        "pass": not issues,
        "issues": issues,
        "body_chars": body_chars,
        "mechanism_chars": mechanism_chars,
        "mechanism_ratio": round(ratio, 3),
        "primary_mechanism_chars": primary_chars,
    }
