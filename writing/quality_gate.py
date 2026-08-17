"""文章级 claim 审计与独立终审。

与旧的逐节检查不同，这里看完整文章，并采用 fail-closed：无法证明通过就不发布。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import config
from processing.llm_client import chat_json, chat_text
from research.evidence import EvidenceStore
from writing.reader_review import reader_review_passed

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
你是技术文章的 claim 审计员。逐项找出文章中所有可由外部资料核实的陈述，不能只检查带引用的句子。
分类：fact（机制/架构/时间等事实）、reported_result（实验或性能结果）、inference（作者从材料推出的解释）、
author_judgment（价值判断）。

规则：
- fact/reported_result 必须带有效 evidence_id，且 excerpt 真正支持完整陈述；否则 unsupported 或 partial。
- inference 可以没有直接证据，但正文必须明确表现为推断，不能冒充来源结论。
- author_judgment 不要求引用。
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
- evidence_honesty：事实、推断与判断是否诚实分开。

fatal_issues 包含任何必须阻止发布的问题：thesis_missing、mechanism_unclear、major_repetition、
forced_connection、unsupported_core_claim、ai_template_prose、scope_failure。
按绝对标准评估，不要因为文章已经写完而倾向通过。
只返回 JSON：{"scores":{...},"fatal_issues":[],"strengths":[],"revision_priorities":[],"overall_pass":false}
"""

CLAIM_REPAIR_PROMPT = """\
你是技术文章的事实编辑。根据文章级 claim 审计反馈，只修复当前小节中被点名的问题。
- partial：收窄或软化到证据实际支持的范围。
- unsupported：若下方证据明确支持修正后的表述，可改写并在句末加对应【EV:evidence_id】；否则删除。
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
) -> QualityGateResult:
    if not reader_review or not reader_review_passed(reader_review):
        raise ArticleQualityError("reader_background_not_proven", {"reader_review": reader_review or {}})
    body = "\n\n".join(f"## {s.get('heading', '')}\n{s.get('text', '')}" for s in sections)
    audit_rounds = []
    audit = {}
    for round_index in range(config.ARTICLE_CLAIM_AUDIT_ROUNDS):
        body = "\n\n".join(f"## {s.get('heading', '')}\n{s.get('text', '')}" for s in sections)
        evidence_store.clear_claims()
        audit = _audit_claims(body, evidence_store)
        blocking = _register_and_find_blocking_claims(audit, evidence_store)
        if not audit.get("claims"):
            blocking.append({"reason": "claim_auditor_returned_no_claims"})
        audit["blocking_claims"] = blocking
        audit["pass"] = not blocking
        audit["round"] = round_index + 1
        audit_rounds.append(audit)
        if not blocking:
            break
        if round_index + 1 >= config.ARTICLE_CLAIM_AUDIT_ROUNDS:
            raise ArticleQualityError("claim_audit_failed", {"claim_audit": audit, "claim_audit_rounds": audit_rounds})
        _repair_blocking_sections(sections, blocking, evidence_store)

    review = _final_review(article_title, thesis, body)
    scores = review.get("scores") or {}
    normalized = {key: _bounded_score(scores.get(key)) for key in FINAL_SCORE_DIMENSIONS}
    review["scores"] = normalized
    review["average"] = round(sum(normalized.values()) / len(normalized), 3) if normalized else 0.0
    fatal = [str(x) for x in review.get("fatal_issues", []) if str(x).strip()]
    passed = review.get("overall_pass") is True and not fatal and review["average"] >= config.ARTICLE_MIN_FINAL_SCORE
    review["pass"] = passed
    if not passed:
        raise ArticleQualityError("editorial_final_review_failed", {"claim_audit": audit, "final_review": review})
    audit["audit_rounds"] = len(audit_rounds)
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
    return chat_json(
        model=config.MODEL_FACTCHECK,
        system_prompt=CLAIM_AUDIT_PROMPT,
        user_prompt="文章：\n" + body + "\n\n可用证据：\n" + json.dumps(evidence, ensure_ascii=False),
        temperature=0.0,
    )


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
        # inference 是作者基于材料作出的解释，不要求来源逐字陈述；审计器负责确认它没有冒充事实。
        # 只有 fact / reported_result 必须得到直接证据支持。
    return blocking


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
        section["text"] = chat_text(
            model=config.MODEL_EDITORIAL,
            system_prompt=CLAIM_REPAIR_PROMPT,
            user_prompt=(
                "审计问题：\n" + json.dumps(relevant, ensure_ascii=False, indent=2)
                + "\n\n可用于修复的证据：\n" + json.dumps(evidence, ensure_ascii=False, indent=2)
                + "\n\n当前小节：\n" + text
            ),
            temperature=0.2,
        )
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
