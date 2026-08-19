"""三角色编辑部：研究包 -> 主编初稿 -> 事实编辑报告 -> 主编一次修订。"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import config
from processing.llm_client import chat_json_with_fallback
from research.evidence import EvidenceStore
from writing.contracts import align_sections
from writing.quality_gate import ArticleQualityError, _validate_article_contract

logger = logging.getLogger(__name__)

MARKER_RE = re.compile(r"【EV:([^】]+)】")
FACT_CATEGORIES = {
    "unsupported_number",
    "factual_mismatch",
    "mechanism_mismatch",
    "scope_overreach",
    "source_attribution",
}

FACT_EDITOR_PROMPT = """\
你是技术文章的事实编辑。主编已经完成全文，你只负责指出会误导读者的事实边界问题，不改稿、不评价文风。

只报告以下问题：
- 数字、实验结果或模型范围不被引用证据支持；
- 方法步骤、训练目标或因果关系与来源矛盾；
- 把单一模型/数据集结果外推成普遍结论；
- 把作者分析写成论文明确结论；
- 引用来源与正文归属不一致。

不要报告：普通段落总结、类比、价值判断、标题偏好、术语中英文选择、没有新增外部事实的作者综合。
不要要求每句作者分析都加“可能/可以推测”。最多返回 12 个高价值问题；quote 必须是正文中包含问题的完整原句。
具体实验比较、模型范围、机制步骤或来源归属若没有证据支持，属于 blocking，即使它不是全文中心结论。
advisory 只用于“现有证据基本支持、但边界或归属还可以更精确”的情况，不能用来容纳无来源的外部事实。

只返回 JSON：
{"issues":[{"severity":"blocking|advisory","category":"unsupported_number|factual_mismatch|mechanism_mismatch|scope_overreach|source_attribution",
"section_index":0,"quote":"正文准确原句","problem":"问题边界","evidence_ids":["ev_x"],"suggested_revision":"尽量小的修改建议"}],
"summary":"一句话事实评价"}
"""

CHIEF_REVISION_PROMPT = """\
你是这篇中文 AI 技术文章的主编。研究助理已提供材料，事实编辑只提供问题清单；文章结构、解释方式和声音由你负责。

根据事实报告对全文做一次有限修订：
- 修复所有 blocking 事实问题；优先收窄、限定或删除被点名的句子，不扩写无关内容；
- 保留未被点名的段落、有效【EV:evidence_id】、中心问题和章节职责；
- 不因事实编辑反馈把全文改成免责声明；正常作者综合可以自然表达；
- 默认读者有软件或 AI 基础，但没关注过该细分方向；背景应足够，但不写术语百科；
- 遵循技术社区用语，不机械翻译产品名和工程对象。产品语境中的 Skills 保留英文；agent、prompt、token 等可自然保留；
- 不得新增证据中没有的数字、实验结果、方法步骤或因果结论。
- 页面会单独渲染 title 和 subtitle；sections 中不得重复文章标题、Markdown H1（#）或标题摘要引用块。

如果材料不足以修复核心事实，decision=reject；否则 decision=publish。
只返回 JSON：
{"decision":"publish|reject","reason":"...","sections":[{"heading":"...","role":"context|mechanism|close","card_index":null,"text":"完整 Markdown 正文"}],
"resolved_issue_indexes":[0],"unresolved_issue_indexes":[]}
"""


@dataclass
class EditorialResult:
    sections: list[dict]
    fact_report: dict
    chief_decision: dict
    deterministic_checks: dict

    def to_dict(self) -> dict:
        return {
            "workflow": "three_role_editorial_v1",
            "fact_editor": self.fact_report,
            "chief_editor": self.chief_decision,
            "deterministic_checks": self.deterministic_checks,
        }


def run_three_role_editorial(
    title: str,
    thesis: str,
    sections: list[dict],
    outline_sections: list[dict],
    mechanism_cards: list[dict],
    evidence_store: EvidenceStore,
) -> EditorialResult:
    initial_checks = deterministic_publish_checks(
        title, thesis, sections, mechanism_cards, evidence_store,
    )
    if not initial_checks["pass"]:
        raise ArticleQualityError("deterministic_draft_check_failed", initial_checks)

    fact_report = run_fact_editor(title, thesis, sections, evidence_store)
    blocking = [issue for issue in fact_report["issues"] if issue["severity"] == "blocking"]
    if not blocking:
        return EditorialResult(
            sections, fact_report,
            {"decision": "publish", "reason": "事实编辑未发现阻断问题", "revision_applied": False},
            initial_checks,
        )

    # 一旦确实需要修订，主编同时看到 advisory，避免已经改稿却留下顺手可修的边界问题；
    # 只有 blocking 必须显式 resolved。
    editorial_issues = blocking + [
        issue for issue in fact_report["issues"] if issue["severity"] != "blocking"
    ]
    revised, decision = revise_by_chief_editor(
        title, thesis, sections, outline_sections, editorial_issues, evidence_store,
    )
    expected_indexes = set(range(len(blocking)))
    resolved_indexes = set(decision.get("resolved_issue_indexes") or [])
    if (
        decision.get("decision") != "publish"
        or decision.get("unresolved_issue_indexes")
        or not expected_indexes.issubset(resolved_indexes)
    ):
        raise ArticleQualityError("chief_editor_rejected", {
            "fact_report": fact_report, "chief_editor": decision,
        })
    unresolved_quotes = [
        issue for issue in blocking
        if any(issue["quote"] in str(section.get("text") or "") for section in revised)
    ]
    if unresolved_quotes:
        raise ArticleQualityError("chief_editor_left_blocking_quotes", {
            "issues": unresolved_quotes, "chief_editor": decision,
        })
    final_checks = deterministic_publish_checks(
        title, thesis, revised, mechanism_cards, evidence_store,
    )
    if not final_checks["pass"]:
        raise ArticleQualityError("chief_editor_regression", {
            "checks": final_checks, "fact_report": fact_report, "chief_editor": decision,
        })
    decision["revision_applied"] = True
    return EditorialResult(revised, fact_report, decision, final_checks)


def run_fact_editor(
    title: str, thesis: str, sections: list[dict], evidence_store: EvidenceStore,
) -> dict:
    # 论文摘要和结果表常出现在全文前 2k 字之后；窗口过短会把真实来源误判为缺证据。
    evidence = _referenced_evidence(sections, evidence_store, max_excerpt=6000)
    article = {
        "title": title,
        "thesis": thesis,
        "sections": [
            {"section_index": index, "heading": s.get("heading", ""), "text": s.get("text", "")}
            for index, s in enumerate(sections)
        ],
    }
    result, model_used = chat_json_with_fallback(
        model=config.MODEL_FACTCHECK,
        fallback_model=config.MODEL_FACTCHECK_FALLBACK,
        system_prompt=FACT_EDITOR_PROMPT,
        user_prompt=(
            "文章：\n" + json.dumps(article, ensure_ascii=False)
            + "\n\n引用证据：\n" + json.dumps(evidence, ensure_ascii=False)
        ),
        temperature=0.0,
    )
    issues = []
    for raw in (result.get("issues") or [])[:12]:
        category = str(raw.get("category") or "")
        quote = str(raw.get("quote") or "").strip()
        problem = str(raw.get("problem") or "").strip()
        suggested_revision = str(raw.get("suggested_revision") or "").strip()
        try:
            section_index = int(raw.get("section_index"))
        except (TypeError, ValueError):
            continue
        if category not in FACT_CATEGORIES or not quote or not (0 <= section_index < len(sections)):
            continue
        if quote not in str(sections[section_index].get("text") or ""):
            logger.warning("事实编辑返回的 quote 不在指定章节，忽略: %s", quote[:80])
            continue
        # 有些模型会先论证“证据支持/不构成问题”，却仍机械地把条目放进 issues。
        # 这种自相矛盾的输出不能触发改稿或阻止发布。
        if (
            suggested_revision in {"无需修改", "无须修改", "不需修改"}
            or any(phrase in problem for phrase in (
                "不构成事实错误", "不构成事实问题", "不构成问题",
                "不构成 blocking", "非 blocking",
            ))
        ):
            continue
        # 过短的片段无法稳定定位事实问题，不能触发全篇修订。
        severity = "blocking" if raw.get("severity") == "blocking" and len(quote) >= 8 else "advisory"
        eids = [str(eid) for eid in raw.get("evidence_ids") or [] if evidence_store.get(str(eid))]
        issues.append({
            "severity": severity,
            "category": category,
            "section_index": section_index,
            "quote": quote,
            "problem": problem,
            "evidence_ids": eids,
            "suggested_revision": suggested_revision,
        })
    return {
        "issues": issues,
        "summary": str(result.get("summary") or "").strip(),
        "model": model_used,
        "blocking_count": sum(issue["severity"] == "blocking" for issue in issues),
    }


def revise_by_chief_editor(
    title: str,
    thesis: str,
    sections: list[dict],
    outline_sections: list[dict],
    blocking_issues: list[dict],
    evidence_store: EvidenceStore,
) -> tuple[list[dict], dict]:
    evidence_ids = [eid for issue in blocking_issues for eid in issue.get("evidence_ids") or []]
    evidence = []
    for eid in dict.fromkeys(evidence_ids):
        ev = evidence_store.get(eid)
        if ev:
            evidence.append({"id": eid, "title": ev.title, "excerpt": ev.excerpt[:6000]})
    result, model_used = chat_json_with_fallback(
        model=config.MODEL_WRITING,
        fallback_model=config.MODEL_WRITING_FALLBACK,
        system_prompt=CHIEF_REVISION_PROMPT,
        user_prompt=(
            "标题：" + title + "\n主结论：" + thesis
            + "\n\n事实编辑报告：\n" + json.dumps(blocking_issues, ensure_ascii=False, indent=2)
            + "\n\n相关证据：\n" + json.dumps(evidence, ensure_ascii=False)
            + "\n\n当前全文：\n" + json.dumps({"sections": sections}, ensure_ascii=False)
        ),
        temperature=0.25,
    )
    revised = align_sections(result.get("sections") or [], outline_sections)
    decision = {
        "decision": "publish" if result.get("decision") == "publish" else "reject",
        "reason": str(result.get("reason") or "").strip(),
        "resolved_issue_indexes": [int(x) for x in result.get("resolved_issue_indexes") or [] if str(x).isdigit()],
        "unresolved_issue_indexes": [int(x) for x in result.get("unresolved_issue_indexes") or [] if str(x).isdigit()],
        "model": model_used,
    }
    return revised, decision


def deterministic_publish_checks(
    title: str,
    thesis: str,
    sections: list[dict],
    mechanism_cards: list[dict],
    evidence_store: EvidenceStore,
) -> dict:
    contract = _validate_article_contract(title, thesis, sections, mechanism_cards)
    invalid_markers = []
    format_violations = []
    for index, section in enumerate(sections):
        text = str(section.get("text") or "")
        if re.search(r"(?m)^#\s+\S", text):
            format_violations.append({"section_index": index, "reason": "section_contains_h1"})
        for marker in MARKER_RE.findall(text):
            for eid in [part.strip() for part in marker.split(",") if part.strip()]:
                if evidence_store.get(eid) is None:
                    invalid_markers.append({"section_index": index, "evidence_id": eid})
    return {
        "pass": contract["pass"] and not invalid_markers and not format_violations,
        "article_contract": contract,
        "invalid_evidence_markers": invalid_markers,
        "format_violations": format_violations,
    }


def _referenced_evidence(
    sections: list[dict], evidence_store: EvidenceStore, *, max_excerpt: int,
) -> list[dict]:
    ids = []
    for section in sections:
        for marker in MARKER_RE.findall(str(section.get("text") or "")):
            ids.extend(part.strip() for part in marker.split(",") if part.strip())
    evidence = []
    for eid in dict.fromkeys(ids):
        ev = evidence_store.get(eid)
        if ev:
            evidence.append({
                "id": eid, "title": ev.title, "url": ev.url,
                "source_type": ev.source_type, "excerpt": ev.excerpt[:max_excerpt],
            })
    return evidence
