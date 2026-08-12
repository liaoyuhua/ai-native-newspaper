"""Weekly Deep Dive v2 选题漏斗。

流程：原始信号 -> 候选技术问题 -> 独立编辑评审 -> Top-K 可成文性侦察 -> 发布/不发布。
热度不再直接决定赢家；任何候选都必须越过绝对质量门槛。
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from datetime import datetime, timezone
from urllib.parse import urlparse

import config
from memory.topic_memory import format_index_for_prompt, load_topic, match_topic_memory
from processing.editorial_models import (
    CandidateQuestion,
    EditorialJudgment,
    FeasibilityProbe,
    aggregate_judgments,
)
from processing.llm_client import chat_json
from processing.topic_selection import collect_all_items
from research.evidence import EvidenceStore
from research.tools import ResearchTools
from sources.common import RawItem
from sources.registry import SourceRegistry

logger = logging.getLogger(__name__)

QUESTION_PROMPT = """\
你是 Weekly Deep Dive 的选题编辑。刊物每周解释一个 AI 方法/算法方向的关键增量，
目标是 8–15 分钟读完、讲清一个中心问题和 1–2 个机制；不是新闻摘要，也不是完整领域综述。

把信号整理成 6–12 个“可回答的技术问题”，而不是宽泛主题。每个候选必须说明：
- why_now：最近 1–3 周出现了什么真实增量；
- technical_delta：相对已有路线具体改变什么；
- reader_takeaway：读者最终能形成什么判断；
- scope：一周研究预算内可以完成，不依赖覆盖整个领域。

不要因为公司或项目热度而假定技术重要；多个媒体转述同一事件只算一个信号。
若只是产品发布、榜单刷新、融资或缺少机制细节，不要生成候选。

只返回 JSON：
{"questions":[{"id":"q1","question":"以问句表达","scope":"...","why_now":"...",
"technical_delta":"...","reader_takeaway":"...","lens":"models_algorithms|systems_infra|other",
"item_ids":["..."],"memory_relation":"new|follow_up","related_topic_id":""}]}
"""

JUDGE_PROMPTS = {
    "research_significance": "你是 AI 研究负责人，重点判断问题是否代表真实的方法增量，而非传播热度。",
    "technical_reader": "你是资深但时间有限的技术读者，重点判断是否愿意花 8–15 分钟读，以及能带走什么。",
    "skeptical_editor": "你是怀疑主义总编，主动寻找炒作、伪新颖、范围过宽、证据单一和历史重复。",
}

JUDGE_RUBRIC = """\
对每个候选按 1–5 分评估以下维度：importance、timeliness、technical_delta、reader_payoff、
attractiveness、independent_confirmation、non_redundancy。3 表示勉强可写，4 表示本周强候选，5 极少使用。
fatal_flaws 填任何足以阻止发布的问题，例如 no_technical_delta、single_marketing_source、too_broad、
not_weekly_scope、duplicate_topic。不要按候选之间相对排名，要按绝对标准评分。
只返回 JSON：{"judgments":[{"id":"q1","scores":{"importance":1,"timeliness":1,
"technical_delta":1,"reader_payoff":1,"attractiveness":1,"independent_confirmation":1,
"non_redundancy":1},"fatal_flaws":[],"reasoning":"..."}]}
"""

PROBE_PROMPT = """\
你是技术文章的预研究编辑。仅根据给定的一手/二手信号，判断候选能否在一周预算内写成高质量文章。
按 1–5 分评价 primary_sources、mechanism_clarity、comparison_value、thesis_potential、weekly_scope_fit。
提出一个可被证据支持或证伪的 proposed_thesis，以及最多两个 mechanism_targets。
missing_evidence 必须诚实列出正式研究阶段仍需核实的材料。
recommendation 只能是 pursue 或 reject；没有一手来源、没有明确机制或范围失控时必须 reject。
只返回 JSON：{"probes":[{"id":"q1","scores":{"primary_sources":1,"mechanism_clarity":1,
"comparison_value":1,"thesis_potential":1,"weekly_scope_fit":1},"proposed_thesis":"...",
"mechanism_targets":["..."],"missing_evidence":["..."],"recommendation":"pursue|reject"}]}
"""


def generate_weekly_proposal_v2(lookback_days: int = 14) -> dict:
    items = collect_all_items(lookback_days)
    source_health = SourceRegistry().summary()
    if not items:
        raise RuntimeError("本周没有抓到任何条目，检查信源是否全部失效")
    items_by_id = {item.item_id: item for item in items}
    candidates = _generate_questions(items, format_index_for_prompt())
    candidates = [c for c in candidates if c.question and _valid_ids(c, items_by_id)]
    if not candidates:
        return _no_publish("no_candidate_questions", items, source_health=source_health)

    _attach_signal_scores(candidates, items_by_id)
    judgments = _judge_candidates(candidates, items_by_id)
    ranked = _rank_candidates(candidates, judgments)
    shortlist = ranked[: config.TOPIC_SHORTLIST_SIZE]
    probes, preliminary_evidence = _probe_candidates(shortlist[: config.TOPIC_PROBE_SIZE], items_by_id)

    for row in shortlist:
        probe = probes.get(row["candidate"]["id"])
        row["feasibility"] = probe.to_dict() if probe else None
        row["final_score"] = _final_score(row, probe)

    eligible = [row for row in shortlist if _eligible(row)]
    eligible.sort(key=lambda row: row["final_score"], reverse=True)
    if not eligible:
        return _no_publish("quality_threshold_not_met", items, shortlist, source_health)
    proposal = _build_proposal(eligible[0], shortlist, items, items_by_id)
    proposal["preliminary_evidence"] = preliminary_evidence
    proposal["source_health"] = source_health
    return proposal


def _generate_questions(items: list[RawItem], memory: str) -> list[CandidateQuestion]:
    ranked = sorted(items, key=lambda x: (x.authority, math.log1p(max(0, x.buzz))), reverse=True)[:140]
    compact = [_item_for_prompt(x, 350) for x in ranked]
    result = chat_json(
        model=config.MODEL_TOPIC_CLUSTERING,
        system_prompt=QUESTION_PROMPT,
        user_prompt=f"历史选题：\n{memory or '(无)'}\n\n近期信号：\n{json.dumps(compact, ensure_ascii=False)}",
        temperature=0.2,
    )
    return [CandidateQuestion.from_dict(raw, i) for i, raw in enumerate(result.get("questions", []))]


def _judge_candidates(
    candidates: list[CandidateQuestion], items_by_id: dict[str, RawItem]
) -> dict[str, list[EditorialJudgment]]:
    payload = [_candidate_payload(c, items_by_id) for c in candidates]
    out: dict[str, list[EditorialJudgment]] = defaultdict(list)
    for perspective in config.TOPIC_JUDGE_PERSPECTIVES:
        result = chat_json(
            model=config.MODEL_TOPIC_SCORING,
            system_prompt=JUDGE_PROMPTS[perspective] + "\n\n" + JUDGE_RUBRIC,
            user_prompt=json.dumps(payload, ensure_ascii=False),
            temperature=0.15,
        )
        for raw in result.get("judgments", []):
            cid = str(raw.get("id") or "")
            if cid:
                out[cid].append(EditorialJudgment.from_dict(perspective, raw))
    return out


def _probe_candidates(
    candidates: list[dict], items_by_id: dict[str, RawItem]
) -> tuple[dict[str, FeasibilityProbe], list[dict]]:
    evidence_store = EvidenceStore()
    tools = ResearchTools(evidence_store)
    payload = []
    has_fulltext: dict[str, bool] = {}
    for row in candidates:
        candidate = _candidate_from_row(row)
        item_candidates = sorted(
            (items_by_id[x] for x in candidate.item_ids if x in items_by_id),
            key=lambda item: item.authority,
            reverse=True,
        )
        probe_docs = []
        for item in item_candidates[: config.TOPIC_PROBE_DOCS_PER_CANDIDATE]:
            result = tools.fetch_fulltext(item.url)
            eid = result.get("evidence_id")
            ev = evidence_store.get(eid) if eid else None
            if ev:
                probe_docs.append({"evidence_id": eid, "title": ev.title, "url": ev.url,
                                   "source_type": ev.source_type, "excerpt": ev.excerpt[:3000]})
        candidate_payload = _candidate_payload(candidate, items_by_id)
        candidate_payload["preliminary_primary_materials"] = probe_docs
        payload.append(candidate_payload)
        row["preliminary_materials"] = probe_docs
        has_fulltext[candidate.id] = any(
            doc.get("source_type") in {"arxiv_fulltext", "webpage"} for doc in probe_docs
        )
    if not payload:
        return {}, []
    result = chat_json(
        model=config.MODEL_TOPIC_SCORING,
        system_prompt=PROBE_PROMPT,
        user_prompt=json.dumps(payload, ensure_ascii=False),
        temperature=0.15,
    )
    probes = {
        str(raw.get("id")): FeasibilityProbe.from_dict(raw)
        for raw in result.get("probes", [])
        if raw.get("id")
    }
    # 摘要可以证明“题目存在”，不能证明机制足够清楚；避免模型因摘要写得漂亮而打满分。
    for cid, probe in probes.items():
        if not has_fulltext.get(cid, False):
            probe.scores["mechanism_clarity"] = min(3.0, probe.scores["mechanism_clarity"])
            probe.scores["comparison_value"] = min(3.0, probe.scores["comparison_value"])
            if "需要获取真正全文以核实机制与实验" not in probe.missing_evidence:
                probe.missing_evidence.append("需要获取真正全文以核实机制与实验")
    return probes, evidence_store.to_list()


def _rank_candidates(candidates, judgments) -> list[dict]:
    rows = []
    for candidate in candidates:
        js = judgments.get(candidate.id, [])
        aggregate = aggregate_judgments(js)
        rows.append({
            "candidate": candidate.to_dict(),
            "judgments": [j.to_dict() for j in js],
            "editorial": aggregate,
            # 热度只占很小部分，且是 0..1 的来源内归一化结果。
            "shortlist_score": round(aggregate["average"] * 0.9 + candidate.signal_score * 5 * 0.1, 3),
        })
    return sorted(rows, key=lambda row: row["shortlist_score"], reverse=True)


def _attach_signal_scores(candidates: list[CandidateQuestion], items_by_id: dict[str, RawItem]) -> None:
    source_max: dict[str, float] = defaultdict(float)
    for item in items_by_id.values():
        source_max[item.source] = max(source_max[item.source], math.log1p(max(0.0, item.buzz)))
    now = datetime.now(timezone.utc)
    for candidate in candidates:
        items = [items_by_id[x] for x in candidate.item_ids if x in items_by_id]
        if not items:
            continue
        domains = {urlparse(x.url).netloc.lower() for x in items if x.url}
        diversity = min(1.0, len(domains) / 4.0)
        authority = sum(x.authority for x in items) / len(items)
        buzz_values = []
        recency_values = []
        for item in items:
            denom = source_max[item.source]
            buzz_values.append(math.log1p(max(0.0, item.buzz)) / denom if denom else 0.0)
            recency_values.append(_recency(item.published, now))
        buzz = sum(buzz_values) / len(buzz_values)
        recency = sum(recency_values) / len(recency_values)
        candidate.signal_score = round(0.3 * diversity + 0.3 * authority + 0.2 * buzz + 0.2 * recency, 4)


def _recency(value: str | None, now: datetime) -> float:
    if not value:
        return 0.35
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        days = max(0.0, (now - dt).total_seconds() / 86400)
        return max(0.0, 1.0 - days / 21.0)
    except ValueError:
        return 0.35


def _final_score(row: dict, probe: FeasibilityProbe | None) -> float:
    if not probe:
        return 0.0
    return round(row["editorial"]["average"] * 0.72 + probe.average * 0.28, 3)


def _eligible(row: dict) -> bool:
    probe = row.get("feasibility") or {}
    return bool(
        row["editorial"]["average"] >= config.TOPIC_MIN_EDITORIAL_SCORE
        and not row["editorial"].get("fatal_flaws")
        and probe.get("average", 0) >= config.TOPIC_MIN_FEASIBILITY_SCORE
        and probe.get("recommendation") == "pursue"
    )


def _build_proposal(selected: dict, shortlist: list[dict], items, items_by_id) -> dict:
    candidate = _candidate_from_row(selected)
    probe = selected["feasibility"]
    member_items = [items_by_id[x] for x in candidate.item_ids if x in items_by_id]
    related = load_topic(candidate.related_topic_id) if candidate.related_topic_id else None
    if related is None:
        related = match_topic_memory(candidate.question, candidate.scope)
    works = [
        {"item_id": x.item_id, "title": x.title, "url": x.url,
         "why": f"用于核实：{candidate.technical_delta}", "source": x.source}
        for x in sorted(member_items, key=lambda x: x.authority, reverse=True)[:8]
    ]
    return {
        "schema_version": "2.0",
        "publish_recommendation": "pursue",
        "selected_topic": {
            "name": candidate.question,
            "description": candidate.scope,
            "lens": candidate.lens,
            "score": selected["final_score"],
            "why_now": candidate.why_now,
            "technical_delta": candidate.technical_delta,
            "reader_takeaway": candidate.reader_takeaway,
            "memory_relation": candidate.memory_relation,
            "related_topic_id": (related or {}).get("topic_id", ""),
            "score_breakdown": {"editorial": selected["editorial"], "feasibility": probe,
                                "signal_score": candidate.signal_score},
        },
        "rationale": candidate.why_now,
        "proposed_thesis": probe.get("proposed_thesis", ""),
        "research_focus": probe.get("mechanism_targets", []),
        "missing_evidence": probe.get("missing_evidence", []),
        "candidate_works": works,
        "shortlist": shortlist,
        "runner_up_topics": [
            {"name": r["candidate"]["question"], "score": r.get("final_score", r["shortlist_score"]),
             "lens": r["candidate"]["lens"]} for r in shortlist if r is not selected
        ],
        "all_items": [x.to_dict() for x in items],
        "raw_topic_member_ids": candidate.item_ids,
    }


def _no_publish(
    reason: str,
    items: list[RawItem],
    shortlist: list[dict] | None = None,
    source_health: dict | None = None,
) -> dict:
    return {"schema_version": "2.0", "publish_recommendation": "skip", "reason": reason,
            "shortlist": shortlist or [], "all_items": [x.to_dict() for x in items],
            "source_health": source_health or {}}


def _item_for_prompt(item: RawItem, snippet_chars: int) -> dict:
    return {"id": item.item_id, "title": item.title, "url": item.url, "source": item.source,
            "authority": item.authority, "published": item.published, "buzz": item.buzz,
            "snippet": item.snippet[:snippet_chars]}


def _candidate_payload(candidate: CandidateQuestion, items_by_id: dict[str, RawItem]) -> dict:
    return {**candidate.to_dict(), "signals": [_item_for_prompt(items_by_id[x], 500)
            for x in candidate.item_ids if x in items_by_id]}


def _valid_ids(candidate: CandidateQuestion, items_by_id: dict[str, RawItem]) -> bool:
    candidate.item_ids = list(dict.fromkeys(x for x in candidate.item_ids if x in items_by_id))
    return bool(candidate.item_ids)


def _candidate_from_row(row: dict) -> CandidateQuestion:
    return CandidateQuestion.from_dict(row["candidate"], 0)
