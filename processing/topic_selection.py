"""
每周广度扫描 -> 聚类成候选 topic -> 客观信号打分 + 编辑镜头加权
-> 选出本周最值得深挖的 1 个 topic
-> 生成提案(topic + 候选 work 清单 + 打分依据)，供人工在 GitHub Issue 里审核确认。

客观信号：多源报道、信源权威度、社区热度、新鲜度。
编辑偏好：优先模型 / 算法 / 方法论；产品与产业可有，但不应成为默认赢家。
"""

from __future__ import annotations

import logging

import config
from processing.dedupe import dedupe_items
from processing.llm_client import chat_json
from sources.arxiv import fetch_arxiv
from sources.common import RawItem
from sources.github_trending import fetch_github_trending
from sources.hackernews import fetch_hackernews
from sources.rss_sources import fetch_rss

logger = logging.getLogger(__name__)


def collect_all_items(lookback_days: int = 8) -> list[RawItem]:
    items: list[RawItem] = []
    items.extend(fetch_rss(lookback_days))
    items.extend(fetch_hackernews(lookback_days))
    items.extend(fetch_arxiv(lookback_days))
    items.extend(fetch_github_trending())

    logger.info("原始条目数: %d", len(items))
    items = dedupe_items(items)
    logger.info("去重后条目数: %d", len(items))
    return items


CLUSTER_SYSTEM_PROMPT = """\
你是「深潜 AI 周刊」的资深编辑助理。本刊每周只深挖一个方向，定位是：
**模型 / 算法 / 方法论优先**；系统、工具链、基础设施可以；产品发布与行业落地可以点到，但不应占多数，更不能做成「AI 到处落地」式大杂烩。

你会看到本周抓取到的一批 AI 相关条目(论文/新闻/开源项目/讨论)。
把它们聚类成若干候选话题(topic)。每个 topic 必须：
- **具体可深挖**：读者读完应带走一个更新后的技术世界模型，而不是「本周 AI 又有新应用」的资讯汇总；
- **好例子**：「扩散模型加速采样」「LLM 推理时缩放与 test-time compute」「多模态 Agent 的长程规划」「某模型发布背后的架构/训练/评测争议」；
- **坏例子**：「AI 行业应用与语音/多模态产品落地」「AI 赋能各行各业」「本周大模型动态」——把金融/税务/电信/教育等不相关落地拼成一题，太泛，禁止；
- 同一技术方向的不同条目要合并；不要为每一条单独开题；也不要用「AI」这种空壳标题。

成题优先级（从高到低）：
1. 模型、算法、训练/推理方法、评测与基准、研究范式；
2. Agent/系统架构、工具链、基础设施、安全与对齐机制（有技术实质时）；
3. 具体产品或产业信号——仅当能收敛到可分析的技术/方法问题（例如某语音系统的实时架构），且不要把多个垂直行业硬揉成一题。

你还会看到本刊的「历史话题记忆」。请据此：
- 若本周条目明显是旧话题的演进/跟进，topic 名称或 description 里写清跟进关系，不要装作全新方向；
- 若只是重复报道、没有回应历史未解问题、也没有新的可观察信号，不要单独成题（可忽略或并入噪音）；
- 真正的新方向照常成题。

条目不需要全部归入某个 topic，明显是噪音/无关的可以不分配给任何 topic。
OpenAI/厂商的行业客户故事若缺乏可深挖的技术内核，宁可丢弃或单独标为 product_industry 的弱信号，也不要并进技术话题里凑热闹。

只返回 JSON，格式：
{
  "topics": [
    {
      "name": "话题名称(简洁，中文，具体)",
      "description": "一到两句话说明这个话题在讲什么（若是跟进请点明相对以往覆盖的增量）",
      "lens": "models_algorithms | systems_infra | product_industry | policy_society | other",
      "item_ids": ["id1", "id2", ...],
      "memory_relation": "new | follow_up | none",
      "related_topic_id": "若是跟进则填历史 topic_id，否则空字符串"
    }
  ]
}
"""


# 聚类时不必把所有条目塞进 prompt：按权威度+热度取前 N 条即可
CLUSTER_INPUT_LIMIT = 120


def cluster_topics(items: list[RawItem], memory_brief: str = "") -> list[dict]:
    ranked = sorted(items, key=lambda it: (it.authority * 2 + it.buzz / 100.0), reverse=True)
    selected = ranked[:CLUSTER_INPUT_LIMIT]
    logger.info("聚类输入条目: %d / %d", len(selected), len(items))

    compact = [
        {
            "id": it.item_id,
            "title": it.title,
            "source": it.source,
            "snippet": it.snippet[:200],
        }
        for it in selected
    ]

    user_prompt = (
        "本刊历史话题记忆：\n"
        + (memory_brief or "(尚无)")
        + "\n\n本周条目列表(JSON)：\n"
        + _to_json_str(compact)
    )
    result = chat_json(
        model=config.MODEL_TOPIC_CLUSTERING,
        system_prompt=CLUSTER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.2,
    )
    return result.get("topics", [])


def _normalize_lens(raw: str | None) -> str:
    lens = (raw or "other").strip().lower()
    if lens in config.TOPIC_LENS_WEIGHTS:
        return lens
    return "other"


def score_topics(topics: list[dict], items_by_id: dict[str, RawItem]) -> list[dict]:
    scored = []
    max_diversity = 1
    max_buzz = 1

    prelim = []
    for topic in topics:
        member_items = [items_by_id[i] for i in topic.get("item_ids", []) if i in items_by_id]
        if not member_items:
            continue
        sources = {it.source for it in member_items}
        diversity = len(sources)
        avg_authority = sum(it.authority for it in member_items) / len(member_items)
        total_buzz = sum(it.buzz for it in member_items)
        prelim.append((topic, member_items, diversity, avg_authority, total_buzz))
        max_diversity = max(max_diversity, diversity)
        max_buzz = max(max_buzz, total_buzz)

    for topic, member_items, diversity, avg_authority, total_buzz in prelim:
        w = config.SCORE_WEIGHTS
        norm_diversity = diversity / max_diversity
        norm_buzz = (total_buzz / max_buzz) if max_buzz > 0 else 0.0
        # 新鲜度：粗略地用"有 published 时间戳的条目占比"近似——完全没有时间戳的话题给中性分。
        dated = [it for it in member_items if it.published]
        novelty = 0.5 if not dated else min(1.0, len(dated) / len(member_items))

        objective = (
            w["source_diversity"] * norm_diversity
            + w["source_authority"] * avg_authority
            + w["community_buzz"] * norm_buzz
            + w["novelty"] * novelty
        )
        lens = _normalize_lens(topic.get("lens"))
        lens_weight = config.TOPIC_LENS_WEIGHTS[lens]
        score = objective * lens_weight

        scored.append(
            {
                **topic,
                "lens": lens,
                "member_count": len(member_items),
                "source_diversity": diversity,
                "avg_authority": round(avg_authority, 3),
                "total_buzz": round(total_buzz, 1),
                "objective_score": round(objective, 4),
                "lens_weight": lens_weight,
                "score": round(score, 4),
            }
        )

    scored.sort(key=lambda t: t["score"], reverse=True)
    return scored


PROPOSAL_SYSTEM_PROMPT = """\
你是「深潜 AI 周刊」总编。本刊深挖偏模型 / 算法 / 方法论；产品与产业只作佐证，不是主线。
已经确定了本周要深挖的话题，以及该话题下的相关条目。
请你：
1. 写一段"为什么本周值得深挖这个话题"的编辑评语(rationale)，语气像一个有判断力的总编，但不要夸张。
   说清楚可深挖的**技术/方法问题**是什么；不要写成行业资讯盘点。
   若提供了历史话题记忆：说明相对以往覆盖的增量（新信号/新工作/回应了哪个未解问题），避免把旧文重写一遍当理由。
2. 从条目里整理出一份"候选精选工作清单"(candidate_works)：每一项是一篇论文/一个项目/一次发布，
   附上一句话说明它为什么入选。优先保留有技术内核的论文、模型卡、系统/方法工作；
   纯客户故事/公关软文可少选或不选。同一个工作只出现一次，去掉明显是重复报道同一件事的条目。
   只基于提供的条目内容整理，不要编造你不确定的信息。
3. 给出 research_focus：本期研究应优先回答的 **1~2 个** 技术/方法问题。
   若是跟进篇，对齐历史 open_questions / follow_up_signals。
   不要列 4 个以上——问题越多，文章越容易摊成综述。

只返回 JSON：
{
  "rationale": "...",
  "candidate_works": [
    {"item_id": "...", "title": "...", "url": "...", "why": "..."}
  ],
  "research_focus": ["..."]
}
"""


def build_proposal(
    top_topic: dict,
    items_by_id: dict[str, RawItem],
    related_memory: dict | None = None,
) -> dict:
    from memory.topic_memory import format_topic_for_prompt

    member_items = [items_by_id[i] for i in top_topic.get("item_ids", []) if i in items_by_id]
    compact = [
        {
            "item_id": it.item_id,
            "title": it.title,
            "url": it.url,
            "source": it.source,
            "snippet": it.snippet[:400],
        }
        for it in member_items
    ]

    user_prompt = (
        f"话题名称: {top_topic['name']}\n"
        f"话题描述: {top_topic.get('description', '')}\n"
        f"与历史记忆关系: {top_topic.get('memory_relation', 'unknown')}\n"
        f"相关条目(JSON): {_to_json_str(compact)}\n"
    )
    if related_memory:
        user_prompt += "\n相关历史话题记忆:\n" + format_topic_for_prompt(related_memory)

    result = chat_json(
        model=config.MODEL_TOPIC_SCORING,
        system_prompt=PROPOSAL_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.4,
    )
    return result


def generate_weekly_proposal(lookback_days: int = 8) -> dict:
    from memory.topic_memory import format_index_for_prompt, load_topic, match_topic_memory

    items = collect_all_items(lookback_days)
    if not items:
        raise RuntimeError("本周没有抓到任何条目，检查信源是否全部失效")

    items_by_id = {it.item_id: it for it in items}
    memory_brief = format_index_for_prompt()

    topics = cluster_topics(items, memory_brief=memory_brief)
    if not topics:
        raise RuntimeError("聚类未产出任何话题")

    scored = score_topics(topics, items_by_id)
    if not scored:
        raise RuntimeError("打分后没有有效话题（可能 item_id 未对齐）")

    top_topic = scored[0]
    related_memory = None
    related_id = (top_topic.get("related_topic_id") or "").strip()
    if related_id:
        related_memory = load_topic(related_id)
    if related_memory is None:
        related_memory = match_topic_memory(
            top_topic["name"], top_topic.get("description", "")
        )

    proposal_detail = build_proposal(top_topic, items_by_id, related_memory=related_memory)

    memory_context = None
    if related_memory:
        from memory.topic_memory import format_topic_for_prompt

        memory_context = {
            "topic_id": related_memory.get("topic_id"),
            "display_name": related_memory.get("display_name"),
            "relation": top_topic.get("memory_relation") or "follow_up",
            "prompt_brief": format_topic_for_prompt(related_memory),
            "research_focus": proposal_detail.get("research_focus", []),
        }

    return {
        "selected_topic": {
            "name": top_topic["name"],
            "description": top_topic.get("description", ""),
            "lens": top_topic.get("lens", "other"),
            "score": top_topic["score"],
            "score_breakdown": {
                "source_diversity": top_topic["source_diversity"],
                "avg_authority": top_topic["avg_authority"],
                "total_buzz": top_topic["total_buzz"],
                "objective_score": top_topic.get("objective_score", top_topic["score"]),
                "lens_weight": top_topic.get("lens_weight", 1.0),
            },
            "memory_relation": top_topic.get("memory_relation", "new"),
            "related_topic_id": (related_memory or {}).get("topic_id", ""),
        },
        "rationale": proposal_detail.get("rationale", ""),
        "candidate_works": proposal_detail.get("candidate_works", []),
        "research_focus": proposal_detail.get("research_focus", []),
        "topic_memory_context": memory_context,
        "runner_up_topics": [
            {
                "name": t["name"],
                "score": t["score"],
                "lens": t.get("lens", "other"),
            }
            for t in scored[1:6]
        ],
        "all_items": [it.to_dict() for it in items],
        "raw_topic_member_ids": top_topic.get("item_ids", []),
    }


def _to_json_str(obj) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)
