"""基于 story brief 一次生成整篇初稿，避免逐节调用造成重复和局部最优。"""

from __future__ import annotations

import json

import config
from processing.llm_client import chat_json_with_fallback
from research.evidence import EvidenceStore
from research.mechanism_cards import format_cards_for_prompt
from writing.contracts import align_sections

FULL_DRAFT_PROMPT = """\
你是 Weekly Deep Dive 的中文主笔。写一篇每周 8–15 分钟可读完的 AI 方法/算法技术文章。
参考高质量研究型技术博客的准确、清楚和证据习惯，但不追求完整领域综述。

硬要求：
1. 整篇只回答一个中心问题，深挖 1–2 个机制；材料多不等于都要写。
2. 先解释旧方法具体卡在哪里，再解释新机制如何改变控制流、表示或优化目标。
3. 具体外部事实、方法细节和实验结果必须来自材料，并在句末加【EV:evidence_id】。
4. 作者推断必须用“这意味着 / 可以推测 / 更合理的理解是”等方式与来源事实区分。
5. 不强行制造故事、三段论、金句或贯穿比喻；比喻只有确实降低理解成本时才使用一次。
6. 不写论文清单，不逐项复述机制卡字段，不为了对称强拼两个无关工作。
7. context 不提前完整讲完后续机制；close 不引入正文没有展开的新路线。
8. 全文约 3000–5000 汉字；以完整解释为准，不灌水。
9. 默认读者有软件或 AI 基础，但没关注过该细分方向。context 在进入论文机制前必须用自然散文补齐
   参与者、正常工作流、既有防御或旧方法、其隐藏假设和新失败；不要写成术语表。

只返回 JSON：{"sections":[{"heading":"必须与大纲对应","role":"context|mechanism|close",
"card_index":null,"text":"Markdown 正文"}]}
"""


def generate_full_draft(
    topic_name: str, outline: dict, dossier: dict, evidence_store: EvidenceStore
) -> tuple[list[dict], str]:
    evidence_ids = []
    for section in outline.get("sections", []):
        evidence_ids.extend(section.get("evidence_ids", []))
    for card in dossier.get("high_mechanism_cards") or []:
        if card.get("evidence_id"):
            evidence_ids.append(card["evidence_id"])
        evidence_ids.extend(
            claim.get("evidence_id") for claim in card.get("evidence", []) if claim.get("evidence_id")
        )
    evidence_ids.extend(
        work.get("evidence_id") for work in dossier.get("key_works", []) if work.get("evidence_id")
    )
    evidence = []
    for eid in dict.fromkeys(evidence_ids):
        ev = evidence_store.get(eid)
        if ev:
            evidence.append({"id": eid, "title": ev.title, "url": ev.url, "excerpt": ev.excerpt[:3200]})

    brief = {
        "topic": topic_name,
        "title": outline.get("title", topic_name),
        "subtitle": outline.get("subtitle", ""),
        "thesis": outline.get("thesis", ""),
        "hook": outline.get("hook", ""),
        "beginner_context": dossier.get("beginner_context", ""),
        "reader_context": dossier.get("reader_context", {}),
        "core_intuition": dossier.get("core_intuition", ""),
        "running_example": dossier.get("running_example", ""),
        "narrative_angle": dossier.get("narrative_angle", ""),
        "outline": outline.get("sections", []),
        "open_questions": (dossier.get("open_questions") or [])[:3],
    }
    result, model_used = chat_json_with_fallback(
        model=config.MODEL_WRITING,
        fallback_model=config.MODEL_WRITING_FALLBACK,
        system_prompt=FULL_DRAFT_PROMPT,
        user_prompt=(
            "Story brief:\n" + json.dumps(brief, ensure_ascii=False, indent=2)
            + "\n\n机制卡：\n" + format_cards_for_prompt(dossier.get("high_mechanism_cards") or [])
            + "\n\n证据片段：\n" + json.dumps(evidence, ensure_ascii=False)
        ),
        temperature=0.4,
    )
    return align_sections(result.get("sections", []), outline.get("sections", [])), model_used
