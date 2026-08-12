"""短深文大纲：hook+context → mechanism×1–2 → close。

机制卡仅作助写；不再有成绩单式札记模式。
"""

from __future__ import annotations

import config
from processing.llm_client import chat_json_with_fallback
from research.mechanism_cards import format_cards_for_prompt

OUTLINE_SYSTEM_PROMPT = """\
你是「深潜 AI 周刊」主笔(中文)。产品是**短深文**：可读、有判断，讲清 1–2 个机制即可。
不是社论金句文，不是成绩单，不是 Lilian 级长篇脚手架。

硬约束：
1. 全文 1 个 thesis。
2. 节结构：
   - context（1节）：用具体失败/需求讲清「以前怎么做、现在卡在哪」。可含硬钩子。零行话。
     若有 2 张机制卡，可在本节末尾附 **唯一一张** 简短 Markdown 对比表（工作/关键差异），表后 1–2 句说明为何深挖它们；只有 1 张卡时不要表。
   - mechanism（1–2节，每张卡一节）：连贯散文讲清机制，覆盖 algorithm_steps（可融入叙述或一段伪代码）。
     禁止 Setup/How/Limits 字段清单；禁止「步骤1/2/3」目录体当主文；禁止「有方案尝试」空指代。
   - close（1节）：一个明确判断 + 指向未解（开放问题正文里点到即可，不必列表腔）。禁止新开第三条战线。
3. 标题数字必须来自机制卡 evidence claims；没有就不写进标题。
4. beginner_context / core_intuition 只作背景素材，不要单独开「直觉」长章。
5. context 必须建立最小背景模型：参与者、正常工作流、既有防御/旧方法、隐藏假设、新失败。
   每节声明 introduces / assumes / reader_questions_answered；专业概念必须先解释、后使用。

只返回 JSON：
{
  "title": "...",
  "subtitle": "...",
  "thesis": "...",
  "hook": "可选；数字必须能在卡 evidence 中找到，否则空字符串",
  "sections": [
    {
      "heading": "...",
      "role": "context | mechanism | close",
      "card_index": null,
      "goal": "撰写指令",
      "max_chars": 800,
      "evidence_ids": ["..."]
      ,"introduces": ["本节首次解释的概念"]
      ,"assumes": ["可安全假设读者已经理解的概念"]
      ,"reader_questions_answered": ["本节回答的冷启动读者问题"]
    }
  ]
}
mechanism 节必须带 card_index（0-based，对应 high 卡顺序）。
"""


def generate_outline(topic_name: str, dossier: dict) -> dict:
    high_cards = list(dossier.get("high_mechanism_cards") or [])
    all_cards = list(dossier.get("mechanism_cards") or [])

    user_prompt = (
        f"话题: {topic_name}\n"
        f"topic_summary: {dossier.get('topic_summary', '')}\n"
        f"beginner_context: {dossier.get('beginner_context', '')}\n"
        f"reader_context: {dossier.get('reader_context', {})}\n"
        f"core_intuition（可选用一次）: {dossier.get('core_intuition', '')}\n"
        f"running_example（可选）: {dossier.get('running_example', '')}\n"
        f"narrative_angle: {dossier.get('narrative_angle', '')}\n\n"
        f"本篇深挖的机制卡（1–2张）:\n{format_cards_for_prompt(high_cards)}\n\n"
        f"其余卡仅可进精选清单:\n{format_cards_for_prompt(_other_cards(all_cards, high_cards))}\n\n"
        f"开放问题: {dossier.get('open_questions', [])}\n"
    )

    outline, model_used = chat_json_with_fallback(
        model=config.MODEL_WRITING,
        fallback_model=config.MODEL_WRITING_FALLBACK,
        system_prompt=OUTLINE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.35,
    )
    outline = _enforce_outline(outline, high_cards, dossier.get("reader_context") or {})
    outline["_model_used"] = model_used
    return outline


def _other_cards(all_cards: list[dict], high_cards: list[dict]) -> list[dict]:
    high_ids = {(c.get("evidence_id"), c.get("title")) for c in high_cards}
    return [c for c in all_cards if (c.get("evidence_id"), c.get("title")) not in high_ids]


def _enforce_outline(outline: dict, high_cards: list[dict], reader_context: dict | None = None) -> dict:
    sections = list(outline.get("sections") or [])
    fixed: list[dict] = []

    context = next((s for s in sections if (s.get("role") or "").lower() in {"context", "overview"}), None)
    if context is None and sections:
        context = {**sections[0], "role": "context"}
    if context is None:
        context = {
            "heading": "问题出在哪",
            "role": "context",
            "goal": "讲清以前怎么做、现在卡在哪",
            "max_chars": 800,
            "evidence_ids": [],
        }
    context["role"] = "context"
    context.setdefault("max_chars", 800)
    context.pop("card_index", None)
    context["evidence_ids"] = (context.get("evidence_ids") or [])[:6]
    context.setdefault("introduces", [])
    context.setdefault("assumes", [])
    context.setdefault("reader_questions_answered", [])
    # 模型偶尔会漏填规划元数据。规范器将必要概念明确安排到背景节，
    # 让后续草稿真正收到解释任务，而不是只在质量门报错。
    required_concepts = [
        str(p.get("concept") or "").strip()
        for p in (reader_context or {}).get("prerequisites", [])
        if isinstance(p, dict) and not p.get("reader_likely_knows") and str(p.get("concept") or "").strip()
    ]
    introduced_anywhere = {
        str(x).strip()
        for section in sections
        for x in (section.get("introduces") or [])
        if str(x).strip()
    }
    missing_concepts = [x for x in required_concepts if x not in introduced_anywhere]
    if missing_concepts:
        context["introduces"] = list(dict.fromkeys(context["introduces"] + missing_concepts))
        context["goal"] = (
            str(context.get("goal") or "讲清以前怎么做、现在卡在哪")
            + "；在进入论文机制前，用自然散文解释：" + "、".join(missing_concepts)
        )
    fixed.append(context)

    mech_sections = [s for s in sections if (s.get("role") or "").lower() == "mechanism"]
    for idx, card in enumerate(high_cards[:2]):
        sec = next((s for s in mech_sections if s.get("card_index") == idx), None)
        if sec is None and idx < len(mech_sections):
            sec = mech_sections[idx]
        if sec is None:
            sec = {
                "heading": card.get("title") or f"机制 {idx + 1}",
                "goal": f"散文讲清 Card {idx}；覆盖全部 algorithm_steps，禁止清单腔",
                "max_chars": 1100,
                "evidence_ids": [],
            }
        sec["role"] = "mechanism"
        sec["card_index"] = idx
        sec.setdefault("max_chars", 1100)
        eids = list(sec.get("evidence_ids") or [])
        if card.get("evidence_id") and card["evidence_id"] not in eids:
            eids.insert(0, card["evidence_id"])
        for ev in card.get("evidence") or []:
            eid = ev.get("evidence_id")
            if eid and eid not in eids:
                eids.append(eid)
        sec["evidence_ids"] = eids[:6]
        sec.setdefault("introduces", [])
        sec.setdefault("assumes", [])
        sec.setdefault("reader_questions_answered", [])
        fixed.append(sec)

    close = next((s for s in sections if (s.get("role") or "").lower() in {"close", "synthesis"}), None)
    if close is None:
        close = {
            "heading": "判断与未解",
            "role": "close",
            "goal": "给出一个判断并点明未解，不新开战线",
            "max_chars": 700,
            "evidence_ids": [],
        }
    close["role"] = "close"
    close.setdefault("max_chars", 700)
    close.pop("card_index", None)
    close.setdefault("introduces", [])
    close.setdefault("assumes", [])
    close.setdefault("reader_questions_answered", [])
    fixed.append(close)

    outline["sections"] = fixed
    return outline
