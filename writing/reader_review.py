"""冷启动读者盲审：只看成稿，发现背景缺口并定点修复 context。"""

from __future__ import annotations

import json

import config
from processing.llm_client import chat_json, chat_json_with_fallback, chat_text

CONTEXT_MAP_PROMPT = """\
你是技术文章的读者体验编辑。根据已有研究摘要，为“有软件或基础 AI 常识、但没关注过该细分方向”的读者建立最小前置知识地图。
只列理解本文中心机制不可缺少的概念，不做领域百科。只返回 JSON：
{"target_reader":"有软件或 AI 基础，但没有持续关注该细分方向",
"prerequisites":[{"concept":"...","reader_likely_knows":false,"plain_definition":"...",
"why_needed":"...","must_explain_before":"...","evidence_id":""}],
"causal_bridge":["4–6 个按顺序排列的因果节点"],
"likely_reader_questions":["3–5 个首次接触者会问的问题"]}
"""

READER_REVIEW_PROMPT = """\
你是一位有软件或基础 AI 常识、但从未关注过本文细分领域的读者。你只能依据成稿作答，
不能用作者的大纲、研究档案或你自己的专业知识替文章脑补。

检查读者在进入核心机制前，能否从文中理解：
1. 系统里的核心对象/参与者是什么；
2. 正常情况下它们如何协作；
3. 旧方法或既有防御在何时、检查什么；
4. 旧方法隐含了什么假设；
5. 新工作改变了哪一步，因此为何值得讨论。

再检查核心专业词是否在首次承担推理作用之前，用日常语言解释。不要因为文风流畅就判通过。

术语必须按“是否阻碍理解中心论证”分级，而不是要求术语零遗漏：
- blocking：不知道它就无法复述系统、旧方法、新变化或核心机制。例如本文方法名、关键对象和关键操作。
- non_blocking：实现细节、常见缩写或数学名词，即使不熟悉也不妨碍读者复述中心论证。
  缩写可建议首次展开；非关键数学术语优先改写成它在本文中的作用，但它们不能单独阻止发布。
- 一个术语只有在确实导致某项 checks=false 或留下关键 unanswered_question 时，才能列为 blocking。
- 保留英文术语不等于没有解释。若上下文已经说明 Skills、agent、token 等对象在本文中做什么，
  不要仅因为没有中文直译就判为 blocking；产品专名被机械翻译反而可能造成误解。

只返回 JSON：
{"checks":{"can_identify_system":false,"can_define_core_objects":false,
"can_explain_normal_workflow":false,"can_explain_old_approach":false,
"can_explain_new_failure":false},
"blocking_gaps":[{"concept":"...","reason":"为什么它阻断核心理解"}],
"nonblocking_terms":[{"term":"...","recommended_action":"expand|rephrase|remove|leave"}],
"unanswered_questions":[],
"repair_instructions":"只修复 blocking_gaps；不要为 nonblocking_terms 扩写背景", "background_pass":false}
"""

REPAIR_PROMPT = """\
你是技术文章作者。只修改下面的背景小节，补齐冷启动读者指出的理解缺口。
- 保留原有论点、引用标记和已支持的事实；不得发明论文结论、数字或 evidence_id。
- 只修复 blocking_gaps；不要为了 nonblocking_terms 增加科普段落。
- 新增内容只解释核心对象、正常工作流、旧方法、隐藏假设与新失败。
- 用自然散文，不写术语表、FAQ 或字段清单；控制新增内容在 400 汉字以内。
- 直接返回修改后的完整 Markdown 小节，不要解释。
"""


def ensure_reader_context(topic_name: str, dossier: dict) -> dict:
    current = dossier.get("reader_context")
    if isinstance(current, dict) and current.get("prerequisites") and current.get("causal_bridge"):
        return current
    result, _ = chat_json_with_fallback(
        model=config.MODEL_WRITING,
        fallback_model=config.MODEL_WRITING_FALLBACK,
        system_prompt=CONTEXT_MAP_PROMPT,
        user_prompt=(
            f"话题：{topic_name}\n"
            f"研究摘要：{dossier.get('topic_summary', '')}\n"
            f"已有外行背景：{dossier.get('beginner_context', '')}\n"
            f"核心判断：{dossier.get('narrative_angle', '')}"
        ),
        temperature=0.15,
    )
    result.setdefault("target_reader", "有软件或 AI 基础，但没有持续关注该细分方向")
    result.setdefault("prerequisites", [])
    result.setdefault("causal_bridge", [])
    result.setdefault("likely_reader_questions", [])
    dossier["reader_context"] = result
    return result


def review_for_cold_reader(title: str, sections: list[dict]) -> dict:
    body = "\n\n".join(f"## {s.get('heading', '')}\n{s.get('text', '')}" for s in sections)
    return chat_json(
        model=config.MODEL_EDITORIAL,
        system_prompt=READER_REVIEW_PROMPT,
        user_prompt=f"标题：{title}\n\n成稿：\n{body}",
        temperature=0.0,
    )


def reader_review_passed(review: dict) -> bool:
    checks = review.get("checks") or {}
    required = (
        "can_identify_system", "can_define_core_objects", "can_explain_normal_workflow",
        "can_explain_old_approach", "can_explain_new_failure",
    )
    checks_pass = all(checks.get(key) is True for key in required)
    if "blocking_gaps" in review:
        blocking = [x for x in review.get("blocking_gaps", []) if x]
    else:
        # 兼容旧审稿结果：若所有理解检查已通过且没有遗留问题，
        # 单独的 undefined_terms 视为非阻断术语，而不是发布失败。
        blocking = [] if checks_pass and not review.get("unanswered_questions") else list(
            review.get("undefined_terms", []) or []
        )
    return checks_pass and not blocking and not [
        x for x in review.get("unanswered_questions", []) if str(x).strip()
    ]


def repair_context_for_reader(context_text: str, review: dict, reader_context: dict) -> str:
    return chat_text(
        model=config.MODEL_EDITORIAL,
        system_prompt=REPAIR_PROMPT,
        user_prompt=(
            "目标读者与前置知识地图（只用于明确修复范围）：\n"
            + json.dumps(reader_context or {}, ensure_ascii=False, indent=2)
            + "\n\n盲审结果：\n" + json.dumps(review, ensure_ascii=False, indent=2)
            + "\n\n当前背景小节：\n" + context_text
        ),
        temperature=0.25,
    )
