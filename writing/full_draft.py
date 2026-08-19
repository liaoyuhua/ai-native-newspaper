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
4. 事实段与作者分析要让读者能够区分。一个推理段落在入口处表明“这意味着 / 更合理的理解是”等即可，
   不要每句话重复“可能、可以推测”，也不要把作者分析伪装成论文原结论。
5. 不强行制造故事、三段论、金句或贯穿比喻；比喻只有确实降低理解成本时才使用一次。
6. 不写论文清单，不逐项复述机制卡字段，不为了对称强拼两个无关工作。
7. context 不提前完整讲完后续机制；close 不引入正文没有展开的新路线。
8. 全文约 3000–5000 汉字；以完整解释为准，不灌水。
9. 默认读者有软件或 AI 基础，但没关注过该细分方向。context 在进入论文机制前必须用自然散文补齐
   参与者、正常工作流、既有防御或旧方法、其隐藏假设和新失败；不要写成术语表。
10. method_role=primary_subject 是本文主角。机制正文至少一半用于它，并覆盖输入、数据/状态构造、
    ≥3 步控制流、训练目标或接口变化、关键实验结果与局限。background 卡只能帮助解释主角。
11. 标题、subtitle、章节标题合计最多使用一个贯穿比喻；技术结论优先直说。
12. 遵循中文技术社区的自然用语，不为中文纯度机械翻译产品名、API 对象或工程术语。
    产品语境中的 Skills 写作“Skills（可复用的任务说明与资源包）”，后文写 Skills，不写“技能”；
    agent、prompt、token、embedding、rollout 等可保留英文。普通能力语境中的 skills 仍可译为“能力”。
    不要在每个中文术语后机械添加括号英文。
13. 页面会单独渲染 title 和 subtitle。任何 section.text 都不得再次写文章标题、subtitle、Markdown H1（#）
    或标题下的摘要引用块；每节正文直接从自然段开始。

只返回 JSON：{"sections":[{"heading":"必须与大纲对应","role":"context|mechanism|close",
"card_index":null,"text":"Markdown 正文"}]}
"""

DRAFT_DEPTH_REPAIR_PROMPT = """\
你是技术文章的深度编辑。初稿没有达到机制覆盖契约，请重写整篇，但只扩展已有证据支持的内容。
- 保留大纲节数、role、card_index 和所有有效【EV:evidence_id】。
- 优先补足 method_role=primary_subject：输入、数据/状态构造、至少3步控制流、目标/接口变化、实验结果与局限。
- background 方法只保留解释主角所需的最少内容。
- 不新增材料中没有的事实、数字或因果结论；信息不足时明确写出证据边界，不要灌水。
- 保留原稿自然的技术术语；产品名和工程对象不要机械翻译，正常作者分析也不要逐句添加免责声明。
- 全文达到配置中的最低深度，机制内容至少占 45%，主角机制节至少 650 汉字。
只返回 JSON：{"sections":[{"heading":"...","role":"context|mechanism|close","card_index":null,"text":"..."}]}
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
        "primary_work_title": dossier.get("primary_work_title", ""),
        "outline": outline.get("sections", []),
        "open_questions": (dossier.get("open_questions") or [])[:3],
    }
    prompt_body = (
        "Story brief:\n" + json.dumps(brief, ensure_ascii=False, indent=2)
        + "\n\n机制卡：\n" + format_cards_for_prompt(dossier.get("high_mechanism_cards") or [])
        + "\n\n证据片段：\n" + json.dumps(evidence, ensure_ascii=False)
    )
    result, model_used = chat_json_with_fallback(
        model=config.MODEL_WRITING,
        fallback_model=config.MODEL_WRITING_FALLBACK,
        system_prompt=FULL_DRAFT_PROMPT,
        user_prompt=prompt_body,
        temperature=0.4,
    )
    sections = align_sections(result.get("sections", []), outline.get("sections", []))
    depth_issues = _draft_depth_issues(sections, outline.get("sections", []))
    if depth_issues:
        repaired, repair_model = chat_json_with_fallback(
            model=config.MODEL_WRITING,
            fallback_model=config.MODEL_WRITING_FALLBACK,
            system_prompt=DRAFT_DEPTH_REPAIR_PROMPT,
            user_prompt=(
                "未通过项：\n- " + "\n- ".join(depth_issues)
                + "\n\n原始初稿：\n" + json.dumps({"sections": sections}, ensure_ascii=False)
                + "\n\n" + prompt_body
            ),
            temperature=0.3,
        )
        sections = align_sections(repaired.get("sections", []), outline.get("sections", []))
        model_used = repair_model
    return sections, model_used


def _draft_depth_issues(sections: list[dict], outline_sections: list[dict]) -> list[str]:
    roles = {
        (s.get("role"), s.get("card_index")): s.get("method_role")
        for s in outline_sections
    }
    counts = [len("".join(str(s.get("text") or "").split())) for s in sections]
    total = sum(counts)
    mechanism = sum(n for n, s in zip(counts, sections) if s.get("role") == "mechanism")
    primary = sum(
        n for n, s in zip(counts, sections)
        if s.get("role") == "mechanism"
        and roles.get((s.get("role"), s.get("card_index"))) == "primary_subject"
    )
    issues = []
    if total < config.ARTICLE_MIN_BODY_CHARS:
        issues.append(f"正文 {total} 字，低于 {config.ARTICLE_MIN_BODY_CHARS}")
    if not total or mechanism / total < config.ARTICLE_MIN_MECHANISM_RATIO:
        issues.append("机制内容占比不足")
    if primary < config.ARTICLE_MIN_PRIMARY_MECHANISM_CHARS:
        issues.append(f"主角机制 {primary} 字，低于 {config.ARTICLE_MIN_PRIMARY_MECHANISM_CHARS}")
    return issues


def expand_primary_after_factcheck(
    section_text: str,
    primary_card: dict,
    evidence_store: EvidenceStore,
    depth_issues: list[str],
) -> tuple[str, str]:
    """事实核查删减后，只补主角机制，并把补写再次交给事实核查。"""
    evidence_ids = [str(primary_card.get("evidence_id") or "")]
    evidence_ids.extend(
        str(x.get("evidence_id") or "") for x in primary_card.get("evidence") or []
    )
    evidence = []
    for eid in dict.fromkeys(x for x in evidence_ids if x):
        ev = evidence_store.get(eid)
        if ev:
            evidence.append({"id": eid, "title": ev.title, "excerpt": ev.excerpt[:3600]})
    result, model_used = chat_json_with_fallback(
        model=config.MODEL_WRITING,
        fallback_model=config.MODEL_WRITING_FALLBACK,
        system_prompt=(
            "你是技术文章的机制编辑。事实核查删掉了没有依据的内容，现在只扩写主角机制节。"
            "必须保留已有有效【EV:evidence_id】，新增事实只能来自给定证据并在句末标相应 evidence_id。"
            "补足输入、数据/状态构造、至少3步控制流、训练目标或接口变化、关键结果与局限。"
            "如果证据没有细节，明确写出边界，不得猜测。目标 750–1000 汉字。"
            "遵循中文技术社区用语，不机械翻译产品名/工程对象，不给正常作者分析逐句加免责声明。"
            '只返回 JSON：{"text":"修改后的完整小节"}'
        ),
        user_prompt=(
            "未通过项：\n- " + "\n- ".join(depth_issues)
            + "\n\n主角机制卡：\n" + json.dumps(primary_card, ensure_ascii=False, indent=2)
            + "\n\n可用证据：\n" + json.dumps(evidence, ensure_ascii=False)
            + "\n\n当前小节：\n" + section_text
        ),
        temperature=0.25,
    )
    return str(result.get("text") or section_text).strip(), model_used
