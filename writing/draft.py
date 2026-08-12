"""逐节撰写短深文。机制卡在幕后约束；读者看到连贯说明文。"""

from __future__ import annotations

import json

import config
from processing.llm_client import chat_text
from research.evidence import EvidenceStore

DRAFT_SYSTEM_PROMPT = """\
你是「深潜 AI 周刊」主笔，正在写短深文的一个小节（中文）。

总原则：讲清机制，读起来像文章。禁止 Setup/How/Limits 字段清单，禁止成绩单口吻。

硬规则：
1. 具体事实只能来自素材；没有的数字/实验不要编。依赖素材的论断末尾加【EV:evidence_id】。
2. 服从本节 role：
   - context：具体失败/需求；以前怎么做、现在卡在哪。有 2 张卡时可在文末附一张短对比表，否则不要表。
   - mechanism：用连贯段落讲清一张机制卡，覆盖全部 algorithm_steps（叙述或一段伪代码）。
     禁止空指代与步骤目录体。
   - close：一个判断 + 未解；不要堆新缩写。
3. 控制在 max_chars 内。少金句。
4. 直接输出 Markdown 正文，不要重复小节标题。
"""


def draft_section(
    section: dict,
    evidence_store: EvidenceStore,
    topic_name: str,
    article_title: str,
    thesis: str = "",
    hook: str = "",
    beginner_context: str = "",
    core_intuition: str = "",
    running_example: str = "",
    mechanism_card: dict | None = None,
    publish_mode: str = "deep_dive",
    high_card_count: int = 1,
) -> str:
    materials = []
    for eid in section.get("evidence_ids", []):
        ev = evidence_store.get(eid)
        if ev:
            materials.append(f"[{eid}] {ev.title}\n来源: {ev.url}\n内容: {ev.excerpt[:2500]}")

    materials_block = (
        "\n\n".join(materials)
        if materials
        else "(本节没有分配到具体素材，不要编造具体事实)"
    )
    max_chars = int(section.get("max_chars") or 900)
    card_block = json.dumps(mechanism_card, ensure_ascii=False, indent=2) if mechanism_card else "(无)"

    user_prompt = (
        f"文章标题: {article_title}\n"
        f"publish_mode: {publish_mode}\n"
        f"high_card_count: {high_card_count}\n"
        f"全文主结论(thesis): {thesis or '(未提供)'}\n"
        f"钩子(hook): {hook or '(无)'}\n"
        f"话题: {topic_name}\n"
        f"beginner_context: {beginner_context or '(无)'}\n"
        f"core_intuition: {core_intuition or '(无)'}\n"
        f"running_example: {running_example or '(无)'}\n"
        f"本节标题: {section['heading']}\n"
        f"本节角色(role): {section.get('role', '')}\n"
        f"本节目标: {section.get('goal', '')}\n"
        f"字数上限: 约 {max_chars} 汉字\n\n"
        f"本节对应机制卡(JSON，勿输出为字段列表):\n{card_block}\n\n"
        f"可用素材:\n{materials_block}\n"
    )

    return chat_text(
        model=config.MODEL_WRITING,
        system_prompt=DRAFT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.45,
    )
