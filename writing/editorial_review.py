"""
编辑审稿：短深文可用性——背景能否复述、机制是否覆盖、禁空指代/成绩单腔。
"""

from __future__ import annotations

import json
import logging
import re

import config
from processing.llm_client import chat_json, chat_text

logger = logging.getLogger(__name__)

REVIEW_ROLES = {"context", "mechanism", "close"}

REVIEW_SYSTEM_PROMPT = """\
你是「深潜 AI 周刊」编辑。验收短深文是否可用，不打文采分。

所有 scores 使用 1–5 分：1=完全不可用，3=达到发布下限，5=非常清楚。不要用 0/1 布尔分。

按 role：
- context：没跟过该方向的人能否复述「以前怎么做、现在卡在哪」？是否避免行业套话与成绩单/评分表口吻？
  context 不负责展开后续机制步骤；mechanism_coverage 必须填 5，missing_steps 必须为空。
- mechanism：机制卡 algorithm_steps 是否被散文覆盖？读者能否大致说出控制流？
  是否出现「有方案/已有研究」空指代？是否 Setup/How/Limits 或步骤目录腔？
- close：是否给出一个清楚判断？有没有突然堆未解释的新缩写/新战线？
  close 不负责重复机制步骤；mechanism_coverage 必须填 5，missing_steps 必须为空。

致命问题按 role 判断：context 背景模糊；mechanism 步骤没写全或空指代；任何 role 出现成绩单/字段清单腔；close 开新战线。
不要因 context/close 没有复述 mechanism card 而判失败。

只返回 JSON：
{
  "scores": {"clarity": 1, "mechanism_coverage": 1, "prose_not_table": 1},
  "overall_pass": false,
  "missing_steps": [],
  "issues": ["..."],
  "revise_instructions": "..."
}
"""

REVISE_SYSTEM_PROMPT = """\
你是原文作者，按编辑反馈修改本节（中文 Markdown）。
- 补机制步骤时写成连贯散文或一段伪代码，不要改成字段列表/成绩单。
- 删除空指代；不要编造新数字。
- 直接输出修改后全文，不要解释。
"""

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")


def review_and_revise_section(
    section_text: str,
    section: dict,
    *,
    thesis: str = "",
    mechanism_card: dict | None = None,
) -> str:
    role = (section.get("role") or "").strip().lower()
    if role not in REVIEW_ROLES:
        return section_text

    review = _review_section(section_text, section, thesis=thesis, mechanism_card=mechanism_card)
    scores = review.get("scores") or {}
    clarity = float(scores.get("clarity") or 0)
    prose = float(scores.get("prose_not_table") or 0)
    mechanism = float(scores.get("mechanism_coverage") or 0)
    if role == "mechanism":
        passed = clarity >= 3 and prose >= 3 and mechanism >= 3
    else:
        # context/close 不因“不含机制步骤”被误判；它们有各自的职责。
        passed = clarity >= 3 and prose >= 3
    logger.info(
        "编辑审稿 role=%s pass=%s scores=%s missing_steps=%s",
        role,
        passed,
        review.get("scores"),
        review.get("missing_steps"),
    )
    if passed:
        return section_text

    instructions = (review.get("revise_instructions") or "").strip()
    issues = list(review.get("issues") or [])
    missing = review.get("missing_steps") or []
    if missing:
        issues.append("未覆盖步骤: " + " | ".join(str(m) for m in missing))
    if not instructions and not issues:
        return section_text
    return _revise_section(section_text, instructions, issues, mechanism_card)


def enforce_title_body_numbers(title: str, body: str, allowed_claims: list[str]) -> str:
    title_nums = set(_NUMBER_RE.findall(title or ""))
    if not title_nums:
        return title
    pool = (body or "") + "\n" + "\n".join(allowed_claims)
    body_nums = set(_NUMBER_RE.findall(pool))
    bad = {n for n in title_nums if n not in body_nums}
    if not bad:
        return title
    cleaned = title
    for n in sorted(bad, key=len, reverse=True):
        cleaned = re.sub(rf"[^，。：:\s]{{0,12}}{re.escape(n)}[^，。：:\s]{{0,12}}", "", cleaned)
    cleaned = re.sub(r"[：:]\s*$", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ：:-—")
    if len(cleaned) < 8:
        return re.sub(r"\d+(?:\.\d+)?%?", "", title).strip(" ：:-—") or title
    logger.info("标题数字与正文不一致，已改写标题: %s -> %s", title, cleaned)
    return cleaned


def _review_section(
    section_text: str,
    section: dict,
    *,
    thesis: str,
    mechanism_card: dict | None,
) -> dict:
    user_prompt = (
        f"role: {section.get('role', '')}\n"
        f"heading: {section.get('heading', '')}\n"
        f"goal: {section.get('goal', '')}\n"
        f"thesis: {thesis or '(无)'}\n"
        f"mechanism_card:\n{json.dumps(mechanism_card or {}, ensure_ascii=False)}\n\n"
        f"正文:\n{section_text}\n"
    )
    try:
        return chat_json(
            model=config.MODEL_EDITORIAL,
            system_prompt=REVIEW_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.2,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("编辑审稿失败，停止发布: %s", exc)
        raise RuntimeError("section_editorial_review_unavailable") from exc


def _revise_section(
    section_text: str,
    instructions: str,
    issues: list,
    mechanism_card: dict | None,
) -> str:
    issue_block = "\n".join(f"- {i}" for i in issues) if issues else "(无)"
    user_prompt = (
        f"问题:\n{issue_block}\n\n"
        f"修改指令:\n{instructions or '(按问题修正)'}\n\n"
        f"机制卡:\n{json.dumps(mechanism_card or {}, ensure_ascii=False)}\n\n"
        f"原始正文:\n{section_text}\n"
    )
    try:
        return chat_text(
        model=config.MODEL_EDITORIAL,
            system_prompt=REVISE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.4,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("编辑改写失败，停止发布: %s", exc)
        raise RuntimeError("section_editorial_revision_failed") from exc
