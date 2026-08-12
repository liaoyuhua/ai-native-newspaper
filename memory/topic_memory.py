"""
跨周 Topic Memory：每期发布后写回「覆盖了什么 / 还欠什么 / 下次跟进信号」，
供后续选题与深度研究读取，避免无记忆的重复浅挖。
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import config
from processing.llm_client import chat_json, chat_json_with_fallback

logger = logging.getLogger(__name__)

MEMORY_ROOT = config.DATA_DIR / "memory"
TOPICS_DIR = MEMORY_ROOT / "topics"
INDEX_PATH = MEMORY_ROOT / "index.json"

DISTILL_SYSTEM_PROMPT = """\
你是「深潜 AI 周刊」的知识管理员。刚发布了一期深度文章，请把本期沉淀成可跨周复用的 Topic Memory。

目标：让未来选题和研究 agent 知道——我们已经讲过什么、刻意没展开什么、什么信号出现时值得再写跟进篇。
不要复述全文，只抽可执行的编辑记忆。

只返回 JSON：
{
  "topic_id": "英文短横线 slug，稳定可复用，如 agent-skills",
  "display_name": "中文话题名",
  "aliases": ["可能的别名/近义说法"],
  "one_line_summary": "一句话概括本刊对该方向的当前理解",
  "covered_angles": ["本期真正展开过的角度，3-6条，要具体"],
  "key_works_covered": [{"title": "...", "url": "...", "note": "一句话"}],
  "open_questions": ["仍未解决、值得继续跟踪的问题"],
  "follow_up_signals": ["出现哪些外部信号时，值得再开一期跟进（具体可观察）"],
  "avoid_repeating": ["下篇若再写同一方向，应避免重复铺陈的内容"]
}
"""


def ensure_dirs() -> None:
    TOPICS_DIR.mkdir(parents=True, exist_ok=True)


def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "-", s, flags=re.UNICODE)
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:80] or "topic"


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("无法解析 %s，使用默认值", path)
        return default


def load_index() -> list[dict]:
    ensure_dirs()
    data = _read_json(INDEX_PATH, {"topics": []})
    return data.get("topics", [])


def save_index(entries: list[dict]) -> None:
    ensure_dirs()
    INDEX_PATH.write_text(
        json.dumps({"topics": entries, "updated_at": _now()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_topic(topic_id: str) -> dict | None:
    path = TOPICS_DIR / f"{topic_id}.json"
    if not path.exists():
        return None
    return _read_json(path, None)


def format_index_for_prompt(limit: int = 30) -> str:
    """给选题阶段用的紧凑记忆摘要。"""
    entries = load_index()[:limit]
    if not entries:
        return "(尚无历史话题记忆)"
    lines = []
    for e in entries:
        oq = e.get("standing_open_questions") or []
        oq_preview = "；".join(oq[:2]) if oq else "无"
        lines.append(
            f"- [{e.get('topic_id')}] {e.get('display_name')} "
            f"(最近: {e.get('last_week_id', '?')}): {e.get('one_line_summary', '')}\n"
            f"  未解问题: {oq_preview}"
        )
    return "\n".join(lines)


def format_topic_for_prompt(memory: dict) -> str:
    """给研究 agent 用的单话题详细记忆。"""
    if not memory:
        return ""
    coverage = memory.get("coverage", [])
    latest = coverage[-1] if coverage else {}
    lines = [
        f"话题: {memory.get('display_name')} (id={memory.get('topic_id')})",
        f"当前理解: {memory.get('one_line_summary', '')}",
        f"已覆盖期数: {len(coverage)}；最近一期: {latest.get('week_id', '无')}",
    ]
    if latest.get("covered_angles"):
        lines.append("最近已展开角度:")
        lines.extend(f"  - {a}" for a in latest["covered_angles"])
    if latest.get("avoid_repeating"):
        lines.append("应避免重复铺陈:")
        lines.extend(f"  - {a}" for a in latest["avoid_repeating"])
    oq = memory.get("standing_open_questions") or latest.get("open_questions") or []
    if oq:
        lines.append("仍待跟踪的开放问题:")
        lines.extend(f"  - {q}" for q in oq)
    signals = latest.get("follow_up_signals") or []
    if signals:
        lines.append("值得跟进的外部信号:")
        lines.extend(f"  - {s}" for s in signals)
    return "\n".join(lines)


def match_topic_memory(topic_name: str, description: str = "") -> dict | None:
    """从 index 里找与本周话题最相关的一条记忆；没有把握则返回 None。"""
    entries = load_index()
    if not entries:
        return None

    # 先做廉价字符串匹配
    name_l = topic_name.lower()
    for e in entries:
        candidates = [e.get("display_name", ""), e.get("topic_id", "")] + list(e.get("aliases") or [])
        for c in candidates:
            if not c:
                continue
            if c.lower() in name_l or name_l in c.lower():
                mem = load_topic(e["topic_id"])
                if mem:
                    return mem

    if len(entries) > 40:
        entries = entries[:40]

    compact = [
        {
            "topic_id": e.get("topic_id"),
            "display_name": e.get("display_name"),
            "aliases": e.get("aliases", []),
            "one_line_summary": e.get("one_line_summary", ""),
        }
        for e in entries
    ]
    result = chat_json(
        model=config.MODEL_TOPIC_SCORING,
        system_prompt=(
            "判断本周话题是否与历史 Topic Memory 中的某一条是同一方向（含明显跟进）。"
            "只有相当确定时才返回 topic_id；否则 matched=false。"
            '只返回 JSON: {"matched": true/false, "topic_id": "...或空字符串", "reason": "一句"}'
        ),
        user_prompt=(
            f"本周话题: {topic_name}\n描述: {description}\n\n历史记忆:\n"
            + json.dumps(compact, ensure_ascii=False)
        ),
        temperature=0.0,
    )
    if result.get("matched") and result.get("topic_id"):
        mem = load_topic(result["topic_id"])
        if mem:
            logger.info("匹配到历史话题记忆: %s (%s)", result["topic_id"], result.get("reason", ""))
            return mem
    return None


def distill_and_upsert(
    *,
    week_id: str,
    topic_name: str,
    article: dict,
    dossier: dict,
    existing: dict | None = None,
) -> dict:
    """从本期文章+研究档案蒸馏记忆并写回磁盘。"""
    ensure_dirs()

    section_outline = [
        {"heading": s.get("heading", ""), "excerpt": (s.get("text") or "")[:400]}
        for s in article.get("sections", [])
    ]
    user_prompt = (
        f"week_id: {week_id}\n"
        f"topic_name: {topic_name}\n"
        f"article_title: {article.get('title', '')}\n"
        f"subtitle: {article.get('subtitle', '')}\n\n"
        f"dossier.topic_summary: {dossier.get('topic_summary', '')}\n"
        f"dossier.narrative_angle: {dossier.get('narrative_angle', '')}\n"
        f"dossier.open_questions: {json.dumps(dossier.get('open_questions', []), ensure_ascii=False)}\n"
        f"article.open_questions: {json.dumps(article.get('open_questions', []), ensure_ascii=False)}\n\n"
        f"关键工作:\n{json.dumps(dossier.get('key_works', [])[:12], ensure_ascii=False)}\n\n"
        f"文章各节摘要:\n{json.dumps(section_outline, ensure_ascii=False)}\n"
    )
    if existing:
        user_prompt += (
            f"\n已有同方向记忆(请在此基础上演进，topic_id 尽量保持 {existing.get('topic_id')}):\n"
            f"{format_topic_for_prompt(existing)}\n"
        )

    distilled, _model_used = chat_json_with_fallback(
        model=config.MODEL_WRITING,
        fallback_model=config.MODEL_WRITING_FALLBACK,
        system_prompt=DISTILL_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.2,
    )

    topic_id = (existing or {}).get("topic_id") or distilled.get("topic_id") or slugify(topic_name)
    topic_id = slugify(str(topic_id).replace("_", "-"))

    coverage_entry = {
        "week_id": week_id,
        "article_title": article.get("title", ""),
        "covered_angles": distilled.get("covered_angles", []),
        "key_works_covered": distilled.get("key_works_covered", []),
        "open_questions": distilled.get("open_questions", []) or article.get("open_questions", []),
        "follow_up_signals": distilled.get("follow_up_signals", []),
        "avoid_repeating": distilled.get("avoid_repeating", []),
    }

    memory = existing or load_topic(topic_id) or {
        "topic_id": topic_id,
        "display_name": topic_name,
        "aliases": [],
        "coverage": [],
    }
    memory["topic_id"] = topic_id
    memory["display_name"] = distilled.get("display_name") or topic_name
    aliases = list(dict.fromkeys([*(memory.get("aliases") or []), *(distilled.get("aliases") or []), topic_name]))
    memory["aliases"] = aliases
    memory["one_line_summary"] = distilled.get("one_line_summary") or memory.get("one_line_summary", "")
    memory["standing_open_questions"] = distilled.get("open_questions") or memory.get("standing_open_questions", [])

    coverage = [c for c in memory.get("coverage", []) if c.get("week_id") != week_id]
    coverage.append(coverage_entry)
    memory["coverage"] = coverage
    memory["updated_at"] = _now()

    out_path = TOPICS_DIR / f"{topic_id}.json"
    out_path.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
    _upsert_index(memory, week_id)
    logger.info("已更新 Topic Memory: %s", out_path)
    return memory


def _upsert_index(memory: dict, week_id: str) -> None:
    entries = load_index()
    topic_id = memory["topic_id"]
    brief = {
        "topic_id": topic_id,
        "display_name": memory.get("display_name", ""),
        "aliases": memory.get("aliases", []),
        "one_line_summary": memory.get("one_line_summary", ""),
        "standing_open_questions": memory.get("standing_open_questions", [])[:5],
        "last_week_id": week_id,
        "coverage_count": len(memory.get("coverage", [])),
    }
    entries = [e for e in entries if e.get("topic_id") != topic_id]
    entries.insert(0, brief)
    save_index(entries)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
