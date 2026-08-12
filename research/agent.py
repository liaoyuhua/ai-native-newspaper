"""
深度研究 agent：给定本周选中的 topic + 候选 work 清单(来自广度扫描)，
自主循环调用 research/tools.py 里的工具，直到判断信息足够，
最终产出一份"研究档案"(research dossier)：关键工作清单 + 背景知识点 + 未解决问题，
每一条都标注 evidence_id，可以在 EvidenceStore 里核对原文。
"""

from __future__ import annotations

import json
import logging

import config
from processing.llm_client import run_tool_loop
from research.evidence import EvidenceStore
from research.tools import TOOL_SCHEMAS, ResearchTools, build_tool_impls

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = """\
你是一位严谨的 AI 领域研究员，正在为「深潜 AI 周刊」做前期研究。
成文目标是**短深文**：讲清 1–2 个机制即可。后续会从关键工作抽取内部「机制卡」。
原则：宁可 2 个工作全文细节扎实，不要 6 个口号式条目。

你手上有几个工具：search_arxiv、search_semantic_scholar、get_semantic_scholar_related、
fetch_fulltext、web_search。请像真正做研究一样自主使用它们：

- 先搞清「从业者/研究者现在具体卡在什么失败或需求上」，再搜代表工作；
- 对准备写入 key_works 的论文/系统，**必须 fetch_fulltext**，确保能抽出：状态/接口、≥3 步控制流、可核对证据；
  不要只看摘要就下结论；
- 如果发现某篇论文引用/被引用了看起来很关键的工作，用 get_semantic_scholar_related 顺着追下去；
- 如果素材里提到了具体的开源项目/产品发布，用 web_search 或 fetch_fulltext 核实官方信息；
- key_works 建议 3–5 个（写作只会深挖其中 1–2 个），但每个都必须有可抓的全文细节；
  另写清背景、核心直觉、贯穿场景、开放问题。不要浅尝辄止。
- 如果提供了「历史话题记忆」：优先深挖其中的 open_questions 与 follow_up_signals；
  对 avoid_repeating / 已覆盖角度只做必要对照，不要把旧文重写一遍；把增量写进 narrative_angle。

【关于准确性，这一点非常重要】：
你最终产出的每一条关键结论/事实，都必须能对应到你真实调用工具后拿到的 evidence_id。
不确定、没有查到确凿依据的内容，宁可不写，也不要编造或凭印象猜测。

当你觉得调研已经足够充分时，不要再调用任何工具，直接输出最终的研究档案，格式为纯 JSON(不要用markdown代码块包裹):
{
  "topic_summary": "对这个方向的整体理解，2-4句话",
  "beginner_context": "给外行的背景：3-5句，零行话。读者读完应能复述『以前怎么做、现在卡在哪、为什么这周值得谈』",
  "reader_context": {
    "target_reader": "有软件或 AI 基础，但没有持续关注该细分方向",
    "prerequisites": [
      {"concept":"理解核心机制前必须知道的概念","reader_likely_knows":false,"plain_definition":"一句零行话定义","why_needed":"它支撑哪一步理解","must_explain_before":"必须早于哪个概念出现","evidence_id":"常识可空；外部事实必须填写"}
    ],
    "causal_bridge": ["按先后顺序列出读者理解问题所需的 4–6 个因果节点"],
    "likely_reader_questions": ["首次接触该方向的读者最可能追问的问题，3–5 条"]
  },
  "core_intuition": "全文核心直觉：一个类比或心智模型 + 为什么成立（2-4句）。读者读完应能用自己的话讲出『所以问题该这样切』",
  "running_example": "贯穿全文的同一个具体场景（ vignette ）。后面每个关键工作都要用它说明差异，不要换场景",
  "key_works": [
    {
      "evidence_id": "从工具结果里拿到的 evidence_id",
      "title": "...",
      "url": "...",
      "why_relevant": "为什么这项工作值得提到",
      "key_findings": "从原文/摘要里提炼出的具体、可核查的要点(不是泛泛而谈)",
      "how_it_handles_the_example": "在 running_example 里，这项工作具体怎么处理（崩溃恢复/隔离/记忆写入等），写清可观察差异"
    }
  ],
  "background_notes": [
    {"point": "一条背景/上下文知识点", "evidence_id": "支撑这条知识点的 evidence_id，如果是常识性背景可以留空字符串"}
  ],
  "open_questions": ["这个方向目前公认的开放问题/争议点，尽量给出来源，最多3条"],
  "narrative_angle": "用1段话给出本期最锋利的编辑判断：读者读完应带走的一个更新后的世界模型是什么"
}
"""


def run_research(
    topic_name: str,
    topic_description: str,
    candidate_works: list[dict],
    topic_memory: dict | None = None,
    research_focus: list[str] | None = None,
    proposed_thesis: str = "",
) -> tuple[dict, EvidenceStore]:
    from memory.topic_memory import format_topic_for_prompt

    evidence_store = EvidenceStore()
    tools = ResearchTools(evidence_store)
    tool_impls = build_tool_impls(tools)

    # 把广度扫描阶段已经找到的候选工作预先注册进证据库，agent 可以直接在此基础上深挖，而不必重新搜。
    seed_lines = []
    for w in candidate_works:
        eid = evidence_store.add(
            title=w.get("title", ""), url=w.get("url", ""), source_type="weekly_scan_seed",
            excerpt=w.get("why", ""),
        )
        seed_lines.append(f"- evidence_id={eid} | {w.get('title', '')} | {w.get('url', '')} | {w.get('why', '')}")

    memory_block = ""
    if topic_memory:
        memory_block = (
            "\n\n【历史话题记忆——请在此基础上做增量研究，不要重复已覆盖角度】\n"
            + format_topic_for_prompt(topic_memory)
        )
    focus_block = ""
    if research_focus:
        focus_block = "\n\n【本期研究优先问题】\n" + "\n".join(f"- {q}" for q in research_focus)
    thesis_block = ""
    if proposed_thesis:
        thesis_block = (
            "\n\n【选题阶段的暂定 thesis——这是待验证假设，不是既定结论】\n"
            + proposed_thesis
            + "\n研究后必须根据证据确认、收窄或推翻它。"
        )

    user_prompt = (
        f"本周选定的话题：{topic_name}\n"
        f"话题描述：{topic_description}\n\n"
        f"广度扫描阶段已经发现的候选工作(已经预先注册为 evidence，可以直接引用，也可以进一步 fetch_fulltext 深挖)：\n"
        + "\n".join(seed_lines)
        + memory_block
        + focus_block
        + thesis_block
        + "\n\n请开始你的深度研究。"
    )

    step_log: list[dict] = []

    def on_step(name, args, result):
        step_log.append({"tool": name, "args": args, "result_preview": str(result)[:300]})
        logger.info("研究 agent 调用工具 %s(%s)", name, {k: v for k, v in args.items() if k != "query"} or args)

    raw_output = run_tool_loop(
        model=config.MODEL_RESEARCH_AGENT,
        system_prompt=AGENT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tools=TOOL_SCHEMAS,
        tool_impls=tool_impls,
        max_tool_calls=config.RESEARCH_MAX_TOOL_CALLS,
        on_step=on_step,
    )

    dossier = _normalize_dossier(_parse_dossier(raw_output))
    dossier["_tool_call_log"] = step_log
    _log_dossier_gaps(dossier)
    return dossier, evidence_store


def _parse_dossier(raw_output: str) -> dict:
    text = raw_output.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        logger.error("研究 agent 最终输出不是合法 JSON，原文前500字: %s", text[:500])
        return {
            "topic_summary": text[:1000],
            "beginner_context": "",
            "reader_context": {},
            "core_intuition": "",
            "running_example": "",
            "key_works": [],
            "background_notes": [],
            "open_questions": [],
            "narrative_angle": "",
            "_parse_error": True,
        }


def _normalize_dossier(dossier: dict) -> dict:
    """补齐写作契约字段，避免下游 outline/draft 缺键。"""
    dossier.setdefault("topic_summary", "")
    dossier.setdefault("beginner_context", "")
    reader_context = dossier.get("reader_context")
    if not isinstance(reader_context, dict):
        reader_context = {}
    reader_context.setdefault("target_reader", "有软件或 AI 基础，但没有持续关注该细分方向")
    reader_context.setdefault("prerequisites", [])
    reader_context.setdefault("causal_bridge", [])
    reader_context.setdefault("likely_reader_questions", [])
    dossier["reader_context"] = reader_context
    dossier.setdefault("core_intuition", "")
    dossier.setdefault("running_example", "")
    dossier.setdefault("background_notes", [])
    dossier.setdefault("open_questions", [])
    dossier.setdefault("narrative_angle", "")
    works = dossier.get("key_works") or []
    for w in works:
        if isinstance(w, dict):
            w.setdefault("how_it_handles_the_example", "")
    dossier["key_works"] = works
    return dossier


def _log_dossier_gaps(dossier: dict) -> None:
    gaps = []
    for key in ("beginner_context", "core_intuition", "running_example"):
        if not str(dossier.get(key) or "").strip():
            gaps.append(key)
    reader_context = dossier.get("reader_context") or {}
    if not reader_context.get("prerequisites"):
        gaps.append("reader_context.prerequisites")
    if not reader_context.get("causal_bridge"):
        gaps.append("reader_context.causal_bridge")
    works = dossier.get("key_works") or []
    missing_example = sum(
        1 for w in works if isinstance(w, dict) and not str(w.get("how_it_handles_the_example") or "").strip()
    )
    if missing_example:
        gaps.append(f"how_it_handles_the_example×{missing_example}")
    if gaps:
        logger.warning("研究档案缺少写作契约字段: %s", ", ".join(gaps))
