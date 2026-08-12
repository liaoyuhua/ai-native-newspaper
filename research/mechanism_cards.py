"""
从研究档案的关键工作中抽取 Mechanism Cards（内部助写结构，不渲染给读者）。

规则优先判定 high；模型的 low 不能一票否决。
可用卡 ≥1 → deep_dive；0 → insufficient（由 publish 跳过发文）。
"""

from __future__ import annotations

import logging
import re

import config
from processing.llm_client import chat_json
from processing.editorial_models import mechanism_is_publishable
from research.evidence import EvidenceStore
from research.tools import ResearchTools

logger = logging.getLogger(__name__)

EXTRACT_SYSTEM_PROMPT = """\
你是技术编辑，任务是从论文/项目原文片段中抽取「机制卡」(Mechanism Card)。
这是内部结构化笔记，供后续写成短深文；不要写散文，不要评价文采。

硬规则：
- 只能依据提供的原文 excerpt；没有的细节不要编。
- algorithm_steps 必须是可执行的控制流/算法步骤（≥3 步），每步写清「做什么」，禁止「提升/赋能/探索/改进效果」空话。
- state_or_interface 写清存什么、谁读写、关键数据结构或 API；写不出就标明不足。
- evidence 里每条 claim 必须能被 excerpt 支持，并填对应 evidence_id。
- confidence 仅作参考：即使你标 low，只要步骤/接口/证据字段填满，下游规则仍可能采用。

只返回 JSON：
{
  "problem": "它要解决的具体问题",
  "inputs_outputs": "输入/输出或任务设定",
  "state_or_interface": "状态表示或关键接口",
  "algorithm_steps": ["步骤1", "步骤2", "步骤3"],
  "evidence": [{"claim": "可核对论断", "evidence_id": "ev_xxx"}],
  "failure_modes": ["失败或边界情况"],
  "limits": ["局限"],
  "gaps": ["原文不足以支持的部分"],
  "confidence": "high | low"
}
"""

_FLUFF_STEP_RE = re.compile(
    r"(提升|赋能|探索|改进|优化效果|增强能力|具有重要意义|值得注意|未来可期|全面)",
)
_MIN_EXCERPT = 400


def extract_and_attach_mechanism_cards(dossier: dict, evidence_store: EvidenceStore) -> dict:
    """抽取机制卡，写入 dossier，并设定 publish_mode。"""
    works = list(dossier.get("key_works") or [])
    max_candidates = int(getattr(config, "MECHANISM_CARD_CANDIDATES", 5))
    tools = ResearchTools(evidence_store)

    cards: list[dict] = []
    for work in works[:max_candidates]:
        _refill_fulltext(work, evidence_store, tools)
        card = _extract_one(work, evidence_store)
        if card:
            cards.append(card)

    scored = [_finalize_card(c, evidence_store) for c in cards]
    high = [c for c in scored if c.get("confidence") == "high"]
    deep_n = int(getattr(config, "MECHANISM_DEEP_DIVE_CARDS", 2))
    min_high = int(getattr(config, "MECHANISM_MIN_HIGH_CARDS", 1))

    dossier["mechanism_cards"] = scored
    dossier["high_mechanism_cards"] = high[:deep_n]
    if len(high) >= min_high:
        dossier["publish_mode"] = "deep_dive"
    else:
        dossier["publish_mode"] = "insufficient"

    logger.info(
        "机制卡: 共 %d 张，high=%d，publish_mode=%s",
        len(scored),
        len(high),
        dossier["publish_mode"],
    )
    for c in scored:
        logger.info(
            "  - [%s] %s (steps=%d, thin=%s, gaps=%s)",
            c.get("confidence"),
            (c.get("title") or "")[:60],
            len(c.get("algorithm_steps") or []),
            c.get("thin_source", False),
            "; ".join((c.get("gaps") or [])[:2]) or "-",
        )
    return dossier


def format_cards_for_prompt(cards: list[dict]) -> str:
    if not cards:
        return "(无)"
    blocks = []
    for i, c in enumerate(cards, 1):
        steps = "\n".join(f"  {j}. {s}" for j, s in enumerate(c.get("algorithm_steps") or [], 1))
        ev = "; ".join(
            f"{e.get('claim', '')}【{e.get('evidence_id', '')}】" for e in (c.get("evidence") or [])
        )
        blocks.append(
            f"## Card {i}: {c.get('title', '')}\n"
            f"evidence_id: {c.get('evidence_id', '')}\n"
            f"url: {c.get('url', '')}\n"
            f"confidence: {c.get('confidence', '')}\n"
            f"problem: {c.get('problem', '')}\n"
            f"inputs_outputs: {c.get('inputs_outputs', '')}\n"
            f"state_or_interface: {c.get('state_or_interface', '')}\n"
            f"algorithm_steps:\n{steps or '  (无)'}\n"
            f"failure_modes: {c.get('failure_modes', [])}\n"
            f"limits: {c.get('limits', [])}\n"
            f"evidence_claims: {ev or '(无)'}\n"
            f"gaps: {c.get('gaps', [])}\n"
        )
    return "\n".join(blocks)


def _refill_fulltext(work: dict, evidence_store: EvidenceStore, tools: ResearchTools) -> None:
    """抽卡前尽量补全文，避免只靠短摘要。"""
    eid = (work.get("evidence_id") or "").strip()
    url = (work.get("url") or "").strip()
    ev = evidence_store.get(eid) if eid else None
    excerpt_len = len((ev.excerpt if ev else "") or "")
    fulltext_types = {"arxiv_fulltext", "webpage"}
    if ev and ev.source_type in fulltext_types and excerpt_len >= _MIN_EXCERPT:
        return
    target = url or eid
    if not target:
        return
    try:
        result = tools.fetch_fulltext(target)
        new_eid = result.get("evidence_id") or eid
        if new_eid and new_eid != eid:
            work["evidence_id"] = new_eid
        logger.info("抽卡前补全文: %s -> eid=%s", (work.get("title") or "")[:50], work.get("evidence_id"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("抽卡前补全文失败 %s: %s", work.get("title", ""), exc)


def _extract_one(work: dict, evidence_store: EvidenceStore) -> dict | None:
    eid = (work.get("evidence_id") or "").strip()
    ev = evidence_store.get(eid) if eid else None
    excerpt = (ev.excerpt if ev else "") or ""
    if not excerpt:
        excerpt = (work.get("key_findings") or "") + "\n" + (work.get("how_it_handles_the_example") or "")
    excerpt = excerpt.strip()
    # 摘要、搜索片段和扫描 seed 即使很长，也不能作为 high mechanism card 的全文依据。
    fulltext_types = {"arxiv_fulltext", "webpage"}
    thin = len(excerpt) < _MIN_EXCERPT or not ev or ev.source_type not in fulltext_types

    if len(excerpt) < 80:
        logger.info("跳过抽卡（素材过短）: %s", work.get("title", ""))
        return {
            "title": work.get("title", ""),
            "url": work.get("url", ""),
            "evidence_id": eid,
            "problem": "",
            "inputs_outputs": "",
            "state_or_interface": "",
            "algorithm_steps": [],
            "evidence": [],
            "failure_modes": [],
            "limits": [],
            "gaps": ["原文/摘要不足以抽取机制"],
            "confidence": "low",
            "thin_source": True,
        }

    user_prompt = (
        f"工作标题: {work.get('title', '')}\n"
        f"url: {work.get('url', '')}\n"
        f"主 evidence_id: {eid}\n"
        f"why_relevant: {work.get('why_relevant', '')}\n"
        f"key_findings: {work.get('key_findings', '')}\n\n"
        f"原文片段:\n{excerpt[:8000]}\n"
    )
    try:
        raw = chat_json(
            model=config.MODEL_MECHANISM,
            system_prompt=EXTRACT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.2,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("抽卡失败 %s: %s", work.get("title", ""), exc)
        return None

    raw["title"] = work.get("title", "") or raw.get("title", "")
    raw["url"] = work.get("url", "") or raw.get("url", "")
    raw["evidence_id"] = eid
    raw["thin_source"] = thin
    if thin:
        gaps = list(raw.get("gaps") or [])
        gaps.append("source excerpt shorter than preferred threshold")
        raw["gaps"] = gaps
    return raw


def _finalize_card(card: dict, evidence_store: EvidenceStore) -> dict:
    """规则优先：字段过线则强制 high；模型 low 只记 gaps。"""
    steps = [str(s).strip() for s in (card.get("algorithm_steps") or []) if str(s).strip()]
    card["algorithm_steps"] = steps
    state = str(card.get("state_or_interface") or "").strip()
    card["state_or_interface"] = state

    evidence = []
    for e in card.get("evidence") or []:
        if not isinstance(e, dict):
            continue
        eid = str(e.get("evidence_id") or "").strip()
        claim = str(e.get("claim") or "").strip()
        if not claim:
            continue
        if eid and evidence_store.get(eid) is None and eid != card.get("evidence_id"):
            continue
        if not eid:
            eid = card.get("evidence_id") or ""
        evidence.append({"claim": claim, "evidence_id": eid})
    card["evidence"] = evidence

    gaps = list(card.get("gaps") or [])
    model_conf = str(card.get("confidence") or "low").strip().lower()

    steps_ok = len(steps) >= 3 and not any(len(s) < 12 or _FLUFF_STEP_RE.search(s) for s in steps)
    state_ok = len(state) >= 20
    evidence_ok = len(evidence) >= 1
    thin = bool(card.get("thin_source"))

    fail_reasons = []
    if not steps_ok:
        fail_reasons.append("algorithm_steps 不足或含空话")
    if not state_ok:
        fail_reasons.append("state_or_interface 不足")
    if not evidence_ok:
        fail_reasons.append("缺少可核对 evidence")
    if thin:
        fail_reasons.append("thin_source")

    # 模型明确认为证据不足时不能靠字符串长度强制升级。
    if mechanism_is_publishable(
        model_confidence=model_conf,
        steps_ok=steps_ok,
        state_ok=state_ok,
        evidence_ok=evidence_ok,
        thin_source=thin,
    ):
        card["confidence"] = "high"
    else:
        card["confidence"] = "low"
        gaps.extend(fail_reasons)
        if model_conf != "high":
            gaps.append("extractor_not_confident")

    card["gaps"] = gaps
    card.setdefault("failure_modes", [])
    card.setdefault("limits", [])
    card.setdefault("problem", "")
    card.setdefault("inputs_outputs", "")
    card.setdefault("thin_source", False)
    return card
