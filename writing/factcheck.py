"""
事实核查 + 修正循环：把草稿里每一个带【EV:id】标记的论断，拿去跟 evidence 原文核对，
不支持/部分支持的论断打回去让模型删除或改成更保守的表述，循环若干轮。

这一步是"严谨可回溯"要求的最后一道关卡：宁可文章少几个具体细节，也不能有编造的引用。
"""

from __future__ import annotations

import json
import logging
import re

import config
from processing.llm_client import chat_json, chat_text
from research.evidence import EvidenceStore

logger = logging.getLogger(__name__)

MARKER_RE = re.compile(r"【EV:([^】]+)】")

VERIFY_SYSTEM_PROMPT = """\
你是一个极其严格的事实核查员。你会看到若干"论断 + 它引用的原始素材片段"配对。
请逐条判断这条论断是否被该素材片段真正支持。

判定标准：
- "supported": 素材明确包含/直接支持这条论断里的具体事实
- "partial": 素材只支持论断的一部分，或论断有一定程度的过度引申/夸大
- "unsupported": 素材完全不支持，或论断包含素材里没有的具体数字/结论(疑似编造)

只返回 JSON: {"verdicts": [{"index": 0, "verdict": "supported|partial|unsupported", "reason": "简短原因"}]}
"""

REVISE_SYSTEM_PROMPT = """\
你是原文的作者，正在根据事实核查反馈修改这一节的正文(中文，Markdown)。
下面会给你原始正文，以及"哪些具体论断被核查为不可靠"及原因。

请修改正文：
- 对于被标记为 unsupported 的论断：删除这句话里编造的具体细节，改写成更保守/概括的表述，
  或者直接删掉这句话(如果删掉不影响上下文，优先删掉)，并去掉对应的【EV:...】标记。
- 对于被标记为 partial 的论断：软化措辞(比如加上"据报道"、"初步显示"等)，让论断的确定程度
  跟素材实际支持的程度匹配，标记可以保留。
- 其余没有被提到的内容不要改动。
- 保持原有的 Markdown 格式和整体结构、篇幅，不要重写整节。
- 直接输出修改后的完整正文，不要输出任何解释。
"""


def verify_and_revise_section(section_text: str, evidence_store: EvidenceStore) -> str:
    text = section_text
    for round_idx in range(config.FACTCHECK_MAX_REVISION_ROUNDS):
        claims = _extract_claims(text)
        if not claims:
            break

        verdicts = _verify_claims(claims, evidence_store)
        problems = [
            (claims[v["index"]], v)
            for v in verdicts
            if v.get("verdict") in ("unsupported", "partial") and 0 <= v.get("index", -1) < len(claims)
        ]

        if not problems:
            logger.info("第 %d 轮核查：全部论断通过", round_idx + 1)
            break

        logger.info("第 %d 轮核查：发现 %d 条问题论断，进行修正", round_idx + 1, len(problems))
        text = _revise(text, problems)

    return _strip_unresolved_markers(text, evidence_store)


def _extract_claims(text: str) -> list[dict]:
    claims = []
    cursor = 0
    for m in MARKER_RE.finditer(text):
        start_context = max(0, m.start() - 300)
        claim_sentence = text[start_context : m.start()].split("。")[-1].strip() or text[start_context:m.start()]
        eids = [e.strip() for e in m.group(1).split(",") if e.strip()]
        claims.append({"claim": claim_sentence, "evidence_ids": eids, "marker_span": (m.start(), m.end())})
        cursor = m.end()
    return claims


def _verify_claims(claims: list[dict], evidence_store: EvidenceStore) -> list[dict]:
    pairs = []
    for idx, c in enumerate(claims):
        excerpts = []
        for eid in c["evidence_ids"]:
            ev = evidence_store.get(eid)
            if ev:
                excerpts.append(f"[{eid}] {ev.excerpt[:1500]}")
        pairs.append({"index": idx, "claim": c["claim"], "evidence_excerpts": excerpts})

    user_prompt = "论断与素材配对列表(JSON):\n" + json.dumps(pairs, ensure_ascii=False)
    result = chat_json(
        model=config.MODEL_FACTCHECK,
        system_prompt=VERIFY_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.0,
    )
    return result.get("verdicts", [])


def _revise(text: str, problems: list[tuple[dict, dict]]) -> str:
    problem_desc = "\n".join(
        f"- 论断: \"{claim['claim']}\" | 判定: {verdict['verdict']} | 原因: {verdict.get('reason', '')}"
        for claim, verdict in problems
    )
    user_prompt = f"原始正文:\n{text}\n\n核查反馈:\n{problem_desc}\n"
    return chat_text(
        model=config.MODEL_EDITORIAL,
        system_prompt=REVISE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.3,
    )


def _strip_unresolved_markers(text: str, evidence_store: EvidenceStore) -> str:
    """安全网：修正轮数用完后，如果还有标记引用了不存在的 evidence_id，直接去掉标记，避免正文里出现死链引用。"""

    def _sub(m: re.Match) -> str:
        eids = [e.strip() for e in m.group(1).split(",") if e.strip()]
        valid = [e for e in eids if evidence_store.get(e)]
        return f"【EV:{','.join(valid)}】" if valid else ""

    return MARKER_RE.sub(_sub, text)
