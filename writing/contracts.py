"""不依赖模型 SDK 的写作数据契约。"""

from __future__ import annotations


def validate_concept_dependencies(outline_sections: list[dict], reader_context: dict) -> list[str]:
    """确定性检查概念是否在依赖它的章节之前被引入。"""
    introduced = {"LLM", "AI", "软件", "文件", "网络", "插件"}
    problems: list[str] = []
    required = {
        str(p.get("concept") or "").strip()
        for p in (reader_context or {}).get("prerequisites", [])
        if isinstance(p, dict) and not p.get("reader_likely_knows") and str(p.get("concept") or "").strip()
    }
    for section in outline_sections:
        heading = str(section.get("heading") or "未命名小节")
        for concept in section.get("assumes") or []:
            concept = str(concept).strip()
            if concept in required and concept not in introduced:
                problems.append(f"{heading} 在解释前假设读者理解 {concept}")
        introduced.update(str(x).strip() for x in (section.get("introduces") or []) if str(x).strip())
    missing = sorted(required - introduced)
    problems.extend(f"大纲未安排解释必要概念 {concept}" for concept in missing)
    return problems


def align_sections(raw_sections: list[dict], outline_sections: list[dict]) -> list[dict]:
    """模型不得通过漏掉机制节来逃避大纲契约。"""
    aligned = []
    for expected in outline_sections:
        role = expected.get("role", "")
        card_index = expected.get("card_index")
        match = next(
            (
                s for s in raw_sections
                if s.get("role") == role
                and (role != "mechanism" or s.get("card_index") == card_index)
            ),
            None,
        )
        if not match or not str(match.get("text") or "").strip():
            raise ValueError(f"整篇初稿缺少必要小节: role={role}, card_index={card_index}")
        aligned.append({
            "heading": str(match.get("heading") or expected.get("heading") or "").strip(),
            "role": role,
            "card_index": card_index,
            "text": str(match.get("text") or "").strip(),
        })
    return aligned
