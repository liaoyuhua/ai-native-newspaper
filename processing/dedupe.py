"""同批次内的去重：按 URL 归一化 + 标题相似度。"""

from __future__ import annotations

from difflib import SequenceMatcher
from urllib.parse import urlparse, urlunparse

from sources.common import RawItem


def _normalize_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "", ""))
    except Exception:  # noqa: BLE001
        return url


def _title_similar(a: str, b: str, threshold: float = 0.88) -> bool:
    if not a or not b:
        return False
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold


def dedupe_items(items: list[RawItem]) -> list[RawItem]:
    seen_urls: set[str] = set()
    kept: list[RawItem] = []

    for item in items:
        norm_url = _normalize_url(item.url)
        if norm_url in seen_urls:
            continue

        is_dup = False
        for existing in kept:
            if _title_similar(item.title, existing.title):
                is_dup = True
                break

        if is_dup:
            continue

        seen_urls.add(norm_url)
        kept.append(item)

    return kept
