"""通过 arXiv API 抓取近期论文（广度扫描阶段只拿标题+摘要，全文抓取留给深度研究阶段）。

https://info.arxiv.org/help/api/index.html
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import feedparser

import config
from sources.common import RawItem, clean_text

logger = logging.getLogger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"


def fetch_arxiv(lookback_days: int = 8) -> list[RawItem]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    items: list[RawItem] = []
    seen: set[str] = set()

    for category in config.ARXIV_CATEGORIES:
        try:
            items.extend(_fetch_category(category, cutoff, seen))
        except Exception:  # noqa: BLE001
            logger.exception("arXiv 分类抓取失败，跳过: %s", category)

    return items


def _fetch_category(category: str, cutoff: datetime, seen: set[str]) -> list[RawItem]:
    import requests

    resp = requests.get(
        ARXIV_API,
        params={
            "search_query": f"cat:{category}",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": config.ARXIV_MAX_RESULTS_PER_CATEGORY,
        },
        timeout=20,
    )
    resp.raise_for_status()
    feed = feedparser.parse(resp.content)

    out = []
    for entry in feed.entries:
        arxiv_id = entry.get("id", "").split("/abs/")[-1]
        if arxiv_id in seen:
            continue

        published = _parse_time(entry.get("published"))
        if published and published < cutoff:
            continue
        seen.add(arxiv_id)

        authors = [a.get("name", "") for a in entry.get("authors", [])]
        out.append(
            RawItem(
                title=clean_text(entry.get("title", "")),
                url=entry.get("link", f"https://arxiv.org/abs/{arxiv_id}"),
                source=f"arXiv/{category}",
                lang="en",
                authority=0.7,
                published=published.isoformat() if published else None,
                snippet=clean_text(entry.get("summary", ""))[:1000],
                extra={"arxiv_id": arxiv_id, "authors": authors, "category": category},
            )
        )
    return out


def _parse_time(value: str | None):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
