"""通过 Algolia 的 Hacker News Search API 抓取 AI 相关讨论。

https://hn.algolia.com/api - 免费、无需 Key。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import requests

import config
from sources.common import RawItem, clean_text

logger = logging.getLogger(__name__)

HN_API = "https://hn.algolia.com/api/v1/search_by_date"


def fetch_hackernews(lookback_days: int = 8) -> list[RawItem]:
    cutoff_ts = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp())
    seen_ids: set[str] = set()
    items: list[RawItem] = []

    for keyword in config.HN_KEYWORDS:
        try:
            items.extend(_search(keyword, cutoff_ts, seen_ids))
        except Exception:  # noqa: BLE001
            logger.exception("Hacker News 关键词抓取失败，跳过: %s", keyword)

    return items


def _search(keyword: str, cutoff_ts: int, seen_ids: set[str]) -> list[RawItem]:
    resp = requests.get(
        HN_API,
        params={
            "query": keyword,
            "tags": "story",
            "numericFilters": f"created_at_i>{cutoff_ts},points>={config.HN_MIN_POINTS}",
            "hitsPerPage": 50,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    out = []
    for hit in data.get("hits", []):
        hn_id = hit.get("objectID")
        if not hn_id or hn_id in seen_ids:
            continue
        seen_ids.add(hn_id)

        url = hit.get("url") or f"https://news.ycombinator.com/item?id={hn_id}"
        points = hit.get("points", 0) or 0
        comments = hit.get("num_comments", 0) or 0

        out.append(
            RawItem(
                title=clean_text(hit.get("title", "")),
                url=url,
                source="Hacker News",
                lang="en",
                authority=0.5,
                published=hit.get("created_at"),
                snippet=clean_text(hit.get("story_text") or "")[:800],
                buzz=float(points + comments * 2),
                extra={
                    "hn_id": hn_id,
                    "points": points,
                    "comments": comments,
                    "hn_discussion_url": f"https://news.ycombinator.com/item?id={hn_id}",
                },
            )
        )
    return out
