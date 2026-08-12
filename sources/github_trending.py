"""抓取 GitHub Trending（周榜），作为"开源侧热度"信号。

GitHub 没有官方 trending API，这里直接解析 trending 页面 HTML。
页面结构一旦大改会导致抓取失败，做好容错（抓不到就跳过，不影响其它信源）。
"""

from __future__ import annotations

import logging

import requests
from bs4 import BeautifulSoup

import config
from sources.common import RawItem, clean_text

logger = logging.getLogger(__name__)

TRENDING_URL = "https://github.com/trending/{language}"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ai-native-newspaper-bot/1.0)"}


def fetch_github_trending() -> list[RawItem]:
    items: list[RawItem] = []
    for lang in config.GITHUB_TRENDING_LANGUAGES:
        try:
            items.extend(_fetch_language(lang))
        except Exception:  # noqa: BLE001
            logger.exception("GitHub Trending 抓取失败，跳过语言: %s", lang)
    return items


def _fetch_language(language: str) -> list[RawItem]:
    url = TRENDING_URL.format(language=language)
    resp = requests.get(url, params={"since": "weekly"}, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    out = []
    for article in soup.select("article.Box-row"):
        link = article.select_one("h2 a")
        if not link:
            continue
        repo_path = link.get("href", "").strip("/")
        if not repo_path:
            continue

        desc_el = article.select_one("p")
        description = clean_text(desc_el.get_text()) if desc_el else ""

        stars_el = article.select_one('a[href$="/stargazers"]')
        stars_period_el = article.select_one("span.d-inline-block.float-sm-right")

        stars_this_week = _parse_int(stars_period_el.get_text() if stars_period_el else "")

        out.append(
            RawItem(
                title=repo_path,
                url=f"https://github.com/{repo_path}",
                source="GitHub Trending",
                lang="en",
                authority=0.55,
                snippet=description[:500],
                buzz=float(stars_this_week),
                extra={"language": language, "stars_this_week": stars_this_week},
            )
        )
    return out


def _parse_int(text: str) -> int:
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else 0
