"""抓取 config.RSS_SOURCES 里配置的官方博客 / 科技媒体。

支持两类抓取：
- fetcher=rss（默认）：标准 RSS/Atom
- fetcher=html_list：没有官方 RSS 时，从列表页解析文章链接（目前用于 Anthropic News）

单个信源抓取失败(网络问题、没有 RSS、格式变化等)只记录日志并跳过，
不能让一个源挂掉拖垮整条流水线——广度扫描本来就是"宁可漏一个源，不能因为一个源挂了整体失败"。
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin, urlparse

import feedparser
import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

import config
from sources.common import RawItem, clean_text
from sources.registry import SourceRegistry

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "ai-native-newspaper/1.0"}
_DATE_RE = re.compile(
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}"
)
_CATEGORY_RE = re.compile(
    r"^(Product|Announcements|Research|Policy|Company|Engineering|News|Economic Research)\s+",
    re.I,
)
_BARE_LABELS = {
    "product",
    "announcements",
    "research",
    "policy",
    "company",
    "engineering",
    "news",
    "economic research",
}


def fetch_rss(lookback_days: int = 8) -> list[RawItem]:
    items: list[RawItem] = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    registry = SourceRegistry()

    for configured in config.RSS_SOURCES:
        if not configured.get("enabled", True):
            logger.info("信源已关闭，跳过: %s", configured["name"])
            continue
        if not registry.should_attempt(configured["name"]):
            logger.warning("信源处于隔离冷却期，跳过: %s", configured["name"])
            continue
        src = registry.effective_config(configured)
        try:
            fetcher = src.get("fetcher", "rss")
            if fetcher == "html_list":
                fetched = _fetch_html_list(src, cutoff)
            else:
                fetched = _fetch_rss_one(src, cutoff)
            items.extend(fetched)
            registry.record_success(src["name"], url=src["url"], fetcher=fetcher, item_count=len(fetched))
        except Exception as exc:  # noqa: BLE001
            logger.warning("信源抓取失败，尝试同域自愈: %s (%s)", src["name"], exc)
            recovered = _attempt_recovery(configured, cutoff, registry)
            if recovered is not None:
                items.extend(recovered)
            else:
                registry.record_failure(src["name"], url=src["url"], error=str(exc))

    registry.save()
    day = datetime.now(timezone.utc).date().isoformat()
    registry.write_snapshot(config.SOURCE_HEALTH_DIR / f"{day}.json")

    return items


def _attempt_recovery(src: dict, cutoff: datetime, registry: SourceRegistry) -> list[RawItem] | None:
    discovered = _discover_official_feed(src)
    if discovered:
        candidate = {**src, "url": discovered, "fetcher": "rss"}
        try:
            fetched = _fetch_rss_one(candidate, cutoff)
            if not fetched:
                raise ValueError("发现的 feed 没有近期条目")
            registry.set_validated_override(src["name"], url=discovered, fetcher="rss")
            registry.record_success(src["name"], url=discovered, fetcher="rss", item_count=len(fetched))
            logger.info("信源已自动切换到同域 feed: %s -> %s", src["name"], discovered)
            return fetched
        except Exception as exc:  # noqa: BLE001
            logger.warning("发现的 feed 验证失败: %s (%s)", discovered, exc)

    if src.get("fallback_fetcher") == "html_list" and src.get("homepage"):
        fallback = {**src, "url": src["homepage"], "fetcher": "html_list"}
        try:
            fetched = _fetch_html_list(fallback, cutoff)
            if not fetched:
                raise ValueError("HTML 列表未解析出近期文章")
            registry.set_validated_override(
                src["name"], url=fallback["url"], fetcher="html_list",
                path_prefix=fallback.get("path_prefix", ""),
            )
            registry.record_success(
                src["name"], url=fallback["url"], fetcher="html_list", item_count=len(fetched)
            )
            logger.info("信源已安全降级为官网文章列表: %s", src["name"])
            return fetched
        except Exception as exc:  # noqa: BLE001
            logger.warning("HTML 列表降级失败: %s (%s)", src["name"], exc)
    return None


def _discover_official_feed(src: dict) -> str | None:
    homepage = src.get("homepage") or _origin(src.get("url", ""))
    if not homepage:
        return None
    try:
        resp = requests.get(homepage, timeout=12, headers=_HEADERS)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "lxml")
    except Exception:  # noqa: BLE001
        return None
    source_host = _normalized_host(homepage)
    for link in soup.select('link[rel~="alternate"][href]'):
        mime = (link.get("type") or "").lower()
        if "rss" not in mime and "atom" not in mime and "xml" not in mime:
            continue
        candidate = urljoin(homepage, link.get("href", ""))
        if _normalized_host(candidate) == source_host:
            return candidate
    return None


def _origin(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}/" if parsed.scheme and parsed.netloc else ""


def _normalized_host(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def _fetch_rss_one(src: dict, cutoff: datetime) -> list[RawItem]:
    # 用 requests 带超时抓取，避免 feedparser 默认卡住很久
    resp = requests.get(src["url"], timeout=12, headers=_HEADERS)
    resp.raise_for_status()
    feed = feedparser.parse(resp.content)
    if getattr(feed, "bozo", 0) and not feed.entries:
        logger.warning(
            "RSS 源解析异常且无条目: %s (%s)",
            src["name"],
            getattr(feed, "bozo_exception", ""),
        )

    out = []
    for entry in feed.entries:
        published = _entry_published(entry)
        if published and published < cutoff:
            continue

        snippet = clean_text(getattr(entry, "summary", "") or getattr(entry, "description", ""))[:800]
        out.append(
            RawItem(
                title=clean_text(getattr(entry, "title", "")),
                url=getattr(entry, "link", ""),
                source=src["name"],
                lang=src.get("lang", "en"),
                authority=src.get("authority", 0.5),
                published=published.isoformat() if published else None,
                snippet=snippet,
            )
        )
        if len(out) >= 30:
            break
    return out


def _fetch_html_list(src: dict, cutoff: datetime) -> list[RawItem]:
    """从新闻列表页抓取文章链接。Anthropic 等站点无官方 RSS 时使用。"""
    resp = requests.get(src["url"], timeout=15, headers=_HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "lxml")

    path_prefix = src.get("path_prefix", "/news/")
    out: list[RawItem] = []
    seen: set[str] = set()

    for a in soup.select(f'a[href*="{path_prefix}"]'):
        href = (a.get("href") or "").split("?")[0].strip()
        if not href:
            continue
        url = urljoin(src["url"], href).rstrip("/")
        if url in seen:
            continue
        # 只要具体文章页，不要列表根路径或过浅路径
        path = urlparse(url).path
        if not path.startswith(path_prefix):
            continue
        slug = path[len(path_prefix) :].strip("/")
        if not slug or "/" in slug:
            continue
        if slug.lower() in {"tag", "product", "announcements", "research", "policy"}:
            continue
        seen.add(url)

        text = a.get_text(" ", strip=True)
        published = _parse_date_from_text(text)
        if published and published < cutoff:
            continue

        title = _title_from_list_text(text, slug)
        if not title:
            continue

        out.append(
            RawItem(
                title=title,
                url=url,
                source=src["name"],
                lang=src.get("lang", "en"),
                authority=src.get("authority", 0.5),
                published=published.isoformat() if published else None,
                snippet=clean_text(text)[:800],
            )
        )
        if len(out) >= 30:
            break

    logger.info("HTML 列表抓取 %s: %d 条", src["name"], len(out))
    return out


def _humanize_slug(slug: str) -> str:
    s = slug.replace("-", " ").strip()
    s = re.sub(r"\b(\d+) s\b", r"\1's", s)
    # 保留常见专有大小写，其余词首大写
    return " ".join(w.upper() if w.isupper() else w.capitalize() for w in s.split())


def _title_from_list_text(text: str, slug: str) -> str:
    text = clean_text(text)
    m = _DATE_RE.search(text)
    if m:
        before = re.sub(
            r"\s+(Product|Announcements|Research|Policy|Company|Engineering|News|Economic Research)\s*$",
            "",
            text[: m.start()].strip(),
            flags=re.I,
        )
        after = _CATEGORY_RE.sub("", text[m.end() :].strip())
        after = re.split(r"(?<=[.!?])\s+", after, maxsplit=1)[0].strip()
        # 日期在中间时，前面通常是标题；但要排除栏目名误当标题。
        if len(before) >= 8 and before.lower() not in _BARE_LABELS:
            return before
        if len(after) >= 8 and after.lower() not in _BARE_LABELS:
            # 导语粘在标题后且很长时，退回用 slug（Anthropic URL 很稳定）
            if len(after) > 90:
                return _humanize_slug(slug)
            return after
    title = _CATEGORY_RE.sub("", text).strip()
    if len(title) >= 8 and title.lower() not in _BARE_LABELS:
        return title
    return _humanize_slug(slug)


def _parse_date_from_text(text: str) -> datetime | None:
    m = _DATE_RE.search(text or "")
    if not m:
        return None
    try:
        dt = date_parser.parse(m.group(0))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, OverflowError, TypeError):
        return None


def _entry_published(entry) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        val = getattr(entry, key, None)
        if val:
            return datetime(*val[:6], tzinfo=timezone.utc)
    return None
