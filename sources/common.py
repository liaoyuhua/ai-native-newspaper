"""所有信源抓取模块共用的数据结构与小工具。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RawItem:
    """一条抓到的原始条目，字段尽量统一，方便后续聚类/打分。"""

    title: str
    url: str
    source: str
    lang: str = "en"
    authority: float = 0.5
    published: str | None = None  # ISO8601 字符串
    snippet: str = ""
    buzz: float = 0.0  # 讨论热度信号（HN points、star 数等，各源量级不同，仅供粗略参考）
    extra: dict = field(default_factory=dict)

    @property
    def item_id(self) -> str:
        return hashlib.sha1(self.url.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {
            "id": self.item_id,
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "lang": self.lang,
            "authority": self.authority,
            "published": self.published,
            "snippet": self.snippet,
            "buzz": self.buzz,
            "extra": self.extra,
        }


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def now_iso() -> str:
    return datetime.utcnow().isoformat()
