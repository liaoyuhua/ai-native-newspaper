"""周编号相关的小工具，统一 week_id 的格式，避免各处写法不一致。"""

from __future__ import annotations

from datetime import date, datetime


def current_week_id(today: date | None = None) -> str:
    today = today or date.today()
    year, week, _ = today.isocalendar()
    return f"{year}-W{week:02d}"


def week_label(week_id: str) -> str:
    year, week = week_id.split("-W")
    return f"{year} 年第 {int(week)} 周"


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")
