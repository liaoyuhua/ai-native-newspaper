"""可持久化的信源健康状态与安全运行时覆盖。"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SourceRegistry:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (config.DATA_DIR / "source_registry.json")
        self.data: dict[str, Any] = {"schema_version": "1.0", "sources": {}}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and isinstance(loaded.get("sources"), dict):
                    self.data = loaded
            except (OSError, json.JSONDecodeError):
                # 损坏的运行时状态不能阻断信源抓取；本轮会生成新的健康快照。
                pass

    def effective_config(self, configured: dict) -> dict:
        result = deepcopy(configured)
        state = self.data["sources"].get(configured["name"], {})
        override = state.get("active_override") or {}
        if override.get("validated") is True:
            result["url"] = override.get("url") or result["url"]
            result["fetcher"] = override.get("fetcher") or result.get("fetcher", "rss")
            if override.get("path_prefix"):
                result["path_prefix"] = override["path_prefix"]
        return result

    def should_attempt(self, name: str) -> bool:
        state = self.data["sources"].get(name, {})
        if state.get("status") != "quarantined":
            return True
        raw = state.get("last_failure_at")
        if not raw:
            return True
        try:
            failed_at = datetime.fromisoformat(raw)
            age_hours = (datetime.now(timezone.utc) - failed_at).total_seconds() / 3600
            return age_hours >= config.SOURCE_QUARANTINE_RETRY_HOURS
        except ValueError:
            return True

    def record_success(self, name: str, *, url: str, fetcher: str, item_count: int) -> None:
        state = self._state(name)
        state.update({
            "status": "healthy" if item_count else "stale",
            "consecutive_failures": 0,
            "last_success_at": _now(),
            "last_endpoint": url,
            "last_fetcher": fetcher,
            "last_item_count": item_count,
            "last_error": "",
        })

    def record_failure(self, name: str, *, url: str, error: str) -> None:
        state = self._state(name)
        failures = int(state.get("consecutive_failures") or 0) + 1
        state.update({
            "status": "quarantined" if failures >= config.SOURCE_QUARANTINE_AFTER_FAILURES else "degraded",
            "consecutive_failures": failures,
            "last_failure_at": _now(),
            "last_endpoint": url,
            "last_error": error[:500],
        })

    def set_validated_override(self, name: str, *, url: str, fetcher: str, path_prefix: str = "") -> None:
        state = self._state(name)
        state["active_override"] = {
            "url": url,
            "fetcher": fetcher,
            "path_prefix": path_prefix,
            "validated": True,
            "validated_at": _now(),
        }
        state["status"] = "recovered"

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = _now()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def write_snapshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), ensure_ascii=False, indent=2), encoding="utf-8")

    def summary(self) -> dict:
        sources = self.data.get("sources", {})
        counts: dict[str, int] = {}
        for state in sources.values():
            status = state.get("status", "unknown")
            counts[status] = counts.get(status, 0) + 1
        return {"generated_at": _now(), "counts": counts, "sources": deepcopy(sources)}

    def _state(self, name: str) -> dict:
        return self.data["sources"].setdefault(name, {"status": "unknown", "consecutive_failures": 0})
