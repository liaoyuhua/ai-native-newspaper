"""记录每次流水线运行的输入、版本、阶段和失败原因。"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RunManifest:
    run_id: str
    workflow: str
    status: str = "running"
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None
    python_version: str = field(default_factory=platform.python_version)
    config: dict[str, Any] = field(default_factory=dict)
    stages: list[dict[str, Any]] = field(default_factory=list)
    decision: dict[str, Any] = field(default_factory=dict)

    def stage(self, name: str, **details: Any) -> None:
        self.stages.append({"name": name, "at": datetime.now(timezone.utc).isoformat(), **details})

    def finish(self, status: str, **decision: Any) -> None:
        self.status = status
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.decision = decision

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
