"""终审阶段的可恢复状态。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REVIEW_CHECKPOINT_VERSION = 3


def draft_fingerprint(title: str, thesis: str, sections: list[dict]) -> str:
    payload = json.dumps(
        {"title": title, "thesis": thesis, "sections": sections},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_review_checkpoint(path: Path | None, fingerprint: str) -> dict:
    if path is None or not path.exists():
        return {"checkpoint_version": REVIEW_CHECKPOINT_VERSION, "draft_fingerprint": fingerprint}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"checkpoint_version": REVIEW_CHECKPOINT_VERSION, "draft_fingerprint": fingerprint}
    if (
        payload.get("checkpoint_version") != REVIEW_CHECKPOINT_VERSION
        or payload.get("draft_fingerprint") != fingerprint
    ):
        return {"checkpoint_version": REVIEW_CHECKPOINT_VERSION, "draft_fingerprint": fingerprint}
    return payload


def write_review_checkpoint(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
