"""
证据库：深度研究 agent 每次真正"拿到一段可核查的原文/摘要"时，都往这里存一条记录。

写作阶段的引用、事实核查阶段的核对，都只认这里存的原文片段——
不认 LLM 自己脑内总结的"印象"，这是保证"每个结论都能回溯到具体来源"的关键。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal


ClaimKind = Literal["fact", "reported_result", "inference", "author_judgment"]


@dataclass
class ClaimRecord:
    id: str
    text: str
    kind: ClaimKind
    evidence_ids: list[str]
    support: str = "unverified"  # supported | partial | unsupported | unverified
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "kind": self.kind,
            "evidence_ids": self.evidence_ids,
            "support": self.support,
            "notes": self.notes,
        }


@dataclass
class Evidence:
    id: str
    document_id: str
    title: str
    url: str
    source_type: str  # "arxiv" | "semantic_scholar" | "webpage" | "web_search_snippet"
    excerpt: str
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "title": self.title,
            "url": self.url,
            "source_type": self.source_type,
            "excerpt": self.excerpt,
            "meta": self.meta,
        }


class EvidenceStore:
    def __init__(self) -> None:
        self._by_id: dict[str, Evidence] = {}
        self._claims: dict[str, ClaimRecord] = {}

    @classmethod
    def from_lists(cls, evidence: list[dict], claims: list[dict] | None = None) -> "EvidenceStore":
        store = cls()
        for raw in evidence:
            eid = str(raw.get("id") or "")
            if not eid:
                continue
            store._by_id[eid] = Evidence(
                id=eid,
                document_id=str(raw.get("document_id") or "doc_legacy"),
                title=str(raw.get("title") or ""),
                url=str(raw.get("url") or ""),
                source_type=str(raw.get("source_type") or "unknown"),
                excerpt=str(raw.get("excerpt") or ""),
                meta=raw.get("meta") or {},
            )
        for raw in claims or []:
            cid = str(raw.get("id") or "")
            if cid:
                store._claims[cid] = ClaimRecord(
                    id=cid,
                    text=str(raw.get("text") or ""),
                    kind=raw.get("kind") or "fact",
                    evidence_ids=[str(x) for x in raw.get("evidence_ids", [])],
                    support=str(raw.get("support") or "unverified"),
                    notes=str(raw.get("notes") or ""),
                )
        return store

    def add(self, title: str, url: str, source_type: str, excerpt: str, meta: dict | None = None) -> str:
        normalized_url = url.strip()
        document_id = "doc_" + hashlib.sha1(normalized_url.encode("utf-8")).hexdigest()[:12]
        # 同一文档的搜索摘要与全文必须是不同证据片段；同类片段则保留信息更丰富的版本。
        identity = f"{normalized_url}|{source_type}"
        eid = "ev_" + hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
        existing = self._by_id.get(eid)
        if existing is None or len(excerpt or "") > len(existing.excerpt or ""):
            self._by_id[eid] = Evidence(
                id=eid, document_id=document_id, title=title, url=normalized_url, source_type=source_type,
                excerpt=excerpt, meta=meta or {},
            )
        return eid

    def add_claim(
        self,
        text: str,
        kind: ClaimKind,
        evidence_ids: list[str] | None = None,
        *,
        support: str = "unverified",
        notes: str = "",
    ) -> str:
        evidence_ids = list(dict.fromkeys(evidence_ids or []))
        cid = "cl_" + hashlib.sha1(f"{kind}|{text}".encode("utf-8")).hexdigest()[:12]
        self._claims[cid] = ClaimRecord(cid, text.strip(), kind, evidence_ids, support, notes)
        return cid

    def get_claim(self, cid: str) -> ClaimRecord | None:
        return self._claims.get(cid)

    def claims(self) -> list[ClaimRecord]:
        return list(self._claims.values())

    def clear_claims(self) -> None:
        self._claims.clear()

    def validate_claim_links(self) -> list[str]:
        errors = []
        for claim in self._claims.values():
            if claim.kind in {"fact", "reported_result"} and not claim.evidence_ids:
                errors.append(f"{claim.id}: verifiable claim has no evidence")
            for eid in claim.evidence_ids:
                if eid not in self._by_id:
                    errors.append(f"{claim.id}: missing evidence {eid}")
        return errors

    def get(self, eid: str) -> Evidence | None:
        return self._by_id.get(eid)

    def all(self) -> list[Evidence]:
        return list(self._by_id.values())

    def to_list(self) -> list[dict]:
        return [e.to_dict() for e in self._by_id.values()]

    def claim_list(self) -> list[dict]:
        return [c.to_dict() for c in self._claims.values()]
