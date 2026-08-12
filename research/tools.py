"""
深度研究 agent 可以调用的工具：搜 arXiv、搜 Semantic Scholar(含引用图)、抓全文、兜底网页搜索。

每个工具在拿到结果时，会把"标题+可核查的原文片段"写入 EvidenceStore，
返回给模型的是精简摘要(含 evidence_id)，模型引用时只需要引用 evidence_id，
后续写作/事实核查阶段再用 evidence_id 去 EvidenceStore 里取真正的原文核对。
"""

from __future__ import annotations

import logging
import re

import requests
from bs4 import BeautifulSoup

import config
from research.evidence import EvidenceStore

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ai-native-newspaper-research-bot/1.0)"}

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


class ResearchTools:
    def __init__(self, evidence_store: EvidenceStore) -> None:
        self.evidence = evidence_store

    # ------------------------------------------------------------------
    # arXiv
    # ------------------------------------------------------------------
    def search_arxiv(self, query: str, max_results: int = None) -> dict:
        import feedparser

        max_results = max_results or config.RESEARCH_MAX_RESULTS_PER_SEARCH
        resp = requests.get(
            "http://export.arxiv.org/api/query",
            params={
                "search_query": f"all:{query}",
                "sortBy": "relevance",
                "sortOrder": "descending",
                "max_results": max_results,
            },
            timeout=20,
        )
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)

        results = []
        for entry in feed.entries:
            arxiv_id = entry.get("id", "").split("/abs/")[-1]
            title = _clean(entry.get("title", ""))
            abstract = _clean(entry.get("summary", ""))
            url = entry.get("link", f"https://arxiv.org/abs/{arxiv_id}")
            authors = [a.get("name", "") for a in entry.get("authors", [])]

            eid = self.evidence.add(
                title=title, url=url, source_type="arxiv", excerpt=abstract,
                meta={"arxiv_id": arxiv_id, "authors": authors, "published": entry.get("published")},
            )
            results.append(
                {
                    "evidence_id": eid,
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "authors": authors,
                    "abstract": abstract[:500],
                    "url": url,
                }
            )
        return {"results": results}

    # ------------------------------------------------------------------
    # Semantic Scholar
    # ------------------------------------------------------------------
    def search_semantic_scholar(self, query: str, max_results: int = None) -> dict:
        max_results = max_results or config.RESEARCH_MAX_RESULTS_PER_SEARCH
        headers = dict(HEADERS)
        if config.SEMANTIC_SCHOLAR_API_KEY:
            headers["x-api-key"] = config.SEMANTIC_SCHOLAR_API_KEY

        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "limit": max_results,
                "fields": "title,abstract,year,authors,citationCount,url,externalIds",
            },
            headers=headers,
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for paper in data.get("data", []):
            title = paper.get("title") or ""
            abstract = paper.get("abstract") or ""
            url = paper.get("url") or ""
            paper_id = paper.get("paperId", "")

            eid = self.evidence.add(
                title=title, url=url or f"https://api.semanticscholar.org/{paper_id}",
                source_type="semantic_scholar", excerpt=abstract,
                meta={
                    "paper_id": paper_id,
                    "year": paper.get("year"),
                    "citation_count": paper.get("citationCount"),
                    "authors": [a.get("name") for a in paper.get("authors", [])],
                },
            )
            results.append(
                {
                    "evidence_id": eid,
                    "paper_id": paper_id,
                    "title": title,
                    "year": paper.get("year"),
                    "citation_count": paper.get("citationCount"),
                    "abstract": abstract[:500],
                    "url": url,
                }
            )
        return {"results": results}

    def get_semantic_scholar_related(self, paper_id: str, direction: str = "citations") -> dict:
        """direction: 'citations'(谁引用了它) 或 'references'(它引用了谁)，用于沿着引用图追溯关键工作。"""
        headers = dict(HEADERS)
        if config.SEMANTIC_SCHOLAR_API_KEY:
            headers["x-api-key"] = config.SEMANTIC_SCHOLAR_API_KEY

        field_name = "citations" if direction == "citations" else "references"
        resp = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}",
            params={"fields": f"{field_name}.title,{field_name}.abstract,{field_name}.url,{field_name}.year,{field_name}.paperId"},
            headers=headers,
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for paper in data.get(field_name, [])[:20]:
            if not paper or not paper.get("title"):
                continue
            title = paper.get("title", "")
            abstract = paper.get("abstract") or ""
            url = paper.get("url") or ""
            eid = self.evidence.add(
                title=title, url=url or f"https://api.semanticscholar.org/{paper.get('paperId', '')}",
                source_type="semantic_scholar", excerpt=abstract,
                meta={"paper_id": paper.get("paperId"), "year": paper.get("year")},
            )
            results.append({"evidence_id": eid, "title": title, "year": paper.get("year"), "url": url})
        return {"results": results}

    # ------------------------------------------------------------------
    # 全文抓取
    # ------------------------------------------------------------------
    def fetch_fulltext(self, url_or_arxiv_id: str) -> dict:
        looks_like_bare_id = "/" not in url_or_arxiv_id and ARXIV_ID_RE.search(url_or_arxiv_id)
        is_arxiv_url = "arxiv.org" in url_or_arxiv_id.lower()

        if looks_like_bare_id or is_arxiv_url:
            match = ARXIV_ID_RE.search(url_or_arxiv_id)
            if match:
                return self._fetch_arxiv_fulltext(match.group(1))

        return self._fetch_generic_page(url_or_arxiv_id)

    def _fetch_arxiv_fulltext(self, arxiv_id: str) -> dict:
        errors = []
        for html_url in (
            f"https://arxiv.org/html/{arxiv_id}",
            f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}",
        ):
            try:
                resp = requests.get(html_url, headers=HEADERS, timeout=25)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "lxml")
                article = soup.select_one("article") or soup.body
                text = _clean(article.get_text(" ")) if article else ""
                if len(text) < 3000:
                    raise ValueError("HTML 正文过短")
                title_el = soup.select_one("h1")
                title = _clean(title_el.get_text()) if title_el else arxiv_id
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{html_url}: {exc}")
        else:
            logger.warning("arXiv HTML 全文抓取失败，回退到 abstract page: %s (%s)", arxiv_id, "; ".join(errors))
            return self._fetch_generic_page(
                f"https://arxiv.org/abs/{arxiv_id}", source_type="arxiv_abstract_page"
            )

        url = f"https://arxiv.org/abs/{arxiv_id}"
        excerpt = text[:12000]
        eid = self.evidence.add(title=title, url=url, source_type="arxiv_fulltext", excerpt=excerpt, meta={"arxiv_id": arxiv_id})
        return {"evidence_id": eid, "title": title, "url": url, "fulltext_excerpt": excerpt[:4000], "length_chars": len(text)}

    def _fetch_generic_page(self, url: str, source_type: str = "webpage") -> dict:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            title = _clean(soup.title.get_text()) if soup.title else url
            paragraphs = [p.get_text(" ") for p in soup.find_all(["p", "li"])]
            text = _clean(" ".join(paragraphs)) or _clean(soup.get_text(" "))
        except Exception as exc:  # noqa: BLE001
            return {"error": f"抓取失败: {exc}"}

        excerpt = text[:12000]
        eid = self.evidence.add(title=title, url=url, source_type=source_type, excerpt=excerpt)
        return {"evidence_id": eid, "title": title, "url": url, "fulltext_excerpt": excerpt[:4000], "length_chars": len(text)}

    # ------------------------------------------------------------------
    # 兜底通用搜索(DuckDuckGo HTML，无需 API Key，best-effort)
    # ------------------------------------------------------------------
    def web_search(self, query: str, max_results: int = 5) -> dict:
        try:
            resp = requests.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                headers=HEADERS,
                timeout=20,
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
        except Exception as exc:  # noqa: BLE001
            return {"error": f"网页搜索失败: {exc}", "results": []}

        results = []
        for res in soup.select(".result")[:max_results]:
            link_el = res.select_one("a.result__a")
            snippet_el = res.select_one(".result__snippet")
            if not link_el:
                continue
            title = _clean(link_el.get_text())
            url = link_el.get("href", "")
            snippet = _clean(snippet_el.get_text()) if snippet_el else ""

            eid = self.evidence.add(title=title, url=url, source_type="web_search_snippet", excerpt=snippet)
            results.append({"evidence_id": eid, "title": title, "url": url, "snippet": snippet})

        return {"results": results}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_arxiv",
            "description": "按关键词搜索 arXiv 论文，返回标题/作者/摘要/链接。用于寻找某方向的论文。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词，英文效果更好"},
                    "max_results": {"type": "integer", "description": "返回条数，默认10"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_semantic_scholar",
            "description": "按关键词搜索 Semantic Scholar，覆盖比 arXiv 更广(含已发表论文)，并带引用数，可用来判断一篇论文的影响力。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_semantic_scholar_related",
            "description": "给定一篇论文的 Semantic Scholar paper_id，获取它引用的工作(references)或引用它的工作(citations)，用于沿着引用图追溯这个方向的关键前置/后续工作。",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string"},
                    "direction": {"type": "string", "enum": ["citations", "references"]},
                },
                "required": ["paper_id", "direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_fulltext",
            "description": "抓取一篇论文(给 arXiv id 或 arxiv.org 链接)或一篇网页/博客(给 URL)的正文全文，用于核实细节、获取摘要之外的具体内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url_or_arxiv_id": {"type": "string", "description": "arXiv id(如 2401.12345)、arxiv.org 链接，或任意网页 URL"},
                },
                "required": ["url_or_arxiv_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "通用网页搜索，兜底用于查找论文/学术数据库覆盖不到的信息(比如某个项目的官方公告、社区讨论)。结果可能不如专用学术搜索精确。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
]


def build_tool_impls(tools: ResearchTools) -> dict:
    return {
        "search_arxiv": tools.search_arxiv,
        "search_semantic_scholar": tools.search_semantic_scholar,
        "get_semantic_scholar_related": tools.get_semantic_scholar_related,
        "fetch_fulltext": tools.fetch_fulltext,
        "web_search": tools.web_search,
    }
