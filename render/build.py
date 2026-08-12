"""把 compose_article() 产出的结构化文章数据渲染成静态 HTML，写入 docs/。"""

from __future__ import annotations

import json
import shutil

import markdown as md
from jinja2 import Environment, FileSystemLoader, select_autoescape

import config

_env = Environment(
    loader=FileSystemLoader(str(config.TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "jinja"]),
)

ARCHIVE_INDEX_PATH = config.ARTICLES_DATA_DIR / "_index.json"


def _base_context(root_prefix: str) -> dict:
    return {
        "site_name": config.SITE_NAME,
        "site_slogan": config.SITE_SLOGAN,
        "site_author": config.SITE_AUTHOR,
        "root_prefix": root_prefix,
        "asset_prefix": root_prefix,
    }


def _markdown_sections(article: dict) -> dict:
    rendered = dict(article)
    rendered["sections"] = [
        {"heading": s["heading"], "html": md.markdown(s["text"], extensions=["extra", "sane_lists"])}
        for s in article["sections"]
    ]
    return rendered


def build_article_page(article: dict, week_id: str, week_label: str, publish_date: str) -> None:
    article_for_render = _markdown_sections(article)

    # 1) 归档路径 docs/articles/<week_id>/index.html
    article_dir = config.DOCS_DIR / "articles" / week_id
    article_dir.mkdir(parents=True, exist_ok=True)
    ctx = {
        **_base_context("../../"),
        "article": article_for_render,
        "week_label": week_label,
        "publish_date": publish_date,
    }
    (article_dir / "index.html").write_text(
        _env.get_template("article.html.jinja").render(**ctx), encoding="utf-8"
    )

    # 2) 最新一期同步到 docs/index.html
    ctx_latest = {**_base_context(""), "article": article_for_render, "week_label": week_label, "publish_date": publish_date}
    (config.DOCS_DIR / "index.html").write_text(
        _env.get_template("article.html.jinja").render(**ctx_latest), encoding="utf-8"
    )

    _sync_style()
    _update_archive_index(week_id, week_label, article["title"], article.get("subtitle", ""), publish_date)
    _build_archive_page()


def _sync_style() -> None:
    src = config.TEMPLATES_DIR / "style.css"
    shutil.copy(src, config.DOCS_DIR / "style.css")
    assets_dir = config.TEMPLATES_DIR / "assets"
    for name in ("logo.png", "favicon.png"):
        src_asset = assets_dir / name
        if src_asset.exists():
            shutil.copy(src_asset, config.DOCS_DIR / name)


def _update_archive_index(week_id: str, week_label: str, title: str, subtitle: str, publish_date: str) -> None:
    issues = _load_archive_index()
    issues = [i for i in issues if i["week_id"] != week_id]
    issues.append(
        {"week_id": week_id, "week_label": week_label, "title": title, "subtitle": subtitle, "date": publish_date}
    )
    issues.sort(key=lambda i: i["week_id"], reverse=True)
    ARCHIVE_INDEX_PATH.write_text(json.dumps(issues, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_archive_index() -> list[dict]:
    if ARCHIVE_INDEX_PATH.exists():
        return json.loads(ARCHIVE_INDEX_PATH.read_text(encoding="utf-8"))
    return []


def _build_archive_page() -> None:
    issues = _load_archive_index()
    ctx = {**_base_context(""), "issues": issues}
    (config.DOCS_DIR / "archive.html").write_text(
        _env.get_template("archive.html.jinja").render(**ctx), encoding="utf-8"
    )
