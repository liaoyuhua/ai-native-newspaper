"""
工作流 B：人工审核提案后触发。
读取提案 -> 深度研究 -> 机制卡助写 -> 短深文撰写/核查 -> 发布；素材不足则跳过发文。
"""

from __future__ import annotations

import argparse
import json
import logging

import config
from memory.topic_memory import distill_and_upsert, load_topic, match_topic_memory
from processing.run_manifest import RunManifest
from render.build import build_article_page
from research.agent import run_research
from research.evidence import EvidenceStore
from research.mechanism_cards import extract_and_attach_mechanism_cards
from weekutil import current_week_id, today_str, week_label
from writing.compose import compose_article
from writing.quality_gate import ArticleQualityError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--week-id", default=None, help="对应 data/proposals/<week_id>.json，默认用当前周")
    parser.add_argument("--resume-research", action="store_true", help="复用已落盘研究档案，只重跑写作与发布")
    parser.add_argument("--resume-draft", action="store_true", help="复用已完成逐节核查的 draft，只重跑文章终审")
    args = parser.parse_args()

    week_id = args.week_id or current_week_id()
    manifest = RunManifest(run_id=f"{week_id}-publish", workflow="weekly_publish_v2")
    proposal_path = config.PROPOSALS_DIR / f"{week_id}.json"
    if not proposal_path.exists():
        raise SystemExit(f"找不到提案文件: {proposal_path}，先运行 main_scan.py 或检查 week_id 是否正确")

    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    if proposal.get("publish_recommendation") == "skip":
        manifest.finish("skipped", reason=proposal.get("reason", "proposal_rejected"))
        manifest.write(config.RUNS_DIR / f"{week_id}-publish.json")
        logger.warning("该周提案未越过选题质量门槛，本次发布正常跳过")
        return
    topic = proposal["selected_topic"]
    manifest.stage("proposal_loaded", question=topic["name"], schema_version=proposal.get("schema_version", "1.0"))
    logger.info("加载提案，话题: %s", topic["name"])

    topic_memory = None
    memory_ctx = proposal.get("topic_memory_context") or {}
    related_id = topic.get("related_topic_id") or memory_ctx.get("topic_id") or ""
    if related_id:
        topic_memory = load_topic(related_id)
    if topic_memory is None:
        topic_memory = match_topic_memory(topic["name"], topic.get("description", ""))
    if topic_memory:
        logger.info("已载入历史话题记忆: %s", topic_memory.get("topic_id"))

    research_focus = proposal.get("research_focus") or memory_ctx.get("research_focus") or []

    research_path = config.RESEARCH_DIR / f"{week_id}.json"
    if args.resume_research:
        if not research_path.exists():
            raise SystemExit(f"找不到可恢复的研究档案: {research_path}")
        saved = json.loads(research_path.read_text(encoding="utf-8"))
        dossier = saved["dossier"]
        evidence_store = EvidenceStore.from_lists(saved.get("evidence", []), saved.get("claims", []))
        manifest.stage("research_resumed", evidence_count=len(evidence_store.all()))
        logger.info("复用研究档案: %s（%d 条证据）", research_path, len(evidence_store.all()))
    else:
        logger.info("开始深度研究 agent ...")
    try:
        if not args.resume_research:
            dossier, evidence_store = run_research(
                topic_name=topic["name"],
                topic_description=topic.get("description", ""),
                candidate_works=proposal.get("candidate_works", []),
                topic_memory=topic_memory,
                research_focus=research_focus,
                proposed_thesis=proposal.get("proposed_thesis", ""),
            )

            logger.info("抽取机制卡（助写，规则优先）...")
            dossier = extract_and_attach_mechanism_cards(dossier, evidence_store)
    except Exception as exc:
        manifest.finish("failed", stage="research", error=str(exc))
        manifest.write(config.RUNS_DIR / f"{week_id}-publish.json")
        raise
    manifest.stage(
        "research_completed",
        evidence_count=len(evidence_store.all()),
        key_work_count=len(dossier.get("key_works", [])),
        high_card_count=len(dossier.get("high_mechanism_cards") or []),
    )

    research_out = {
        "dossier": dossier,
        "evidence": evidence_store.to_list(),
        "claims": evidence_store.claim_list(),
    }
    research_path.write_text(
        json.dumps(research_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "深度研究完成，共积累 %d 条证据，%d 个关键工作，publish_mode=%s，high=%d",
        len(evidence_store.all()),
        len(dossier.get("key_works", [])),
        dossier.get("publish_mode"),
        len(dossier.get("high_mechanism_cards") or []),
    )

    if dossier.get("publish_mode") == "insufficient":
        skip_payload = {
            "week_id": week_id,
            "topic_name": topic["name"],
            "reason": "可用机制卡不足，跳过发布以避免空心/成绩单式文章",
            "mechanism_cards": dossier.get("mechanism_cards") or [],
            "date": today_str(),
        }
        skip_path = config.ARTICLES_DATA_DIR / f"{week_id}.skipped.json"
        skip_path.write_text(json.dumps(skip_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.error(
            "publish_mode=insufficient：已写入 %s，不更新 docs/。请换选题或补充可抓全文的代表工作后重跑。",
            skip_path,
        )
        manifest.finish("skipped", reason="insufficient_mechanism_evidence")
        manifest.write(config.RUNS_DIR / f"{week_id}-publish.json")
        return

    logger.info("开始撰写短深文与事实核查 ...")
    try:
        draft_path = config.ARTICLES_DATA_DIR / f"{week_id}.draft.json"
        article = compose_article(
            topic["name"], dossier, evidence_store,
            draft_checkpoint=draft_path,
            resume_draft=args.resume_draft,
        )
    except ArticleQualityError as exc:
        skip_payload = {
            "week_id": week_id,
            "topic_name": topic["name"],
            "reason": exc.reason,
            "quality_report": exc.report,
            "claims": evidence_store.claim_list(),
            "date": today_str(),
        }
        skip_path = config.ARTICLES_DATA_DIR / f"{week_id}.skipped.json"
        skip_path.write_text(json.dumps(skip_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.error("文章未通过质量闸门，已写入 %s，不更新 docs/", skip_path)
        manifest.finish("skipped", reason=exc.reason, quality_report=exc.report)
        manifest.write(config.RUNS_DIR / f"{week_id}-publish.json")
        return
    except Exception as exc:
        manifest.finish("failed", stage="writing", error=str(exc))
        manifest.write(config.RUNS_DIR / f"{week_id}-publish.json")
        raise

    (config.ARTICLES_DATA_DIR / f"{week_id}.json").write_text(
        json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    build_article_page(article, week_id=week_id, week_label=week_label(week_id), publish_date=today_str())
    logger.info("已发布: docs/articles/%s/index.html（并同步为最新一期 docs/index.html）", week_id)

    memory_warning = ""
    try:
        logger.info("写回 Topic Memory ...")
        memory = distill_and_upsert(
            week_id=week_id,
            topic_name=topic["name"],
            article=article,
            dossier=dossier,
            existing=topic_memory,
        )
        logger.info("Topic Memory 已更新: %s", memory.get("topic_id"))
    except Exception as exc:
        # 页面已完成原子发布；Topic Memory 是后置增强，不应把成功发布误报为失败。
        memory_warning = str(exc)
        logger.exception("Topic Memory 更新失败，保留已发布页面并记录告警")

    manifest.finish(
        "completed",
        article_title=article.get("title", ""),
        quality_report=article.get("quality_report", {}),
        memory_warning=memory_warning,
    )
    manifest.write(config.RUNS_DIR / f"{week_id}-publish.json")


if __name__ == "__main__":
    main()
