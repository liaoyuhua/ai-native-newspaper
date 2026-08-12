"""
工作流 A 的入口：每周定时运行。
广度扫描 -> 聚类打分 -> 选出本周话题 -> 生成提案 -> 存档 -> 开 Issue 供人工审核。
"""

from __future__ import annotations

import argparse
import json
import logging

import config
from processing.github_issue import create_proposal_issue
from processing.run_manifest import RunManifest
from processing.topic_selection_v2 import generate_weekly_proposal_v2
from weekutil import current_week_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-days", type=int, default=14, help="扫描窗口，默认 14 天")
    parser.add_argument("--no-issue", action="store_true", help="只生成本地提案，不创建 GitHub Issue")
    args = parser.parse_args()

    week_id = current_week_id()
    manifest = RunManifest(run_id=f"{week_id}-scan", workflow="weekly_scan_v2")
    manifest.config = {"lookback_days": args.lookback_days, "schema_version": "2.0", "no_issue": args.no_issue}
    logger.info("开始本周(%s)广度扫描与选题", week_id)

    try:
        proposal = generate_weekly_proposal_v2(lookback_days=args.lookback_days)
    except Exception as exc:
        manifest.finish("failed", error=str(exc))
        manifest.write(config.RUNS_DIR / f"{week_id}-scan.json")
        raise

    out_path = config.PROPOSALS_DIR / f"{week_id}.json"
    out_path.write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("提案已写入 %s", out_path)
    manifest.stage(
        "selection_completed",
        item_count=len(proposal.get("all_items", [])),
        shortlist_count=len(proposal.get("shortlist", [])),
        recommendation=proposal.get("publish_recommendation"),
        source_health_counts=(proposal.get("source_health") or {}).get("counts", {}),
    )
    if proposal.get("publish_recommendation") != "pursue":
        manifest.finish("skipped", reason=proposal.get("reason", "quality_threshold_not_met"))
        manifest.write(config.RUNS_DIR / f"{week_id}-scan.json")
        logger.warning("本周没有候选越过质量门槛，不创建发布提案")
        return

    topic = proposal["selected_topic"]
    logger.info("本周选定问题: %s (score=%.3f)", topic["name"], topic["score"])
    manifest.finish("completed", selected_question=topic["name"], score=topic["score"])
    manifest.write(config.RUNS_DIR / f"{week_id}-scan.json")

    issue_url = None if args.no_issue else create_proposal_issue(week_id, proposal)
    if issue_url:
        logger.info("请前往 Issue 审核: %s", issue_url)


if __name__ == "__main__":
    main()
