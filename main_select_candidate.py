"""人工从当周 shortlist 选择候补题目，并安全重建正式发布提案。"""

from __future__ import annotations

import argparse
import json
import logging

import config
from processing.github_issue import create_proposal_issue
from processing.run_manifest import RunManifest
from processing.topic_selection_v2 import promote_shortlist_candidate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--week-id", required=True)
    parser.add_argument("--rank", type=int, required=True, help="shortlist 中的名次，从 1 开始")
    parser.add_argument("--reason", required=True, help="人工选择该候选的编辑理由")
    parser.add_argument("--no-issue", action="store_true")
    args = parser.parse_args()

    path = config.PROPOSALS_DIR / f"{args.week_id}.json"
    if not path.exists():
        raise SystemExit(f"找不到提案文件: {path}")
    original = json.loads(path.read_text(encoding="utf-8"))
    proposal = promote_shortlist_candidate(original, args.rank, args.reason)
    path.write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8")

    topic = proposal["selected_topic"]
    manifest = RunManifest(run_id=f"{args.week_id}-human-selection", workflow="human_candidate_selection")
    manifest.finish("completed", selected_question=topic["name"], candidate_rank=args.rank,
                    override_reason=args.reason)
    manifest.write(config.RUNS_DIR / f"{args.week_id}-human-selection.json")
    logger.info("已选择候补 #%d: %s", args.rank, topic["name"])

    if not args.no_issue:
        create_proposal_issue(args.week_id, proposal)


if __name__ == "__main__":
    main()
