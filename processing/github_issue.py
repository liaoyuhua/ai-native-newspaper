"""在 GitHub 上创建"本周提案"Issue，供人工审核确认后再手动触发 publish 工作流。"""

from __future__ import annotations

import logging

import requests

import config

logger = logging.getLogger(__name__)


def create_proposal_issue(week_id: str, proposal: dict) -> str | None:
    if not config.GITHUB_TOKEN or not config.GITHUB_REPOSITORY:
        logger.warning("未配置 GITHUB_TOKEN/GITHUB_REPOSITORY，跳过创建 Issue（本地调试模式）")
        return None

    topic = proposal["selected_topic"]
    works_lines = "\n".join(
        f"- [{w['title']}]({w['url']}) —— {w.get('why', '')}" for w in proposal.get("candidate_works", [])
    )
    runner_ups = "\n".join(
        f"- {t['name']} (score={t['score']}, lens={t.get('lens', '?')})"
        for t in proposal.get("runner_up_topics", [])
    )
    focus = proposal.get("research_focus") or []
    focus_lines = "\n".join(f"- {q}" for q in focus) if focus else "(无特别指定，按常规深挖)"
    memory_ctx = proposal.get("topic_memory_context") or {}
    memory_block = ""
    if memory_ctx:
        memory_block = f"""
**历史话题记忆**（{memory_ctx.get('relation', 'related')} → `{memory_ctx.get('topic_id', '')}`）：
```
{memory_ctx.get('prompt_brief', '')}
```
"""
    bd = topic.get("score_breakdown") or {}
    is_v2 = proposal.get("schema_version") == "2.0"
    if is_v2:
        score_line = (
            f"**综合分**: {topic['score']}（编辑评审={bd.get('editorial', {}).get('average')}, "
            f"可成文性={bd.get('feasibility', {}).get('average')}, 信号分={bd.get('signal_score')}）"
        )
    else:
        objective = bd.get("objective_score", topic["score"])
        lens_w = bd.get("lens_weight", 1.0)
        score_line = (
            f"**镜头**: {topic.get('lens', 'other')}（编辑权重 {lens_w}）\n"
            f"**打分**: {topic['score']} = 客观分 {objective} × 镜头权重"
            f"（多源覆盖={bd.get('source_diversity')}, 平均权威度={bd.get('avg_authority')}, 热度={bd.get('total_buzz')}）"
        )

    body = f"""\
## 本周提案：{topic['name']}

{score_line}
**与历史关系**: {topic.get('memory_relation', 'new')}

**总编评语**：
{proposal.get('rationale', '')}

**本期研究优先问题**：
{focus_lines}
{memory_block}
**候选精选工作**：
{works_lines or '(无)'}

**本周落选的其它候选话题**（仅供参考，供你决定是否想换一个方向）：
{runner_ups or '(无)'}

---

如果同意这个方向，去 Actions -> `publish` 工作流手动触发（week_id 填 `{week_id}`），系统会开始深度研究并生成文章。
如果想换一个话题/调整候选工作，直接编辑 `data/proposals/{week_id}.json` 后提交，再触发 publish。
"""

    resp = requests.post(
        f"https://api.github.com/repos/{config.GITHUB_REPOSITORY}/issues",
        headers={
            "Authorization": f"token {config.GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
        },
        json={"title": f"[提案] {week_id}: {topic['name']}", "body": body, "labels": ["proposal"]},
        timeout=15,
    )
    if resp.status_code >= 300:
        logger.error("创建 Issue 失败: %s %s", resp.status_code, resp.text[:500])
        return None

    issue_url = resp.json().get("html_url")
    logger.info("已创建提案 Issue: %s", issue_url)
    return issue_url
