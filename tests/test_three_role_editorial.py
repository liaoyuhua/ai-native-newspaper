import unittest
from unittest.mock import patch

from research.evidence import EvidenceStore
from writing.quality_gate import ArticleQualityError
from writing.three_role_editorial import (
    deterministic_publish_checks,
    run_fact_editor,
    run_three_role_editorial,
)


def valid_article():
    sections = [
        {"heading": "背景", "role": "context", "card_index": None, "text": "背景解释" * 180},
        {"heading": "机制", "role": "mechanism", "card_index": 0, "text": "主角机制步骤" * 180},
        {"heading": "结论", "role": "close", "card_index": None, "text": "作者判断" * 100},
    ]
    outline = [
        {"heading": "背景", "role": "context", "card_index": None},
        {"heading": "机制", "role": "mechanism", "card_index": 0},
        {"heading": "结论", "role": "close", "card_index": None},
    ]
    cards = [{"method_role": "primary_subject"}]
    return sections, outline, cards


class ThreeRoleEditorialTest(unittest.TestCase):
    def test_fact_editor_ignores_non_exact_quote_and_unknown_category(self):
        sections, _, _ = valid_article()
        raw = {"issues": [
            {"severity": "blocking", "category": "factual_mismatch", "section_index": 0,
             "quote": "正文里不存在的句子", "problem": "x"},
            {"severity": "blocking", "category": "style_problem", "section_index": 0,
             "quote": "背景解释", "problem": "x"},
        ], "summary": "ok"}
        with patch("writing.three_role_editorial.chat_json_with_fallback", return_value=(raw, "m")):
            report = run_fact_editor("标题", "判断", sections, EvidenceStore())
        self.assertEqual(report["issues"], [])

    def test_fact_editor_ignores_self_dismissing_issue(self):
        sections, _, _ = valid_article()
        quote = "背景解释" * 4
        raw = {"issues": [{
            "severity": "blocking", "category": "factual_mismatch", "section_index": 0,
            "quote": quote, "problem": "证据支持该表述，不构成事实错误",
            "suggested_revision": "无需修改",
        }], "summary": "ok"}
        with patch("writing.three_role_editorial.chat_json_with_fallback", return_value=(raw, "m")):
            report = run_fact_editor("标题", "判断", sections, EvidenceStore())
        self.assertEqual(report["issues"], [])

    def test_no_blocking_fact_issue_skips_second_rewrite(self):
        sections, outline, cards = valid_article()
        with patch("writing.three_role_editorial.run_fact_editor", return_value={
            "issues": [], "summary": "pass", "model": "m", "blocking_count": 0,
        }), patch("writing.three_role_editorial.revise_by_chief_editor") as revise:
            result = run_three_role_editorial(
                "技术标题", "有限判断", sections, outline, cards, EvidenceStore(),
            )
        revise.assert_not_called()
        self.assertFalse(result.chief_decision["revision_applied"])

    def test_chief_revision_cannot_destroy_article_depth(self):
        sections, outline, cards = valid_article()
        issue = {"severity": "blocking", "category": "factual_mismatch", "section_index": 1,
                 "quote": "主角机制步骤主角机制步骤", "problem": "过强", "evidence_ids": []}
        collapsed = [
            {"heading": "背景", "role": "context", "card_index": None, "text": "短"},
            {"heading": "机制", "role": "mechanism", "card_index": 0, "text": "已改"},
            {"heading": "结论", "role": "close", "card_index": None, "text": "短"},
        ]
        with patch("writing.three_role_editorial.run_fact_editor", return_value={
            "issues": [issue], "summary": "", "model": "m", "blocking_count": 1,
        }), patch("writing.three_role_editorial.revise_by_chief_editor", return_value=(
            collapsed, {
                "decision": "publish", "resolved_issue_indexes": [0],
                "unresolved_issue_indexes": [],
            },
        )):
            with self.assertRaises(ArticleQualityError) as raised:
                run_three_role_editorial(
                    "技术标题", "有限判断", sections, outline, cards, EvidenceStore(),
                )
        self.assertEqual(raised.exception.reason, "chief_editor_regression")

    def test_chief_must_explicitly_resolve_every_blocking_issue(self):
        sections, outline, cards = valid_article()
        issue = {"severity": "blocking", "category": "factual_mismatch", "section_index": 1,
                 "quote": "主角机制步骤主角机制步骤", "problem": "过强", "evidence_ids": []}
        revised = [dict(section) for section in sections]
        revised[1]["text"] = revised[1]["text"].replace(issue["quote"], "收窄后的机制描述", 1)
        with patch("writing.three_role_editorial.run_fact_editor", return_value={
            "issues": [issue], "summary": "", "model": "m", "blocking_count": 1,
        }), patch("writing.three_role_editorial.revise_by_chief_editor", return_value=(
            revised, {
                "decision": "publish", "resolved_issue_indexes": [],
                "unresolved_issue_indexes": [],
            },
        )):
            with self.assertRaises(ArticleQualityError) as raised:
                run_three_role_editorial(
                    "技术标题", "有限判断", sections, outline, cards, EvidenceStore(),
                )
        self.assertEqual(raised.exception.reason, "chief_editor_rejected")

    def test_unknown_evidence_marker_is_deterministic_blocker(self):
        sections, _, cards = valid_article()
        sections[0]["text"] += "【EV:missing】"
        result = deterministic_publish_checks(
            "技术标题", "有限判断", sections, cards, EvidenceStore(),
        )
        self.assertFalse(result["pass"])
        self.assertEqual(result["invalid_evidence_markers"][0]["evidence_id"], "missing")

    def test_section_level_h1_is_deterministic_blocker(self):
        sections, _, cards = valid_article()
        sections[0]["text"] = "# 重复的文章标题\n\n" + sections[0]["text"]
        result = deterministic_publish_checks(
            "技术标题", "有限判断", sections, cards, EvidenceStore(),
        )
        self.assertFalse(result["pass"])
        self.assertEqual(result["format_violations"][0]["reason"], "section_contains_h1")


if __name__ == "__main__":
    unittest.main()
