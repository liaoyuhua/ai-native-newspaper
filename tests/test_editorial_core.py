from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config
from processing.editorial_models import (
    EditorialJudgment,
    FeasibilityProbe,
    aggregate_judgments,
    mechanism_is_publishable,
)
from processing.run_manifest import RunManifest
from research.evidence import EvidenceStore
from sources.registry import SourceRegistry
from writing.contracts import align_sections, validate_concept_dependencies
from writing.reader_review import ensure_reader_context, reader_review_passed
from writing.outline import _enforce_outline
from writing.quality_gate import _keyword_overlap, _register_and_find_blocking_claims
from processing.topic_selection_v2 import promote_shortlist_candidate
from sources.common import RawItem


class EditorialModelsTest(unittest.TestCase):
    def test_blank_ci_setting_uses_default(self):
        import os

        os.environ["TEST_BLANK_SETTING"] = ""
        self.assertEqual(config._env_int("TEST_BLANK_SETTING", 7), 7)

    def test_scores_are_bounded_and_aggregated_by_median(self):
        judgments = [
            EditorialJudgment.from_dict("a", {"scores": {"importance": 99}}),
            EditorialJudgment.from_dict("b", {"scores": {"importance": 4}}),
            EditorialJudgment.from_dict("c", {"scores": {"importance": 1}}),
        ]
        result = aggregate_judgments(judgments)
        self.assertEqual(result["scores"]["importance"], 4.0)
        self.assertEqual(result["scores"]["technical_delta"], 1.0)

    def test_feasibility_defaults_missing_dimensions_to_failure(self):
        probe = FeasibilityProbe.from_dict({"scores": {"primary_sources": 5}, "recommendation": "pursue"})
        self.assertLess(probe.average, 2.0)

    def test_human_selection_rebuilds_complete_proposal(self):
        item = RawItem(title="Paper", url="https://example.test/paper", source="arxiv", authority=1.0)
        row = {
            "candidate": {
                "id": "q1", "question": "候选问题？", "scope": "限定范围", "why_now": "本周有新工作",
                "technical_delta": "改变分析单元", "reader_takeaway": "理解新边界", "lens": "systems_infra",
                "item_ids": [item.item_id], "signal_score": 0.8,
            },
            "editorial": {"average": 3.4, "fatal_flaws": []},
            "feasibility": {
                "average": 3.6, "recommendation": "pursue", "proposed_thesis": "待验证判断",
                "mechanism_targets": ["机制 A"], "missing_evidence": [], "scores": {},
            },
            "final_score": 3.456,
            "shortlist_score": 3.4,
        }
        source = {"publish_recommendation": "skip", "reason": "quality_threshold_not_met",
                  "shortlist": [row], "all_items": [item.to_dict()]}
        with patch("processing.topic_selection_v2.match_topic_memory", return_value=None):
            promoted = promote_shortlist_candidate(source, 1, "编辑认为该问题符合本刊方向")
        self.assertEqual(promoted["publish_recommendation"], "pursue")
        self.assertEqual(promoted["selected_topic"]["name"], "候选问题？")
        self.assertEqual(promoted["human_override"]["candidate_rank"], 1)
        self.assertEqual(promoted["candidate_works"][0]["url"], item.url)

    def test_human_selection_cannot_bypass_fatal_flaw(self):
        item = RawItem(title="Paper", url="https://example.test/bad", source="blog")
        row = {
            "candidate": {"id": "q1", "question": "问题？", "item_ids": [item.item_id]},
            "editorial": {"average": 4.0, "fatal_flaws": ["single_marketing_source"]},
            "feasibility": {"average": 4.0, "recommendation": "pursue"},
            "final_score": 4.0,
        }
        with self.assertRaisesRegex(ValueError, "致命缺陷"):
            promote_shortlist_candidate(
                {"shortlist": [row], "all_items": [item.to_dict()]}, 1, "人工选择"
            )


class EvidenceStoreTest(unittest.TestCase):
    def test_abstract_and_fulltext_are_distinct_spans_of_one_document(self):
        store = EvidenceStore()
        abstract_id = store.add("Paper", "https://example.test/paper", "arxiv", "abstract")
        fulltext_id = store.add("Paper", "https://example.test/paper", "arxiv_fulltext", "full text")
        self.assertNotEqual(abstract_id, fulltext_id)
        self.assertEqual(store.get(abstract_id).document_id, store.get(fulltext_id).document_id)

    def test_longer_duplicate_span_replaces_shorter_one(self):
        store = EvidenceStore()
        eid = store.add("Paper", "https://example.test/paper", "webpage", "short")
        store.add("Paper", "https://example.test/paper", "webpage", "a substantially longer excerpt")
        self.assertEqual(store.get(eid).excerpt, "a substantially longer excerpt")

    def test_verifiable_claim_without_evidence_is_invalid(self):
        store = EvidenceStore()
        store.add_claim("The method is twice as fast.", "reported_result", [])
        self.assertEqual(len(store.validate_claim_links()), 1)

    def test_claim_audit_round_can_replace_stale_ledger(self):
        store = EvidenceStore()
        store.add_claim("unsupported old sentence", "fact", [])
        store.clear_claims()
        self.assertEqual(store.claim_list(), [])

    def test_low_confidence_mechanism_is_not_promoted_by_field_length(self):
        self.assertFalse(mechanism_is_publishable(
            model_confidence="low", steps_ok=True, state_ok=True, evidence_ok=True, thin_source=False
        ))


class FullDraftContractTest(unittest.TestCase):
    def test_missing_required_section_fails_closed(self):
        outline = [
            {"heading": "背景", "role": "context"},
            {"heading": "机制", "role": "mechanism", "card_index": 0},
        ]
        raw = [{"heading": "背景", "role": "context", "text": "正文"}]
        with self.assertRaises(ValueError):
            align_sections(raw, outline)

    def test_claim_repair_keyword_mapping_prefers_relevant_section(self):
        claim = "ElasticBack 使用条件触发器激活后门"
        relevant = "这一节解释 ElasticBack 的触发短语与规则如何共同生效。"
        unrelated = "这一节解释跨技能工件传递。"
        self.assertGreater(_keyword_overlap(claim, relevant), _keyword_overlap(claim, unrelated))

    def test_explicit_inference_does_not_require_direct_source_statement(self):
        store = EvidenceStore()
        audit = {"claims": [{
            "text": "可以推测，攻击语义被分散存储。",
            "kind": "inference",
            "evidence_ids": [],
            "support": "unsupported",
            "reason": "作者解释性推断",
        }]}
        self.assertEqual(_register_and_find_blocking_claims(audit, store), [])

    def test_concept_must_be_introduced_before_it_is_assumed(self):
        reader_context = {"prerequisites": [{
            "concept": "技能扫描器", "reader_likely_knows": False,
        }]}
        outline = [
            {"heading": "背景", "introduces": [], "assumes": ["技能扫描器"]},
            {"heading": "机制", "introduces": ["技能扫描器"], "assumes": []},
        ]
        issues = validate_concept_dependencies(outline, reader_context)
        self.assertTrue(any("解释前" in issue for issue in issues))

    def test_concept_dependency_passes_when_context_introduces_it(self):
        reader_context = {"prerequisites": [{
            "concept": "技能扫描器", "reader_likely_knows": False,
        }]}
        outline = [
            {"heading": "背景", "introduces": ["技能扫描器"], "assumes": []},
            {"heading": "机制", "introduces": [], "assumes": ["技能扫描器"]},
        ]
        self.assertEqual(validate_concept_dependencies(outline, reader_context), [])

    def test_reader_review_fails_if_any_core_object_is_undefined(self):
        review = {
            "background_pass": True,
            "checks": {
                "can_identify_system": True,
                "can_define_core_objects": True,
                "can_explain_normal_workflow": True,
                "can_explain_old_approach": True,
                "can_explain_new_failure": True,
            },
            "undefined_terms": ["Agent Skill"],
        }
        self.assertFalse(reader_review_passed(review))

    def test_existing_reader_context_does_not_call_model(self):
        context = {"prerequisites": [{"concept": "技能"}], "causal_bridge": ["A", "B"]}
        dossier = {"reader_context": context}
        self.assertIs(ensure_reader_context("topic", dossier), context)

    def test_outline_normalizer_assigns_missing_concepts_to_context(self):
        outline = {"sections": [{"heading": "背景", "role": "context", "goal": "解释问题"}]}
        reader_context = {"prerequisites": [{
            "concept": "技能扫描器", "reader_likely_knows": False,
        }]}
        fixed = _enforce_outline(outline, [], reader_context)
        self.assertIn("技能扫描器", fixed["sections"][0]["introduces"])
        self.assertIn("技能扫描器", fixed["sections"][0]["goal"])


class RunManifestTest(unittest.TestCase):
    def test_manifest_is_persisted(self):
        manifest = RunManifest("run-1", "scan")
        manifest.stage("collect", item_count=10)
        manifest.finish("completed", selected="q1")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            manifest.write(path)
            text = path.read_text(encoding="utf-8")
        self.assertIn('"status": "completed"', text)
        self.assertIn('"item_count": 10', text)


class SourceRegistryTest(unittest.TestCase):
    def test_validated_override_changes_runtime_config_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = SourceRegistry(Path(tmp) / "sources.json")
            configured = {"name": "Lab", "url": "https://lab.test/old.xml", "fetcher": "rss"}
            registry.set_validated_override("Lab", url="https://lab.test/feed.xml", fetcher="rss")
            effective = registry.effective_config(configured)
        self.assertEqual(effective["url"], "https://lab.test/feed.xml")
        self.assertEqual(configured["url"], "https://lab.test/old.xml")

    def test_repeated_failures_quarantine_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = SourceRegistry(Path(tmp) / "sources.json")
            for _ in range(config.SOURCE_QUARANTINE_AFTER_FAILURES):
                registry.record_failure("Lab", url="https://lab.test/feed", error="404")
            state = registry.summary()["sources"]["Lab"]
        self.assertEqual(state["status"], "quarantined")
        self.assertFalse(registry.should_attempt("Lab"))


if __name__ == "__main__":
    unittest.main()
