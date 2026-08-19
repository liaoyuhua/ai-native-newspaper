import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from openai import APIConnectionError

from processing.llm_client import run_tool_loop
from research.evidence import EvidenceStore
from writing.quality_gate import _audit_claims_by_section, _soften_inference_language
from writing.review_checkpoint import load_review_checkpoint, REVIEW_CHECKPOINT_VERSION


class ResumableReviewTest(unittest.TestCase):
    def test_old_review_policy_checkpoint_is_invalidated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.json"
            path.write_text(json.dumps({
                "draft_fingerprint": "fp", "reader_complete": True, "sections": [{"text": "old"}],
            }), encoding="utf-8")
            restored = load_review_checkpoint(path, "fp")
        self.assertEqual(restored["checkpoint_version"], REVIEW_CHECKPOINT_VERSION)
        self.assertNotIn("sections", restored)

    def test_inference_language_cleanup_is_deterministic_and_targeted(self):
        sections = [{"text": "SPP 证明预训练可以形成方向。另一个事实不变。"}]
        changed = _soften_inference_language(sections, [{
            "text": "SPP 证明预训练可以形成方向",
            "kind": "inference",
            "support": "overstated_inference",
        }])
        self.assertEqual(changed, 1)
        self.assertIn("一种可能的解释是，SPP 提示预训练可以形成方向", sections[0]["text"])
        self.assertIn("另一个事实不变", sections[0]["text"])

    def test_claim_audit_resumes_after_last_completed_section(self):
        sections = [
            {"heading": "a", "text": "first"},
            {"heading": "b", "text": "second"},
        ]
        checkpoint = {"draft_fingerprint": "fp"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.json"
            with patch("writing.quality_gate._audit_claims", side_effect=[
                {"claims": [{"text": "one"}]}, RuntimeError("temporary failure"),
            ]) as audit:
                with self.assertRaisesRegex(RuntimeError, "temporary"):
                    _audit_claims_by_section(sections, EvidenceStore(), 1, checkpoint, path)
                self.assertEqual(audit.call_count, 2)

            restored = json.loads(path.read_text(encoding="utf-8"))
            with patch("writing.quality_gate._audit_claims", return_value={
                "claims": [{"text": "two"}],
            }) as audit:
                result = _audit_claims_by_section(
                    sections, EvidenceStore(), 1, restored, path,
                )
                self.assertEqual(audit.call_count, 1)
                self.assertEqual([claim["text"] for claim in result["claims"]], ["one", "two"])

    def test_research_loop_switches_model_without_losing_context(self):
        request = MagicMock()
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content="done", tool_calls=None,
        ))])
        with patch("processing.llm_client._create_completion", side_effect=[
            APIConnectionError(request=request), response,
        ]) as create:
            output = run_tool_loop(
                "primary", "system", "user", [], {}, 0, fallback_model="fallback",
            )
        self.assertEqual(output, "done")
        self.assertEqual(create.call_args_list[0].kwargs["model"], "primary")
        self.assertEqual(create.call_args_list[1].kwargs["model"], "fallback")
        self.assertEqual(
            create.call_args_list[0].kwargs["messages"],
            create.call_args_list[1].kwargs["messages"],
        )


if __name__ == "__main__":
    unittest.main()
