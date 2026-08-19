import tempfile
import unittest
from pathlib import Path

from writing.depth_revision import ArticleDepthValidator
from writing.revision_controller import RevisionController
from writing.revision_state import ArticleState, QualityIssue, ValidationResult


class LengthValidator:
    def __init__(self, target: int = 10):
        self.target = target

    def validate(self, state: ArticleState) -> ValidationResult:
        length = len(state.sections[0]["text"])
        issues = [] if length >= self.target else [QualityIssue(
            "short", "depth", "too short", "grow", deficit=(self.target - length) / self.target,
        )]
        return ValidationResult("length", issues, {"length": length})


class GrowRepairer:
    def __init__(self, amount: int):
        self.amount = amount

    def repair(self, state: ArticleState, issues: list[QualityIssue]) -> ArticleState:
        candidate = state.clone()
        candidate.sections[0]["text"] += "x" * self.amount
        return candidate


def make_state(text: str = "") -> ArticleState:
    return ArticleState("t", "", "thesis", [{"heading": "h", "text": text}])


class RevisionControllerTest(unittest.TestCase):
    def test_depth_validator_matches_final_gate_character_count(self):
        article = ArticleState(
            "t", "", "thesis",
            [{"heading": "h", "role": "mechanism", "card_index": 0,
              "text": "正文【EV:ev_very_long_identifier】"}],
            mechanism_cards=[{"method_role": "primary_subject"}],
        )
        result = ArticleDepthValidator().validate(article)
        self.assertEqual(result.metrics["body_chars"], 2)
        self.assertEqual(result.metrics["primary_mechanism_chars"], 2)

    def test_revises_until_validator_passes_and_saves_rounds(self):
        with tempfile.TemporaryDirectory() as directory:
            result = RevisionController(
                [LengthValidator()], {"grow": GrowRepairer(5)},
                max_rounds=3, snapshot_dir=Path(directory),
            ).run(make_state())
            self.assertEqual(result.status, "passed")
            self.assertEqual(result.state.sections[0]["text"], "x" * 10)
            self.assertEqual(len(result.rounds), 2)
            self.assertTrue((Path(directory) / "round-02.json").exists())

    def test_rolls_back_candidate_without_improvement(self):
        result = RevisionController(
            [LengthValidator()], {"grow": GrowRepairer(0)}, max_rounds=3,
        ).run(make_state("abc"))
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.stop_reason, "no_measurable_improvement")
        self.assertEqual(result.state.sections[0]["text"], "abc")
        self.assertEqual(len(result.rounds), 1)

    def test_returns_best_state_when_budget_exhausted(self):
        result = RevisionController(
            [LengthValidator(20)], {"grow": GrowRepairer(3)}, max_rounds=2,
        ).run(make_state())
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.stop_reason, "round_budget_exhausted")
        self.assertEqual(result.state.sections[0]["text"], "x" * 6)

    def test_does_not_repair_initially_valid_state(self):
        result = RevisionController(
            [LengthValidator(3)], {"grow": GrowRepairer(100)}, max_rounds=3,
        ).run(make_state("done"))
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.stop_reason, "already_passed")
        self.assertEqual(result.rounds, [])


if __name__ == "__main__":
    unittest.main()
