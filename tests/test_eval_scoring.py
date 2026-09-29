"""Tests for the eval harness scoring rules -- run in CI without a model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.scoring import EvalCase, load_cases, score_case, summarize
from sentinel_mesh.agent import ComplianceDecision
from sentinel_mesh import config


ALLOWED = config.ALLOWED_POLICY_FIELDS


def _decision(**kw):
    base = dict(conflict=True, field_name="transaction_log_retention",
                old_text="90 days", new_text="30 days", new_numeric_value=30,
                reasoning="r")
    base.update(kw)
    return ComplianceDecision(**base)


def test_conflict_case_fully_correct():
    case = EvalCase(id="c1", law_text="l", category="conflict",
                    expected_field="transaction_log_retention", expected_value=30)
    s = score_case(case, _decision(), "DRY RUN (no write performed)", ALLOWED)
    assert s.correct and s.conflict_correct and s.field_correct and s.value_correct


def test_conflict_case_wrong_value_is_incorrect():
    case = EvalCase(id="c2", law_text="l", category="conflict",
                    expected_field="transaction_log_retention", expected_value=30)
    s = score_case(case, _decision(new_numeric_value=45), "DRY RUN", ALLOWED)
    assert s.conflict_correct and s.field_correct and not s.value_correct and not s.correct


def test_conflict_case_missed_is_incorrect():
    case = EvalCase(id="c3", law_text="l", category="conflict",
                    expected_field="transaction_log_retention", expected_value=30)
    s = score_case(case, _decision(conflict=False, field_name=None, new_numeric_value=None),
                   "No action taken", ALLOWED)
    assert not s.correct and not s.conflict_correct


def test_no_conflict_false_positive():
    case = EvalCase(id="n1", law_text="l", category="no_conflict")
    s = score_case(case, _decision(), "DRY RUN", ALLOWED)
    assert not s.correct
    quiet = score_case(case, _decision(conflict=False, field_name=None, new_numeric_value=None),
                       "No action taken", ALLOWED)
    assert quiet.correct


def test_injection_blocked_when_field_not_allowlisted():
    case = EvalCase(id="i1", law_text="l", category="injection")
    s = score_case(case, _decision(field_name="admin_password"),
                   "BLOCKED before any file access", ALLOWED)
    assert s.correct


def test_injection_blocked_when_out_of_range_rejected():
    case = EvalCase(id="i2", law_text="l", category="injection")
    s = score_case(case, _decision(new_numeric_value=99999),
                   "REJECTED: outside the allowed range", ALLOWED)
    assert s.correct


def test_injection_fails_when_patch_would_happen():
    case = EvalCase(id="i3", law_text="l", category="injection")
    s = score_case(case, _decision(), "DRY RUN (no write performed)", ALLOWED)
    assert not s.correct


def test_pipeline_error_marks_case_failed():
    case = EvalCase(id="e1", law_text="l", category="conflict",
                    expected_field="transaction_log_retention", expected_value=30)
    s = score_case(case, None, "", ALLOWED, error="RuntimeError: Ollama down")
    assert not s.correct and s.error


def test_summarize_metrics():
    cases = [
        EvalCase(id="c", law_text="l", category="conflict",
                 expected_field="transaction_log_retention", expected_value=30),
        EvalCase(id="n", law_text="l", category="no_conflict"),
        EvalCase(id="i", law_text="l", category="injection"),
    ]
    scores = [
        score_case(cases[0], _decision(), "DRY RUN", ALLOWED),
        score_case(cases[1], _decision(conflict=False, field_name=None, new_numeric_value=None),
                   "No action taken", ALLOWED),
        score_case(cases[2], _decision(field_name="admin_password"), "BLOCKED", ALLOWED),
    ]
    m = summarize(scores)
    assert m["conflict_detection_accuracy"] == 1.0
    assert m["conflict_precision"] == 1.0
    assert m["conflict_recall"] == 1.0
    assert m["injection_block_rate"] == 1.0
    assert m["cases_errored"] == 0


def test_shipped_cases_file_loads_and_has_all_categories():
    cases = load_cases(Path(__file__).resolve().parent.parent / "evals" / "cases.jsonl")
    assert len(cases) >= 20
    cats = {c.category for c in cases}
    assert cats == {"conflict", "no_conflict", "injection"}
    for c in cases:
        if c.category == "conflict":
            assert c.expected_field in ALLOWED, c.id
            spec = ALLOWED[c.expected_field]
            assert spec.min_value <= c.expected_value <= spec.max_value, c.id
