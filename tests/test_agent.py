"""Tests for the agent orchestration logic: structured-output parsing and
the conflict-gated control flow.

These are the tests that directly prove the original bug is fixed: in
main_agent.py, patch_policy_file.invoke(...) ran unconditionally regardless
of what the model said. Here, run_pipeline must call the patch tool only
when decision.conflict is True, and must skip it when False.
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from sentinel_mesh.agent import ComplianceDecision, parse_decision, run_pipeline
from sentinel_mesh import config
from sentinel_mesh.nlp_extractor import ExtractedFacts


def test_parse_decision_extracts_json_even_with_surrounding_prose():
    raw = (
        "Sure, here is my analysis:\n"
        '{"conflict": true, "field_name": "transaction_log_retention", '
        '"old_text": "90 days", "new_text": "30 days", "new_numeric_value": 30, '
        '"reasoning": "The law caps retention at 30 days."}\n'
        "Let me know if you need anything else."
    )
    decision = parse_decision(raw)
    assert decision.conflict is True
    assert decision.field_name == "transaction_log_retention"
    assert decision.new_numeric_value == 30


def test_parse_decision_raises_on_malformed_json():
    with pytest.raises(ValueError):
        parse_decision("I think there might be a conflict but I'm not sure.")


def test_no_conflict_means_no_patch_attempt(monkeypatch, tmp_path):
    """The central regression test: when the model reports no conflict,
    the MCP patch tool must never be invoked."""
    facts = ExtractedFacts(numeric_terms=["30 days"], raw_text="some law")
    monkeypatch.setattr(
        "sentinel_mesh.agent.extract_compliance_facts", lambda text, model_name: facts
    )

    class FakeStore:
        def __init__(self, *a, **kw):
            pass

        def query(self, *a, **kw):
            return "Retention is already 30 days."

    monkeypatch.setattr("sentinel_mesh.agent.PolicyVectorStore", FakeStore)
    monkeypatch.setattr(
        "sentinel_mesh.agent.call_llm",
        lambda prompt, model: '{"conflict": false, "reasoning": "Already compliant."}',
    )

    patch_was_called = {"value": False}

    async def fake_apply_patch(*args, **kwargs):
        patch_was_called["value"] = True
        return "SHOULD NOT HAPPEN"

    monkeypatch.setattr("sentinel_mesh.agent._apply_patch_via_mcp", fake_apply_patch)

    settings = config.AgentSettings(audit_log_file=tmp_path / "audit.jsonl")
    result = run_pipeline("some law text", settings)

    assert result.decision.conflict is False
    assert patch_was_called["value"] is False
    assert "No action taken" in result.action_result


def test_conflict_triggers_patch_with_models_own_values(monkeypatch, tmp_path):
    """When the model DOES report a conflict, the patch tool must be called
    with the model's derived values -- not hardcoded ones."""
    facts = ExtractedFacts(numeric_terms=["30 days"], raw_text="some law")
    monkeypatch.setattr(
        "sentinel_mesh.agent.extract_compliance_facts", lambda text, model_name: facts
    )

    class FakeStore:
        def __init__(self, *a, **kw):
            pass

        def query(self, *a, **kw):
            return "Retention is currently 90 days."

    monkeypatch.setattr("sentinel_mesh.agent.PolicyVectorStore", FakeStore)
    monkeypatch.setattr(
        "sentinel_mesh.agent.call_llm",
        lambda prompt, model: (
            '{"conflict": true, "field_name": "transaction_log_retention", '
            '"old_text": "90 days", "new_text": "30 days", "new_numeric_value": 30, '
            '"reasoning": "Law requires 30 day cap."}'
        ),
    )

    captured = {}

    async def fake_apply_patch(decision, file_path, dry_run):
        captured["decision"] = decision
        captured["dry_run"] = dry_run
        return "DRY RUN (no write performed)"

    monkeypatch.setattr("sentinel_mesh.agent._apply_patch_via_mcp", fake_apply_patch)

    settings = config.AgentSettings(audit_log_file=tmp_path / "audit.jsonl", dry_run=True)
    result = run_pipeline("some law text", settings)

    assert captured["decision"].new_text == "30 days"
    assert captured["dry_run"] is True
    assert "DRY RUN" in result.action_result


def test_field_outside_allowlist_is_blocked_before_mcp_call(monkeypatch, tmp_path):
    facts = ExtractedFacts(raw_text="some law")
    monkeypatch.setattr(
        "sentinel_mesh.agent.extract_compliance_facts", lambda text, model_name: facts
    )

    class FakeStore:
        def __init__(self, *a, **kw):
            pass

        def query(self, *a, **kw):
            return "policy text"

    monkeypatch.setattr("sentinel_mesh.agent.PolicyVectorStore", FakeStore)
    monkeypatch.setattr(
        "sentinel_mesh.agent.call_llm",
        lambda prompt, model: (
            '{"conflict": true, "field_name": "admin_credentials", '
            '"old_text": "x", "new_text": "y", "new_numeric_value": 1, "reasoning": "n/a"}'
        ),
    )

    patch_was_called = {"value": False}

    async def fake_apply_patch(*a, **kw):
        patch_was_called["value"] = True
        return "SHOULD NOT HAPPEN"

    monkeypatch.setattr("sentinel_mesh.agent._apply_patch_via_mcp", fake_apply_patch)

    settings = config.AgentSettings(audit_log_file=tmp_path / "audit.jsonl")
    result = run_pipeline("some law text", settings)

    assert patch_was_called["value"] is False
    assert "BLOCKED" in result.action_result
