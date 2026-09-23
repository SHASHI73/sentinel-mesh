"""Tests for the patch tool's safety gates: allowlist, range checks,
dry-run default, and atomic/backup writes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from sentinel_mesh import config
from sentinel_mesh.mcp_server import patch_policy_value


@pytest.fixture
def policy_file(tmp_path, monkeypatch):
    p = tmp_path / "internal_policy.txt"
    p.write_text(
        "Transaction logging history is preserved natively for a standard duration of 90 days.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "AUDIT_LOG_FILE", tmp_path / "audit_log.jsonl")
    return p


def test_dry_run_is_the_default_and_does_not_write(policy_file):
    result = patch_policy_value(
        field_name="transaction_log_retention",
        old_text="90 days",
        new_text="30 days",
        new_numeric_value=30,
        file_path=str(policy_file),
    )
    assert "DRY RUN" in result
    assert "90 days" in policy_file.read_text(encoding="utf-8")  # unchanged


def test_apply_with_dry_run_false_actually_writes(policy_file):
    result = patch_policy_value(
        field_name="transaction_log_retention",
        old_text="90 days",
        new_text="30 days",
        new_numeric_value=30,
        file_path=str(policy_file),
        dry_run=False,
    )
    assert "APPLIED" in result
    assert "30 days" in policy_file.read_text(encoding="utf-8")
    # a backup file should exist
    backups = list(policy_file.parent.glob("*.bak.*"))
    assert len(backups) == 1


def test_unknown_field_is_rejected_before_any_file_access(policy_file):
    result = patch_policy_value(
        field_name="admin_password",  # not in the allowlist
        old_text="90 days",
        new_text="anything",
        new_numeric_value=1,
        file_path=str(policy_file),
        dry_run=False,
    )
    assert "REJECTED" in result
    assert "not an allowed policy field" in result
    assert "90 days" in policy_file.read_text(encoding="utf-8")  # unchanged


def test_out_of_range_value_is_rejected(policy_file):
    result = patch_policy_value(
        field_name="transaction_log_retention",
        old_text="90 days",
        new_text="99999 days",
        new_numeric_value=99999,  # outside the allowed [1, 365] range
        file_path=str(policy_file),
        dry_run=False,
    )
    assert "REJECTED" in result
    assert "outside the allowed range" in result


def test_missing_old_text_is_rejected(policy_file):
    result = patch_policy_value(
        field_name="transaction_log_retention",
        old_text="120 days",  # not actually in the file
        new_text="30 days",
        new_numeric_value=30,
        file_path=str(policy_file),
        dry_run=False,
    )
    assert "REJECTED" in result
    assert "not found" in result
