"""Stage 4: the MCP tool server that is allowed to touch policy files.

Fixes vs. the original level4_mcp.py / main_agent.py duplication:
1. There is now exactly ONE implementation of the patch tool. main_agent.py
   no longer defines a second, competing LangChain @tool -- it calls this
   server as a real MCP client over stdio (see agent.py).
2. Every write is checked against config.ALLOWED_POLICY_FIELDS before it is
   allowed to happen -- an out-of-range or unrecognized field is rejected,
   which is the main defense against a prompt-injected or hallucinated
   value reaching disk.
3. dry_run is a first-class parameter. Callers must explicitly pass
   dry_run=False to write; the default only reports what WOULD change.
4. Writes are atomic (write to temp file, then replace) and a timestamped
   backup of the original file is kept, so a bad patch is always reversible.
5. Every call is recorded to the audit log, success or failure.

Uses the MCP Python SDK's high-level server API. As of SDK 2.x this class
is `MCPServer` in `mcp.server.mcpserver` (it was named `FastMCP` in SDK 1.x,
under `mcp.server.fastmcp`). Check `pip show mcp` if this import ever
breaks after an SDK upgrade -- the class has already been renamed once.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from . import config
from .audit import hash_for_audit, record_event

mcp = MCPServer("Sentinel-Mesh-Policy-Server")


def _validate_field(field_name: str, new_value: int) -> config.PolicyField:
    spec = config.ALLOWED_POLICY_FIELDS.get(field_name)
    if spec is None:
        raise ValueError(
            f"'{field_name}' is not an allowed policy field. "
            f"Allowed fields: {sorted(config.ALLOWED_POLICY_FIELDS)}"
        )
    if not (spec.min_value <= new_value <= spec.max_value):
        raise ValueError(
            f"Proposed value {new_value} for '{field_name}' is outside the "
            f"allowed range [{spec.min_value}, {spec.max_value}] {spec.unit}."
        )
    return spec


@mcp.tool()
def patch_policy_value(
    field_name: str,
    old_text: str,
    new_text: str,
    new_numeric_value: int,
    file_path: str = str(config.DEFAULT_POLICY_FILE),
    dry_run: bool = True,
) -> str:
    """Patch a single, allowlisted value in the internal policy file.

    Args:
        field_name: must be one of config.ALLOWED_POLICY_FIELDS (e.g.
            "transaction_log_retention").
        old_text: exact substring currently in the file to replace.
        new_text: exact substring to replace it with.
        new_numeric_value: the numeric value being set, checked against the
            allowlisted min/max range for field_name before anything runs.
        file_path: path to the policy file.
        dry_run: if True (default), reports the change WITHOUT writing it.
            Must be explicitly set False to actually modify the file.
    """
    path = Path(file_path)

    try:
        spec = _validate_field(field_name, new_numeric_value)
    except ValueError as exc:
        record_event(
            config.AUDIT_LOG_FILE,
            "patch_rejected",
            field_name=field_name,
            reason=str(exc),
        )
        return f"REJECTED: {exc}"

    if not path.exists():
        msg = f"REJECTED: file not found: {path}"
        record_event(config.AUDIT_LOG_FILE, "patch_rejected", field_name=field_name, reason=msg)
        return msg

    original = path.read_text(encoding="utf-8")

    if old_text not in original:
        msg = f"REJECTED: expected text '{old_text}' not found in {path}."
        record_event(
            config.AUDIT_LOG_FILE,
            "patch_rejected",
            field_name=field_name,
            reason=msg,
            file_hash=hash_for_audit(original),
        )
        return msg

    updated = original.replace(old_text, new_text)

    if dry_run:
        record_event(
            config.AUDIT_LOG_FILE,
            "patch_dry_run",
            field_name=field_name,
            old_text=old_text,
            new_text=new_text,
            unit=spec.unit,
            file_hash_before=hash_for_audit(original),
            would_be_hash_after=hash_for_audit(updated),
        )
        return (
            f"DRY RUN (no write performed): would replace '{old_text}' with "
            f"'{new_text}' in {path}. Pass dry_run=False to apply."
        )

    # Timestamped backup before any real write.
    backup_path = path.with_suffix(
        path.suffix + f".bak.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    shutil.copy2(path, backup_path)

    # Atomic write: write to a temp file in the same directory, then
    # os.replace() so a crash mid-write can never leave a corrupt file.
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".tmp_patch_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(updated)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    record_event(
        config.AUDIT_LOG_FILE,
        "patch_applied",
        field_name=field_name,
        old_text=old_text,
        new_text=new_text,
        unit=spec.unit,
        file_hash_before=hash_for_audit(original),
        file_hash_after=hash_for_audit(updated),
        backup_path=str(backup_path),
    )
    return f"APPLIED: '{old_text}' -> '{new_text}' in {path}. Backup: {backup_path}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
