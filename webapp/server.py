"""FastAPI web app: a human-in-the-loop front end for Sentinel-Mesh.

Flow:
  1. POST /api/analyze runs the pipeline in dry-run mode and returns every
     stage's output (facts, retrieved policy, decision, diff preview).
  2. If a patch is proposed, the decision is stored SERVER-SIDE under a
     one-shot pending id. The browser only ever holds that id.
  3. POST /api/apply {pending_id} applies exactly the stored decision --
     the client cannot edit field names or values on the way through, and
     the MCP server re-validates the allowlist before writing.

The UI is a single static page (webapp/static/index.html): paste a new
regulation, watch each stage run, review the diff, click approve.
"""
from __future__ import annotations

import difflib
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from sentinel_mesh import config
from sentinel_mesh.agent import (
    ComplianceDecision,
    PipelineResult,
    apply_decision,
    run_pipeline,
)
from sentinel_mesh.audit import hash_for_audit

BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "static" / "index.html"

app = FastAPI(title="Sentinel-Mesh Web", version="1.0.0")

# One-shot store of dry-run decisions awaiting human approval. Server-side
# on purpose: the browser can approve or discard, never edit, a proposal.
PENDING_TTL_SECONDS = 30 * 60


class PendingPatch(BaseModel):
    law_text: str
    decision: ComplianceDecision
    policy_hash: str
    created_at: float


_PENDING: dict[str, PendingPatch] = {}


def _prune_pending() -> None:
    cutoff = time.time() - PENDING_TTL_SECONDS
    for pid in [k for k, v in _PENDING.items() if v.created_at < cutoff]:
        del _PENDING[pid]


def _make_settings(dry_run: bool) -> config.AgentSettings:
    """Isolated so tests can point the app at a temp policy file."""
    return config.AgentSettings(dry_run=dry_run)


class AnalyzeRequest(BaseModel):
    law_text: str


class ApplyRequest(BaseModel):
    pending_id: str


def _build_diff(policy_text: str, old_text: str, new_text: str) -> Optional[dict]:
    """Unified-diff preview of what the patch would do, or None if the
    old_text is not present (the MCP server would reject it anyway)."""
    if old_text not in policy_text:
        return None
    updated = policy_text.replace(old_text, new_text)
    diff = list(
        difflib.unified_diff(
            policy_text.splitlines(),
            updated.splitlines(),
            fromfile="internal_policy.txt (current)",
            tofile="internal_policy.txt (proposed)",
            lineterm="",
        )
    )
    return {"diff": diff, "updated_policy": updated}


def _pipeline_payload(result: PipelineResult, settings: config.AgentSettings) -> dict:
    policy_path = Path(settings.policy_file)
    policy_text = policy_path.read_text(encoding="utf-8") if policy_path.exists() else ""

    payload = {
        "facts": asdict(result.extracted_facts),
        "facts_hint": result.extracted_facts.as_query_hint(),
        "retrieved_policy": result.retrieved_policy,
        "decision": result.decision.model_dump() if result.decision else None,
        "llm_raw_response": result.llm_raw_response,
        "action_result": result.action_result,
        "pending_id": None,
        "preview": None,
    }

    # Offer an approval button only when the dry run actually validated a
    # real, allowlisted patch (the MCP dry-run call already re-checked it).
    if (
        result.decision
        and result.decision.conflict
        and result.action_result.startswith("DRY RUN")
    ):
        preview = _build_diff(
            policy_text, result.decision.old_text or "", result.decision.new_text or ""
        )
        _prune_pending()
        pending_id = uuid.uuid4().hex[:12]
        _PENDING[pending_id] = PendingPatch(
            law_text=result.extracted_facts.raw_text,
            decision=result.decision,
            policy_hash=hash_for_audit(policy_text),
            created_at=time.time(),
        )
        payload["pending_id"] = pending_id
        payload["preview"] = preview

    return payload


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/api/policy")
def get_policy() -> dict:
    settings = _make_settings(dry_run=True)
    path = Path(settings.policy_file)
    return {
        "path": str(path),
        "content": path.read_text(encoding="utf-8") if path.exists() else "",
    }


@app.get("/api/audit")
def get_audit(limit: int = 50) -> dict:
    log_file = config.AUDIT_LOG_FILE
    if not log_file.exists():
        return {"entries": []}
    lines = log_file.read_text(encoding="utf-8").splitlines()
    return {"entries": lines[-limit:]}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    law_text = req.law_text.strip()
    if not law_text:
        raise HTTPException(status_code=400, detail="law_text must not be empty.")
    settings = _make_settings(dry_run=True)
    try:
        result = run_pipeline(law_text, settings)
    except RuntimeError as exc:
        # e.g. Ollama not running / model not pulled -- surface as a 503
        # with the actionable message agent.py already builds.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        # Model replied with unparseable JSON -- show the raw response.
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return _pipeline_payload(result, settings)


@app.post("/api/apply")
def apply(req: ApplyRequest) -> dict:
    _prune_pending()
    pending = _PENDING.pop(req.pending_id, None)
    if pending is None:
        raise HTTPException(
            status_code=404,
            detail="Unknown or expired pending id. Re-run analyze to get a fresh proposal.",
        )

    settings = _make_settings(dry_run=False)
    policy_path = Path(settings.policy_file)
    current_text = policy_path.read_text(encoding="utf-8") if policy_path.exists() else ""
    if hash_for_audit(current_text) != pending.policy_hash:
        raise HTTPException(
            status_code=409,
            detail="The policy file changed since this analysis. Re-run analyze before applying.",
        )

    action_result = apply_decision(pending.decision, settings)
    new_text = policy_path.read_text(encoding="utf-8")
    return {
        "action_result": action_result,
        "policy": new_text,
        "applied": action_result.startswith("APPLIED"),
    }
