"""Stage 3 + orchestration: LLM reasoning that actually gates the action.

This is the central fix versus the original main_agent.py. Previously the
Ollama call happened, its output was printed, and then the file patch fired
unconditionally with hardcoded values -- the model's reasoning had no
effect on the outcome. Here, the model is prompted for strict structured
JSON (a ComplianceDecision), that JSON is parsed and validated, and the MCP
patch tool is only ever invoked when the model reports a genuine conflict --
using the values it derived, not hardcoded ones.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, ValidationError

from . import config
from .audit import record_event
from .nlp_extractor import ExtractedFacts, extract_compliance_facts
from .rag_store import PolicyVectorStore

try:
    import ollama
except ImportError:  # pragma: no cover
    ollama = None


class ComplianceDecision(BaseModel):
    """Structured output the LLM must return. Nothing downstream trusts
    free-text -- only this parsed, typed object."""

    conflict: bool
    field_name: Optional[str] = None
    old_text: Optional[str] = None
    new_text: Optional[str] = None
    new_numeric_value: Optional[int] = None
    reasoning: str = ""


@dataclass
class PipelineResult:
    extracted_facts: ExtractedFacts
    retrieved_policy: Optional[str]
    decision: Optional[ComplianceDecision]
    action_result: str
    llm_raw_response: str = ""


PROMPT_TEMPLATE = """You are a compliance-checking assistant. Compare the NEW LAW against \
the CURRENT INTERNAL POLICY and decide whether the policy violates the law.

NEW LAW:
{law_text}

RELEVANT FACTS EXTRACTED FROM THE LAW:
{facts_hint}

CURRENT INTERNAL POLICY (retrieved from the policy database):
{current_policy}

Respond with ONLY a single JSON object, no prose, no markdown fences, matching \
exactly this shape:
{{
  "conflict": true or false,
  "field_name": one of {allowed_fields} or null if no conflict,
  "old_text": the exact substring in the current policy that must change, or null,
  "new_text": the exact replacement substring, or null,
  "new_numeric_value": the new number being set, or null,
  "reasoning": a one or two sentence explanation
}}

If there is no conflict, set conflict to false and leave the other fields null.
"""


def build_prompt(law_text: str, facts: ExtractedFacts, current_policy: str) -> str:
    return PROMPT_TEMPLATE.format(
        law_text=law_text,
        facts_hint=facts.as_query_hint(),
        current_policy=current_policy,
        allowed_fields=sorted(config.ALLOWED_POLICY_FIELDS),
    )


def call_llm(prompt: str, model: str = config.OLLAMA_MODEL) -> str:
    if ollama is None:
        raise RuntimeError("The `ollama` package is not installed. Run `pip install ollama`.")
    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    except Exception as exc:  # noqa: BLE001 - surfacing any Ollama/connection failure clearly
        raise RuntimeError(
            f"Could not reach Ollama or model '{model}' is not pulled. "
            f"Run `ollama pull {model}` and ensure the Ollama service is running. "
            f"Original error: {exc}"
        ) from exc
    return response["message"]["content"]


def parse_decision(raw_response: str) -> ComplianceDecision:
    """Extract the JSON object from the model's reply. Small local models
    sometimes wrap JSON in prose or code fences despite instructions, so we
    pull out the first {...} block before parsing."""
    match = re.search(r"\{.*\}", raw_response, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response:\n{raw_response}")
    try:
        payload = json.loads(match.group(0))
        return ComplianceDecision(**payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"LLM response did not match the expected schema: {exc}\nRaw: {raw_response}") from exc


async def _apply_patch_via_mcp(decision: ComplianceDecision, file_path: str, dry_run: bool) -> str:
    """Call the MCP policy server as a real MCP client over stdio.

    This replaces the original design where main_agent.py contained its own
    duplicate copy of the patch logic. There is now exactly one
    implementation (mcp_server.py), and it is reached only through the MCP
    protocol, the same way any other MCP client would call it.
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command="python",
        args=["-m", "sentinel_mesh.mcp_server"],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "patch_policy_value",
                arguments={
                    "field_name": decision.field_name,
                    "old_text": decision.old_text,
                    "new_text": decision.new_text,
                    "new_numeric_value": decision.new_numeric_value,
                    "file_path": file_path,
                    "dry_run": dry_run,
                },
            )
            return "\n".join(block.text for block in result.content if hasattr(block, "text"))


def run_pipeline(law_text: str, settings: config.AgentSettings = config.AgentSettings()) -> PipelineResult:
    """Run the full extract -> retrieve -> reason -> (maybe) act pipeline."""

    # Stage 1: NLP extraction -- now actually consumed in Stage 2's query.
    facts = extract_compliance_facts(law_text, model_name=settings.spacy_model)
    record_event(settings.audit_log_file, "nlp_extracted", facts=facts.as_query_hint())

    # Stage 2: RAG retrieval, informed by the extracted facts.
    store = PolicyVectorStore(
        persist_dir=settings.chroma_dir,
        embedding_model=settings.embedding_model,
    )
    query = f"policy retention specification {facts.as_query_hint()}"
    current_policy = store.query(query) or "No matching internal policy found."
    record_event(settings.audit_log_file, "rag_retrieved", query=query, found=bool(current_policy))

    # Stage 3: LLM reasoning -> structured decision.
    prompt = build_prompt(law_text, facts, current_policy)
    raw_response = call_llm(prompt, model=settings.ollama_model)
    decision = parse_decision(raw_response)
    record_event(
        settings.audit_log_file,
        "llm_decision",
        conflict=decision.conflict,
        field_name=decision.field_name,
        reasoning=decision.reasoning,
    )

    # Stage 4: act ONLY if the model reported a real, allowlisted conflict.
    if not decision.conflict:
        action_result = "No action taken: model reported no compliance conflict."
        return PipelineResult(facts, current_policy, decision, action_result, raw_response)

    if decision.field_name not in settings.allowed_fields:
        action_result = (
            f"BLOCKED before any file access: field '{decision.field_name}' is not in "
            f"the allowlist {sorted(settings.allowed_fields)}."
        )
        record_event(settings.audit_log_file, "patch_blocked_by_allowlist", field_name=decision.field_name)
        return PipelineResult(facts, current_policy, decision, action_result, raw_response)

    action_result = asyncio.run(
        _apply_patch_via_mcp(decision, str(settings.policy_file), dry_run=settings.dry_run)
    )
    return PipelineResult(facts, current_policy, decision, action_result, raw_response)
