"""Central configuration for Sentinel-Mesh.

Keeping this in one place makes two things auditable at a glance:
1. Which local models the pipeline depends on.
2. Exactly which policy fields the agent is allowed to touch, and the
   legal range of values for each one. This allowlist is the main defense
   against a prompt-injected or hallucinated patch writing something
   unintended into a production config file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_POLICY_FILE = DATA_DIR / "internal_policy.txt"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_store"
AUDIT_LOG_FILE = DATA_DIR / "audit_log.jsonl"

OLLAMA_MODEL = "llama3.2:1b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_sm"

RAG_COLLECTION_NAME = "company_policies"


@dataclass(frozen=True)
class PolicyField:
    """Describes one field the agent is permitted to modify, and the
    bounds a new value must satisfy before a patch is allowed to run."""

    name: str
    current_pattern: str  # substring expected in the policy file today
    min_value: int
    max_value: int
    unit: str = "days"


# Allowlist: the agent may ONLY ever patch these named fields, and only to
# a value inside the stated bounds. Anything the model proposes outside of
# this list, or outside the numeric range, is rejected before any file
# write is attempted.
ALLOWED_POLICY_FIELDS: dict[str, PolicyField] = {
    "transaction_log_retention": PolicyField(
        name="transaction_log_retention",
        current_pattern="preserved natively for a standard duration of",
        min_value=1,
        max_value=365,
        unit="days",
    ),
    "session_cache_ttl": PolicyField(
        name="session_cache_ttl",
        current_pattern="set to clear after",
        min_value=1,
        max_value=720,
        unit="hours",
    ),
}


@dataclass
class AgentSettings:
    ollama_model: str = OLLAMA_MODEL
    embedding_model: str = EMBEDDING_MODEL
    spacy_model: str = SPACY_MODEL
    policy_file: Path = DEFAULT_POLICY_FILE
    audit_log_file: Path = AUDIT_LOG_FILE
    chroma_dir: Path = CHROMA_PERSIST_DIR
    dry_run: bool = True
    allowed_fields: dict[str, PolicyField] = field(default_factory=lambda: ALLOWED_POLICY_FIELDS)
