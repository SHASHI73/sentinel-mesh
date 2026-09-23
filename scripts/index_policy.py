#!/usr/bin/env python3
"""Index (or re-index) the internal policy document into the vector store.
Run this once before run_agent.py, and again any time the policy file
changes outside of the agent itself."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel_mesh import config
from sentinel_mesh.rag_store import PolicyVectorStore


def main() -> None:
    text = config.DEFAULT_POLICY_FILE.read_text(encoding="utf-8")
    store = PolicyVectorStore(
        persist_dir=config.CHROMA_PERSIST_DIR,
        collection_name=config.RAG_COLLECTION_NAME,
        embedding_model=config.EMBEDDING_MODEL,
    )
    store.index_document(
        doc_id="doc_sec_2026_ret01",
        text=text,
        source=str(config.DEFAULT_POLICY_FILE.name),
    )
    print(f"Indexed {config.DEFAULT_POLICY_FILE} into collection '{config.RAG_COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
