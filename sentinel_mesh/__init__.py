"""Sentinel-Mesh: a local-first compliance monitoring agent.

Pipeline: NLP entity extraction -> RAG policy retrieval -> LLM conflict
reasoning (via a locally hosted Ollama model) -> gated, audited policy patch
through an MCP tool server.
"""

__version__ = "1.0.0"
