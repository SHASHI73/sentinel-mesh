"""Stage 2: local vector store over internal policy documents.

Fix vs. the original level3_rag.py / main_agent.py split: both entry points
now go through the same get_or_create_collection call, so there is no
order-dependent crash if the agent runs before the indexer has ever been
run once.
"""
from __future__ import annotations

from pathlib import Path

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:  # pragma: no cover
    chromadb = None
    embedding_functions = None


class PolicyVectorStore:
    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = "company_policies",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        if chromadb is None:
            raise RuntimeError("chromadb is not installed. Run `pip install chromadb`.")
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        # get_or_create everywhere: eliminates the original bug where the
        # indexer used get_or_create_collection but the agent used
        # get_collection and would crash if run first.
        self._collection = self._client.get_or_create_collection(
            name=collection_name, embedding_function=self._embed_fn
        )

    def index_document(self, doc_id: str, text: str, source: str, category: str = "compliance") -> None:
        self._collection.upsert(
            documents=[text],
            ids=[doc_id],
            metadatas=[{"source": source, "category": category}],
        )

    def query(self, query_text: str, n_results: int = 1) -> str | None:
        results = self._collection.query(query_texts=[query_text], n_results=n_results)
        docs = results.get("documents") or []
        if docs and docs[0]:
            return docs[0][0]
        return None
