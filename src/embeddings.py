from __future__ import annotations

from typing import List, Dict, Any, Optional
import os
import json

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _sanitize_meta_value(v: Any) -> Any:
    """
    Chroma metadata values must be: str, int, float, bool, None.
    Convert lists/dicts to JSON strings.
    """
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _sanitize_meta_value(v) for k, v in meta.items()}


class EmbedStore:
    """
    Local embeddings + Chroma vector store.
    - Embeddings: SentenceTransformers (offline after first download)
    - Vector DB: Chroma (persisted on disk)
    """

    def __init__(
        self,
        persist_dir: str = "chroma",
        collection_name: str = "isps",
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        # Embedding model (local)
        self.model = SentenceTransformer(model_name)

        # Chroma client (persistent)
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # Collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embs = self.model.encode(texts, normalize_embeddings=True)
        return embs.tolist()

    def upsert_items(self, items: List[Dict[str, Any]]) -> None:
        """
        items: [{"id": "...", "text": "...", "type": "strategy/action", ...metadata}]
        """
        if not items:
            return

        ids = [it["id"] for it in items]
        docs = [it["text"] for it in items]

        # everything except id/text as metadata
        metas = []
        for it in items:
            meta = {k: v for k, v in it.items() if k not in ("id", "text")}
            metas.append(_sanitize_metadata(meta))

        embs = self.embed_texts(docs)

        self.collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embs,
        )

    def query(self, query_text: str, n_results: int = 5, where: Optional[dict] = None) -> Dict[str, Any]:
        """
        distance is cosine distance if hnsw:space=cosine (lower = better)
        """
        q_emb = self.embed_texts([query_text])[0]
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return res

    def get(self, ids: List[str]) -> Dict[str, Any]:
        return self.collection.get(ids=ids, include=["documents", "metadatas"])
