"""Vector store: embedding (sentence-transformers) + penyimpanan (ChromaDB)."""
import chromadb
from sentence_transformers import SentenceTransformer

from .config import CHROMA_DIR, EMBED_MODEL, PASSAGE_PREFIX, QUERY_PREFIX

BATCH = 256


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.col = self.client.get_or_create_collection(
            "modul", metadata={"hnsw:space": "cosine"}
        )
        # Download model sekali (~120 MB), setelah itu jalan offline.
        self.model = SentenceTransformer(EMBED_MODEL)

    def _embed(self, texts: list[str], prefix: str) -> list[list[float]]:
        vecs = self.model.encode(
            [prefix + t for t in texts], normalize_embeddings=True, show_progress_bar=False
        )
        return vecs.tolist()

    def add_document(self, doc_name: str, chunks: list[dict]) -> int:
        # Upload ulang file dengan nama sama = ganti isi lama.
        self.delete_document(doc_name)
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i : i + BATCH]
            self.col.add(
                ids=[f"{doc_name}::{i + j}" for j in range(len(batch))],
                documents=[c["text"] for c in batch],
                embeddings=self._embed([c["text"] for c in batch], PASSAGE_PREFIX),
                metadatas=[{"source": c["source"], "page": c["page"]} for c in batch],
            )
        return len(chunks)

    def search(self, query: str, k: int) -> list[dict]:
        if self.col.count() == 0:
            return []
        res = self.col.query(
            query_embeddings=self._embed([query], QUERY_PREFIX),
            n_results=min(k, self.col.count()),
        )
        hits = []
        for text, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append(
                {
                    "text": text,
                    "source": meta["source"],
                    "page": meta["page"],
                    "score": round(1 - dist, 4),  # cosine similarity
                }
            )
        return hits

    def list_documents(self) -> list[dict]:
        metas = self.col.get(include=["metadatas"])["metadatas"]
        counts: dict[str, int] = {}
        for m in metas:
            counts[m["source"]] = counts.get(m["source"], 0) + 1
        return [{"name": k, "chunks": v} for k, v in sorted(counts.items())]

    def delete_document(self, doc_name: str) -> None:
        self.col.delete(where={"source": doc_name})
