import pickle
import re
from pathlib import Path
from typing import List, Dict

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[a-zA-Z0-9_+\-/]+", text)


class BM25Index:
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.bm25 = None
        self.chunk_ids = []
        self.corpus_tokens = []

    def build(self, chunks: List[Dict]) -> None:
        self.chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        self.corpus_tokens = [
            tokenize(chunk.get("embedding_text") or chunk.get("text", ""))
            for chunk in chunks
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "chunk_ids": self.chunk_ids,
                "corpus_tokens": self.corpus_tokens,
            }, f)

    def load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError("BM25 index not found. Run: python scripts/build_indexes.py")

        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunk_ids = data["chunk_ids"]
        self.corpus_tokens = data["corpus_tokens"]

    def search(self, query: str, top_k: int = 30) -> List[Dict]:
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True
        )[:top_k]

        return [
            {
                "chunk_id": self.chunk_ids[idx],
                "score": float(score),
                "source": "bm25"
            }
            for idx, score in ranked
            if score > 0
        ]
