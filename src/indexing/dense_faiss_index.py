import json
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np


class DenseFaissIndex:
    def __init__(self, index_path: Path, id_map_path: Path):
        self.index_path = index_path
        self.id_map_path = id_map_path
        self.index = None
        self.id_map = {}

    def build(self, embeddings, chunk_ids: List[str]) -> None:
        embeddings = np.asarray(embeddings, dtype="float32")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.id_map = {str(i): chunk_id for i, chunk_id in enumerate(chunk_ids)}

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.id_map_path, "w", encoding="utf-8") as f:
            json.dump(self.id_map, f, indent=2)

    def load(self) -> None:
        if not self.index_path.exists() or not self.id_map_path.exists():
            raise FileNotFoundError("FAISS index not found. Run: python scripts/build_indexes.py")

        self.index = faiss.read_index(str(self.index_path))
        with open(self.id_map_path, "r", encoding="utf-8") as f:
            self.id_map = json.load(f)

    def search(self, query_embedding, top_k: int = 30) -> List[Dict]:
        query_embedding = np.asarray([query_embedding], dtype="float32")
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "chunk_id": self.id_map[str(idx)],
                "score": float(score),
                "source": "dense"
            })

        return results
