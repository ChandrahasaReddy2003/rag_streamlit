import re
from typing import Dict, List


class Reranker:
    """
    Lightweight offline reranker.

    This does not download a cross-encoder model. It reranks using token overlap
    and small metadata boosts for definitions, procedures, and requirements.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or "local-token-overlap-reranker"
        self.model = True
        self.load_error = None

    def is_available(self) -> bool:
        return True

    @staticmethod
    def _tokens(text: str) -> set:
        return set(re.findall(r"[a-zA-Z0-9_+-]+", (text or "").lower()))

    def rerank(self, query: str, candidate_chunks: List[Dict], top_k: int = 10) -> List[Dict]:
        if not candidate_chunks:
            return []

        query_tokens = self._tokens(query)
        query_lower = query.lower()
        reranked = []

        for rank, chunk in enumerate(candidate_chunks):
            chunk_text = chunk.get("embedding_text") or chunk.get("text", "")
            chunk_tokens = self._tokens(chunk_text)

            overlap = len(query_tokens.intersection(chunk_tokens))
            coverage = overlap / max(len(query_tokens), 1)

            boost = 0.0
            if chunk.get("contains_requirement"):
                boost += 0.04
            if chunk.get("contains_procedure"):
                boost += 0.04
            if chunk.get("contains_definition") and any(
                marker in query_lower for marker in ["define", "definition", "meaning", "what is", "what are"]
            ):
                boost += 0.10
            if chunk.get("chunk_type") == "table" and any(
                marker in query_lower for marker in ["limit", "value", "count", "number", "table"]
            ):
                boost += 0.05

            # Tiny rank prior so earlier fused results still matter when scores tie.
            rank_prior = 1.0 / (1000 + rank)

            item = dict(chunk)
            item["rerank_score"] = float(coverage + boost + rank_prior)
            reranked.append(item)

        return sorted(reranked, key=lambda item: item["rerank_score"], reverse=True)[:top_k]
