from typing import Dict, List

from src.config import CHUNKS_PATH, FAISS_INDEX_PATH, FAISS_ID_MAP_PATH, BM25_INDEX_PATH
from src.utils.jsonl import load_jsonl
from src.indexing.embedding_model import EmbeddingModel
from src.indexing.dense_faiss_index import DenseFaissIndex
from src.indexing.bm25_index import BM25Index
from src.retrieval.query_analyzer import analyze_query
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import Reranker
from src.retrieval.context_builder import build_context


class RetrievalService:
    def __init__(self, use_reranker: bool = True):
        self.chunks = load_jsonl(CHUNKS_PATH)
        if not self.chunks:
            raise FileNotFoundError("No chunks found. Run: python scripts/build_indexes.py")

        self.chunk_store = {chunk["chunk_id"]: chunk for chunk in self.chunks}

        self.embedding_model = EmbeddingModel()

        self.dense_index = DenseFaissIndex(
            index_path=FAISS_INDEX_PATH,
            id_map_path=FAISS_ID_MAP_PATH
        )
        self.dense_index.load()

        self.bm25_index = BM25Index(BM25_INDEX_PATH)
        self.bm25_index.load()

        self.use_reranker = use_reranker
        self.reranker = Reranker() if use_reranker else None

    def retrieve(
        self,
        query: str,
        dense_top_k: int = 30,
        bm25_top_k: int = 30,
        fusion_top_k: int = 25,
        final_top_k: int = 8,
        include_neighbors: bool = True,
    ) -> Dict:
        query_info = analyze_query(query)
        query_embedding = self.embedding_model.encode([query], show_progress_bar=False)[0]

        dense_results = self.dense_index.search(query_embedding=query_embedding, top_k=dense_top_k)
        bm25_results = self.bm25_index.search(query=query, top_k=bm25_top_k)

        fused_results = reciprocal_rank_fusion([dense_results, bm25_results])[:fusion_top_k]
        candidate_chunks = [
            self.chunk_store[item["chunk_id"]]
            for item in fused_results
            if item["chunk_id"] in self.chunk_store
        ]

        reranker_used = False
        reranker_error = None

        if self.use_reranker and self.reranker:
            reranker_used = self.reranker.is_available()
            reranker_error = self.reranker.load_error
            ranked_chunks = self.reranker.rerank(query=query, candidate_chunks=candidate_chunks, top_k=final_top_k)
        else:
            ranked_chunks = candidate_chunks[:final_top_k]

        final_context = build_context(
            selected_chunks=ranked_chunks,
            chunk_store=self.chunk_store,
            include_neighbors=include_neighbors,
            max_chunks=max(final_top_k, 10)
        )

        return {
            "query": query,
            "query_info": query_info,
            "stats": {
                "total_chunks": len(self.chunks),
                "dense_results": len(dense_results),
                "bm25_results": len(bm25_results),
                "fused_results": len(fused_results),
                "final_context_chunks": len(final_context),
                "reranker_used": reranker_used,
                "reranker_error": reranker_error,
            },
            "dense_results": dense_results[:10],
            "bm25_results": bm25_results[:10],
            "retrieved_chunks": final_context,
        }
