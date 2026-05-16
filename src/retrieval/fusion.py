from typing import List, Dict


def reciprocal_rank_fusion(result_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
    fused = {}
    sources = {}

    for results in result_lists:
        for rank, result in enumerate(results):
            chunk_id = result["chunk_id"]
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            sources.setdefault(chunk_id, set()).add(result.get("source", "unknown"))

    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)

    return [
        {
            "chunk_id": chunk_id,
            "score": score,
            "source": "+".join(sorted(sources.get(chunk_id, []))) or "fusion"
        }
        for chunk_id, score in ranked
    ]
