from typing import List, Dict


def build_context(
    selected_chunks: List[Dict],
    chunk_store: Dict[str, Dict],
    include_neighbors: bool = True,
    max_chunks: int = 10,
) -> List[Dict]:
    final = []
    seen = set()

    def add_chunk(chunk_id: str):
        if chunk_id and chunk_id in chunk_store and chunk_id not in seen:
            final.append(chunk_store[chunk_id])
            seen.add(chunk_id)

    for chunk in selected_chunks:
        if include_neighbors:
            add_chunk(chunk.get("prev_chunk_id"))

        add_chunk(chunk["chunk_id"])

        if include_neighbors:
            add_chunk(chunk.get("next_chunk_id"))

        if len(final) >= max_chunks:
            break

    return final[:max_chunks]
