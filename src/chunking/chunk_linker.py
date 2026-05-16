from typing import List, Dict


def link_chunks(chunks: List[Dict]) -> List[Dict]:
    chunks = sorted(chunks, key=lambda item: (item["doc_id"], item["chunk_id"]))

    for index, chunk in enumerate(chunks):
        if index > 0 and chunks[index - 1]["doc_id"] == chunk["doc_id"]:
            chunk["prev_chunk_id"] = chunks[index - 1]["chunk_id"]

        if index < len(chunks) - 1 and chunks[index + 1]["doc_id"] == chunk["doc_id"]:
            chunk["next_chunk_id"] = chunks[index + 1]["chunk_id"]

    return chunks
