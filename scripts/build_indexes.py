import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import pandas as pd
from tqdm import tqdm

from src.config import (
    RAW_DOCS_DIR,
    PAGES_PATH,
    SECTIONS_PATH,
    CHUNKS_PATH,
    CHUNK_STATS_PATH,
    FAISS_INDEX_PATH,
    FAISS_ID_MAP_PATH,
    BM25_INDEX_PATH,
)
from src.ingestion.document_loader import discover_documents, load_document_pages
from src.ingestion.text_cleaner import remove_repeated_headers_footers
from src.chunking.section_detector import detect_sections
from src.chunking.adaptive_chunker import adaptive_chunk_section
from src.chunking.chunk_linker import link_chunks
from src.indexing.embedding_model import EmbeddingModel
from src.indexing.dense_faiss_index import DenseFaissIndex
from src.indexing.bm25_index import BM25Index
from src.utils.jsonl import save_jsonl


def build_chunks():
    all_pages = []
    all_sections = []
    all_chunks = []

    files = discover_documents(RAW_DOCS_DIR)
    if not files:
        raise FileNotFoundError(f"No supported files found in {RAW_DOCS_DIR}")

    print(f"Building chunks for {len(files)} documents...")

    for index, file_path in enumerate(tqdm(files, desc="Processing documents"), start=1):
        # Prefer a stable doc_id from filenames such as D01_..., D17_..., etc.
        # This keeps citations stable even if file ordering changes.
        match = re.match(r"^(D\d+)", file_path.stem, re.IGNORECASE)
        if match:
            doc_id = match.group(1).upper()
        else:
            doc_id = f"D{index:02d}"

        pages = load_document_pages(file_path, doc_id=doc_id)
        pages = remove_repeated_headers_footers(pages)
        sections = detect_sections(pages)

        for section in sections:
            section_chunks = adaptive_chunk_section(section)
            all_chunks.extend(section_chunks)

        all_pages.extend(pages)
        all_sections.extend(sections)

    all_chunks = link_chunks(all_chunks)

    save_jsonl(all_pages, PAGES_PATH)
    save_jsonl(all_sections, SECTIONS_PATH)
    save_jsonl(all_chunks, CHUNKS_PATH)

    stats_df = pd.DataFrame([
        {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "title": chunk["title"],
            "chunk_type": chunk["chunk_type"],
            "token_count": chunk["token_count"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "section_path": chunk["section_path"],
        }
        for chunk in all_chunks
    ])
    CHUNK_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(CHUNK_STATS_PATH, index=False)

    print(f"Pages saved: {len(all_pages)} -> {PAGES_PATH}")
    print(f"Sections saved: {len(all_sections)} -> {SECTIONS_PATH}")
    print(f"Chunks saved: {len(all_chunks)} -> {CHUNKS_PATH}")
    print(f"Chunk stats saved -> {CHUNK_STATS_PATH}")

    return all_chunks


def build_indexes(chunks):
    if not chunks:
        raise ValueError("No chunks available for indexing.")

    print("Loading embedding model...")
    embedding_model = EmbeddingModel()

    texts = [chunk.get("embedding_text") or chunk.get("text", "") for chunk in chunks]
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    print("Creating dense embeddings...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    print("Building FAISS index...")
    dense_index = DenseFaissIndex(
        index_path=FAISS_INDEX_PATH,
        id_map_path=FAISS_ID_MAP_PATH,
    )
    dense_index.build(embeddings, chunk_ids)
    dense_index.save()

    print("Building BM25 index...")
    bm25_index = BM25Index(BM25_INDEX_PATH)
    bm25_index.build(chunks)
    bm25_index.save()

    print("Indexing completed successfully.")


def main():
    chunks = build_chunks()
    build_indexes(chunks)


if __name__ == "__main__":
    main()
