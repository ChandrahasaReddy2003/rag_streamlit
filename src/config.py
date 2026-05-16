from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"
PROCESSED_DIR = DATA_DIR / "processed"

INDEX_DIR = BASE_DIR / "indexes"
FAISS_DIR = INDEX_DIR / "faiss"
BM25_DIR = INDEX_DIR / "bm25"
METADATA_DIR = INDEX_DIR / "metadata"

PAGES_PATH = PROCESSED_DIR / "pages.jsonl"
SECTIONS_PATH = PROCESSED_DIR / "sections.jsonl"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
CHUNK_STATS_PATH = PROCESSED_DIR / "chunk_stats.csv"

FAISS_INDEX_PATH = FAISS_DIR / "index.faiss"
FAISS_ID_MAP_PATH = FAISS_DIR / "id_map.json"
BM25_INDEX_PATH = BM25_DIR / "bm25.pkl"

# The current implementation uses offline local hashing embeddings and a lightweight reranker.
# These names are kept for future replacement with real embedding/reranker models.
EMBEDDING_MODEL_NAME = "local-hashing-vectorizer"
RERANKER_MODEL_NAME = "local-token-overlap-reranker"

MIN_CHUNK_TOKENS = 100
TARGET_CHUNK_TOKENS = 450
MAX_CHUNK_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 80

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}
