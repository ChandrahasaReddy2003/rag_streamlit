# Complex RAG Project

VS Code / PyCharm ready RAG project for adaptive chunking, hybrid retrieval, reranking, and Groq-based answer generation.

## What is included

- PDF, TXT, and Markdown ingestion
- Header/footer cleanup
- Section detection
- Adaptive chunking with metadata-rich chunk objects
- FAISS dense retrieval using an offline hashing-vector embedding fallback
- BM25 sparse retrieval
- Reciprocal Rank Fusion
- Lightweight offline reranking
- Previous/next chunk context expansion
- Streamlit app for retrieval and LLM answer testing
- Groq LLM integration using `.env`
- `truststore` SSL fix for corporate Windows laptops
- Token-safe context trimming for Groq free/on-demand limits

## Folder structure

```text
complex_rag_project/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw_docs/          # Put PDFs/TXT/MD files here
│   ├── processed/         # Generated chunks/pages/sections
│   └── manifest/
├── indexes/
│   ├── faiss/
│   └── bm25/
├── scripts/
│   ├── build_indexes.py
│   ├── test_groq_connection.py
│   └── test_retrieval.py
├── src/
│   ├── chunking/
│   ├── generation/
│   ├── indexing/
│   ├── ingestion/
│   ├── retrieval/
│   └── utils/
├── .env.example
├── requirements.txt
└── README.md
```

## Setup in VS Code

Open the `complex_rag_project` folder in VS Code terminal.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Add Groq key

Copy `.env.example` to `.env`.

```env
GROQ_API_KEY=your_regenerated_groq_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

For better answer quality, try:

```env
GROQ_MODEL=llama-3.3-70b-versatile
```

Do not commit `.env` to GitHub.

## Add documents

Put your PDFs inside:

```text
data/raw_docs/
```

You can also keep PDFs in subfolders. The loader searches recursively.

## Build chunks and indexes

```powershell
python scripts/build_indexes.py
```

Expected outputs:

```text
data/processed/pages.jsonl
data/processed/sections.jsonl
data/processed/chunks.jsonl
data/processed/chunk_stats.csv
indexes/faiss/index.faiss
indexes/faiss/id_map.json
indexes/bm25/bm25.pkl
```

## Test Groq connection

```powershell
python scripts/test_groq_connection.py
```

Expected:

```text
API key found: True
Status code: 200
```

## Run Streamlit app

```powershell
streamlit run app/streamlit_app.py
```

Recommended app settings for Groq free/on-demand tier:

```text
Final top-k chunks: 4 or 5
Include previous/next chunks: OFF
Dense top-k: 20
BM25 top-k: 20
Fusion top-k: 15
```

This avoids Groq token-per-minute 413 errors.

## Good test questions

```text
What are the key responsibilities of the quality unit in GMP manufacturing?
```

```text
What are the main expectations for data integrity in pharmaceutical manufacturing?
```

```text
What controls are expected for computerized systems used in GMP activities?
```

```text
What is the role of quality risk management in process validation?
```

```text
Compare the expectations for process validation in FDA guidance and EU GMP Annex 15.
```

## Important notes

- The current dense retrieval uses a local hashing-vector fallback so the project runs without downloading Hugging Face models.
- BM25 remains strong for exact regulatory terms such as Annex 11, Q7, ALCOA+, CAPA, audit trail, and validation.
- For a final hackathon version, replace `src/indexing/embedding_model.py` with Azure OpenAI / OpenAI / Gemini embeddings for better semantic retrieval.
- If Groq returns a 413 error, reduce final top-k and disable neighbor chunks.
