import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.config import CHUNKS_PATH, RAW_DOCS_DIR, CHUNK_STATS_PATH
from src.generation.answer_generator import AnswerGenerator
from src.retrieval.retrieval_service import RetrievalService
from src.utils.jsonl import load_jsonl


st.set_page_config(
    page_title="Complex RAG Retrieval Tester",
    layout="wide",
)

st.title("Complex RAG Retrieval Tester")
st.caption(
    "Adaptive chunking + FAISS dense retrieval + BM25 sparse retrieval + "
    "rank fusion + lightweight reranking + Groq LLM answer generation"
)

with st.sidebar:
    st.header("Settings")

    use_reranker = st.toggle(
        "Use lightweight reranker",
        value=False,
        help="Offline token-overlap reranker. No external model download required.",
    )

    include_neighbors = st.toggle(
        "Include previous/next chunks",
        value=False,
        help="Turn off for Groq free/on-demand tier to avoid token limit issues.",
    )

    final_top_k = st.slider("Final top-k chunks", 3, 15, 5)
    dense_top_k = st.slider("Dense top-k", 10, 60, 20)
    bm25_top_k = st.slider("BM25 top-k", 10, 60, 20)
    fusion_top_k = st.slider("Fusion top-k", 10, 50, 15)

    st.divider()
    st.write("Raw docs folder:")
    st.code(str(RAW_DOCS_DIR), language="text")
    st.write("Chunks file:")
    st.code(str(CHUNKS_PATH), language="text")


def clear_service_cache():
    load_service.clear()
    load_answer_generator.clear()


@st.cache_resource(show_spinner=True)
def load_service(use_reranker_value: bool):
    return RetrievalService(use_reranker=use_reranker_value)


@st.cache_resource(show_spinner=True)
def load_answer_generator():
    return AnswerGenerator()


if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""
if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = None


left, right = st.columns([2, 1])

with right:
    st.subheader("Project Status")

    raw_files = (
        sorted([p for p in RAW_DOCS_DIR.rglob("*") if p.is_file()])
        if RAW_DOCS_DIR.exists()
        else []
    )
    raw_pdf_files = [p for p in raw_files if p.suffix.lower() == ".pdf"]

    st.metric("Raw files", len(raw_files))
    st.metric("Raw PDF files", len(raw_pdf_files))

    if CHUNKS_PATH.exists():
        chunks = load_jsonl(CHUNKS_PATH)
        st.metric("Chunks", len(chunks))
        processed_docs = len(set(chunk["doc_id"] for chunk in chunks)) if chunks else 0
        st.metric("Processed documents", processed_docs)
    else:
        chunks = []
        st.metric("Chunks", 0)
        st.metric("Processed documents", 0)

    if CHUNK_STATS_PATH.exists():
        stats_df = pd.read_csv(CHUNK_STATS_PATH)
        st.write("Chunk stats preview")
        st.dataframe(stats_df.head(20), use_container_width=True)
    else:
        st.warning("Indexes not built yet. Run: python scripts/build_indexes.py")


with left:
    st.subheader("Search")

    example_queries = [
        "What are the key responsibilities of the quality unit in GMP manufacturing?",
        "What are the main expectations for data integrity in pharmaceutical manufacturing?",
        "What controls are expected for computerized systems used in GMP activities?",
        "What is the role of quality risk management in process validation?",
        "How should change control be managed in a pharmaceutical quality system?",
        "Compare the expectations for process validation in FDA guidance and EU GMP Annex 15.",
        "If a computerized system generates GMP records, what expectations apply to system validation, user access, audit trails, record retention, and data review?",
    ]

    selected_example = st.selectbox("Example query", [""] + example_queries)
    query = st.text_area("Enter your query", value=selected_example, height=100)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        search_clicked = st.button("Retrieve Chunks", type="primary")

    with col_b:
        if st.button("Reload Indexes"):
            clear_service_cache()
            st.session_state["last_result"] = None
            st.session_state["last_answer"] = None
            st.success("Cache cleared. Search again to reload indexes.")

    if search_clicked:
        if not query.strip():
            st.warning("Enter a query first.")
        else:
            try:
                service = load_service(use_reranker)
                result = service.retrieve(
                    query=query.strip(),
                    dense_top_k=dense_top_k,
                    bm25_top_k=bm25_top_k,
                    fusion_top_k=fusion_top_k,
                    final_top_k=final_top_k,
                    include_neighbors=include_neighbors,
                )

                st.session_state["last_result"] = result
                st.session_state["last_query"] = query.strip()
                st.session_state["last_answer"] = None

            except Exception as exc:
                st.error(str(exc))
                st.info("Make sure you have run: python scripts/build_indexes.py")

    result = st.session_state.get("last_result")

    if result:
        st.subheader("Query Analysis")
        st.json(result["query_info"])

        st.subheader("Retrieval Stats")
        st.json(result["stats"])

        st.subheader("Retrieved Context Chunks")

        for idx, chunk in enumerate(result["retrieved_chunks"], start=1):
            label = f"{idx}. {chunk['chunk_id']} | {chunk['chunk_type']} | {chunk['title']}"

            with st.expander(label, expanded=(idx <= 3)):
                c1, c2, c3 = st.columns(3)
                c1.metric("Doc ID", chunk["doc_id"])
                c2.metric("Tokens", chunk["token_count"])
                c3.metric("Pages", f"{chunk['page_start']}–{chunk['page_end']}")

                st.write("**Section Path:**")
                st.code(chunk["section_path"], language="text")

                st.write("**Breadcrumb:**")
                st.code(chunk["breadcrumb_text"], language="text")

                st.write("**Text:**")
                st.write(chunk["text"])

                with st.expander("Full chunk metadata"):
                    st.json(chunk)

        st.divider()
        st.subheader("LLM Answer")

        st.info(
            "For Groq free/on-demand tier, keep Final top-k around 4–5 and neighbors OFF "
            "to avoid token-per-minute limits."
        )

        if st.button("Generate Answer from Retrieved Chunks", type="secondary"):
            try:
                answer_generator = load_answer_generator()

                with st.spinner("Generating answer using Groq..."):
                    final_answer = answer_generator.answer(
                        query=st.session_state["last_query"],
                        chunks=result["retrieved_chunks"],
                    )

                st.session_state["last_answer"] = final_answer

            except Exception as exc:
                st.error(str(exc))
                st.info(
                    "Check .env, network/SSL settings, and token limits. "
                    "If Groq returns 413, reduce final_top_k and keep neighbors OFF."
                )

        if st.session_state.get("last_answer"):
            st.markdown(st.session_state["last_answer"])
