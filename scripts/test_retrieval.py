import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.retrieval.retrieval_service import RetrievalService


def main():
    service = RetrievalService(use_reranker=False)
    query = input("Enter query: ").strip()
    result = service.retrieve(query, final_top_k=5)

    print("\nQuery Analysis:")
    print(result["query_info"])

    print("\nRetrieved Chunks:")
    for idx, chunk in enumerate(result["retrieved_chunks"], start=1):
        print("=" * 80)
        print(f"{idx}. {chunk['chunk_id']} | {chunk['chunk_type']} | {chunk['title']}")
        print(f"Section: {chunk['section_path']}")
        print(f"Pages: {chunk['page_start']} - {chunk['page_end']}")
        print(chunk["text"][:1000])


if __name__ == "__main__":
    main()
