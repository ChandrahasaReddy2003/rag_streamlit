from typing import Dict, List

from src.generation.groq_client import GroqLLMClient


class AnswerGenerator:
    def __init__(self):
        self.llm = GroqLLMClient()

    @staticmethod
    def approx_tokens(text: str) -> int:
        # Conservative approximation: 1 token is roughly 4 characters.
        return max(1, len(text or "") // 4)

    @staticmethod
    def trim_text(text: str, max_chars: int = 2200) -> str:
        if not text:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n\n[Text truncated for token limit]"

    def build_context_text(self, chunks: List[Dict], max_context_tokens: int = 3500) -> str:
        blocks = []
        used_tokens = 0

        for i, chunk in enumerate(chunks, start=1):
            chunk_text = self.trim_text(chunk.get("text", ""), max_chars=2200)
            chunk_id = chunk.get("chunk_id")

            block = f"""
[RETRIEVED_CHUNK_{i}]
chunk_id: {chunk_id}
doc_id: {chunk.get("doc_id")}
title: {chunk.get("title")}
section_path: {chunk.get("section_path")}
pages: {chunk.get("page_start")} - {chunk.get("page_end")}
chunk_type: {chunk.get("chunk_type")}

text:
{chunk_text}
""".strip()

            block_tokens = self.approx_tokens(block)
            if used_tokens + block_tokens > max_context_tokens:
                break

            blocks.append(block)
            used_tokens += block_tokens

        return "\n\n---\n\n".join(blocks)

    def answer(self, query: str, chunks: List[Dict]) -> str:
        context_text = self.build_context_text(chunks=chunks, max_context_tokens=3500)

        system_prompt = """
You are a strict RAG answer generator.

Rules:
1. Use only the retrieved context.
2. Do not use outside knowledge.
3. Cite every important claim using the exact chunk_id value from the context, for example [D04_S023_001].
4. Never cite as "Chunk 1", "Chunk 2", "the retrieved context", or "the document" without chunk_id.
5. If the context is insufficient, say exactly: "The retrieved context does not contain enough information to answer this."
6. Do not invent document names, page numbers, requirements, examples, standards, or procedures.
7. If the question asks for comparison, compare only what is present in the chunks.
8. Keep the answer structured and professional.
9. Prefer concise answers.
"""

        user_prompt = f"""
Question:
{query}

Retrieved context:
{context_text}

Generate the final answer using only the retrieved context.
"""

        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]

        return self.llm.generate(
            messages=messages,
            temperature=0.1,
            max_tokens=700,
        )
