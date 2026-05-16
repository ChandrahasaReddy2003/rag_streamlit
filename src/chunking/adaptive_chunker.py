import re
from typing import List, Dict

from src.config import MIN_CHUNK_TOKENS, TARGET_CHUNK_TOKENS, MAX_CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS
from src.chunking.chunk_classifier import classify_chunk


def count_tokens(text: str) -> int:
    return len(text.split())


def split_into_paragraphs(text: str) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    return paragraphs


def split_long_text_by_sentence(text: str, max_tokens: int = MAX_CHUNK_TOKENS) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts = []
    buffer = []
    buffer_tokens = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_tokens = count_tokens(sentence)
        if buffer and buffer_tokens + sentence_tokens > max_tokens:
            parts.append(" ".join(buffer))
            buffer = []
            buffer_tokens = 0
        buffer.append(sentence)
        buffer_tokens += sentence_tokens

    if buffer:
        parts.append(" ".join(buffer))
    return parts


def get_overlap_text(previous_text: str, overlap_tokens: int = CHUNK_OVERLAP_TOKENS) -> str:
    words = previous_text.split()
    if len(words) <= overlap_tokens:
        return previous_text
    return " ".join(words[-overlap_tokens:])


def build_embedding_text(title: str, section_path: str, breadcrumb_text: str, text: str) -> str:
    return f"{title}\n{section_path}\n{breadcrumb_text}\n\n{text}"


def create_chunk(section: Dict, text: str, chunk_counter: int) -> Dict:
    token_count = count_tokens(text)
    classification = classify_chunk(text)
    breadcrumb_text = f'{section["title"]} > {section["section_path"]}'
    chunk_id = f'{section["section_id"]}_{chunk_counter:03d}'

    return {
        "chunk_id": chunk_id,
        "doc_id": section["doc_id"],
        "title": section["title"],
        "section_path": section["section_path"],
        "page_start": section["page_start"],
        "page_end": section["page_end"],
        "chunk_type": classification["chunk_type"],
        "text": text.strip(),
        "breadcrumb_text": breadcrumb_text,
        "token_count": token_count,
        "prev_chunk_id": None,
        "next_chunk_id": None,
        "parent_section_id": section["section_id"],
        "heading_level": section["heading_level"],
        "contains_table": classification["contains_table"],
        "contains_definition": classification["contains_definition"],
        "contains_procedure": classification["contains_procedure"],
        "contains_requirement": classification["contains_requirement"],
        "keywords": [],
        "source_file": section.get("source_file", ""),
        "embedding_text": build_embedding_text(
            section["title"],
            section["section_path"],
            breadcrumb_text,
            text.strip()
        )
    }


def adaptive_chunk_section(section: Dict) -> List[Dict]:
    text = section.get("text", "").strip()
    if not text:
        return []

    # Keep small sections as one chunk.
    if count_tokens(text) <= MAX_CHUNK_TOKENS:
        return [create_chunk(section, text, 1)]

    paragraphs = split_into_paragraphs(text)
    chunks = []
    buffer = []
    buffer_tokens = 0
    chunk_counter = 0
    last_chunk_text = ""

    for paragraph in paragraphs:
        paragraph_tokens = count_tokens(paragraph)

        # Break extremely long paragraphs at sentence boundaries.
        paragraph_parts = [paragraph]
        if paragraph_tokens > MAX_CHUNK_TOKENS:
            paragraph_parts = split_long_text_by_sentence(paragraph, MAX_CHUNK_TOKENS)

        for part in paragraph_parts:
            part_tokens = count_tokens(part)

            if buffer and buffer_tokens + part_tokens > MAX_CHUNK_TOKENS:
                chunk_text = "\n\n".join(buffer).strip()
                chunk_counter += 1
                chunks.append(create_chunk(section, chunk_text, chunk_counter))
                last_chunk_text = chunk_text

                overlap = get_overlap_text(last_chunk_text)
                buffer = [overlap] if overlap else []
                buffer_tokens = count_tokens(overlap)

            buffer.append(part)
            buffer_tokens += part_tokens

            # Flush once we reach target size and next paragraph would likely overflow.
            if buffer_tokens >= TARGET_CHUNK_TOKENS:
                chunk_text = "\n\n".join(buffer).strip()
                if count_tokens(chunk_text) >= MIN_CHUNK_TOKENS:
                    chunk_counter += 1
                    chunks.append(create_chunk(section, chunk_text, chunk_counter))
                    last_chunk_text = chunk_text
                    overlap = get_overlap_text(last_chunk_text)
                    buffer = [overlap] if overlap else []
                    buffer_tokens = count_tokens(overlap)

    if buffer:
        final_text = "\n\n".join(buffer).strip()
        # Avoid creating a tiny trailing chunk. Merge into previous if too small.
        if chunks and count_tokens(final_text) < MIN_CHUNK_TOKENS:
            merged_text = chunks[-1]["text"] + "\n\n" + final_text
            chunk_counter = len(chunks)
            chunks[-1] = create_chunk(section, merged_text, chunk_counter)
        else:
            chunk_counter += 1
            chunks.append(create_chunk(section, final_text, chunk_counter))

    return chunks
