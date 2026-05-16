from pathlib import Path
from typing import Dict, List

from src.config import SUPPORTED_EXTENSIONS


def _read_pdf_pages(file_path: Path, doc_id: str, title: str) -> List[Dict]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError("PyMuPDF is required for PDF processing. Install with: pip install pymupdf") from exc

    pdf = fitz.open(file_path)
    pages = []

    for page_index, page in enumerate(pdf):
        text = page.get_text("text") or ""
        pages.append({
            "doc_id": doc_id,
            "title": title,
            "source_file": file_path.name,
            "page_number": page_index + 1,
            "text": text,
        })

    pdf.close()
    return pages


def _read_text_pages(file_path: Path, doc_id: str, title: str, words_per_page: int = 700) -> List[Dict]:
    text = file_path.read_text(encoding="utf-8", errors="ignore")

    if "\f" in text:
        raw_pages = text.split("\f")
    else:
        words = text.split()
        raw_pages = [
            " ".join(words[i:i + words_per_page])
            for i in range(0, len(words), words_per_page)
        ] or [text]

    pages = []
    for page_index, page_text in enumerate(raw_pages):
        pages.append({
            "doc_id": doc_id,
            "title": title,
            "source_file": file_path.name,
            "page_number": page_index + 1,
            "text": page_text,
        })

    return pages


def _clean_title_from_filename(file_path: Path) -> str:
    title = file_path.stem
    # Keep title readable while chunk IDs use doc_id separately.
    title = title.replace("_", " ").replace("-", " ").strip()
    return title.title()


def load_document_pages(file_path: Path, doc_id: str) -> List[Dict]:
    suffix = file_path.suffix.lower().strip()
    title = _clean_title_from_filename(file_path)

    if suffix == ".pdf":
        return _read_pdf_pages(file_path, doc_id, title)

    if suffix in {".txt", ".md", ".markdown"}:
        return _read_text_pages(file_path, doc_id, title)

    raise ValueError(f"Unsupported file type: {file_path.name}")


def discover_documents(raw_docs_dir: Path) -> List[Path]:
    files = [
        path for path in raw_docs_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower().strip() in SUPPORTED_EXTENSIONS
        and not path.name.startswith("~$")
    ]

    files = sorted(files, key=lambda p: p.name.lower())
    print(f"Found {len(files)} supported documents in {raw_docs_dir}")
    return files
