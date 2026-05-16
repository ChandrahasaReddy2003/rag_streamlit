import re
from typing import List, Dict


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*[-–—]?\s*\d+\s*[-–—]?\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def remove_repeated_headers_footers(pages: List[Dict]) -> List[Dict]:
    if len(pages) <= 2:
        for page in pages:
            page["text"] = clean_text(page.get("text", ""))
        return pages

    first_lines = {}
    last_lines = {}

    for page in pages:
        lines = [line.strip() for line in page.get("text", "").splitlines() if line.strip()]
        if not lines:
            continue
        first_lines[lines[0]] = first_lines.get(lines[0], 0) + 1
        last_lines[lines[-1]] = last_lines.get(lines[-1], 0) + 1

    threshold = max(2, int(len(pages) * 0.4))
    repeated = {
        line for line, count in {**first_lines, **last_lines}.items()
        if count >= threshold and len(line) < 160
    }

    cleaned_pages = []
    for page in pages:
        lines = [line.strip() for line in page.get("text", "").splitlines()]
        lines = [line for line in lines if line not in repeated]
        new_page = dict(page)
        new_page["text"] = clean_text("\n".join(lines))
        cleaned_pages.append(new_page)

    return cleaned_pages
