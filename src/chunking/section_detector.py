import re
from typing import List, Dict

HEADING_PATTERNS = [
    r"^\d+(\.\d+)*\s+[A-Z][A-Za-z0-9 ,:()/%&+\-]+$",
    r"^Chapter\s+\d+[:.\s]+.+$",
    r"^Annex\s+[A-Za-z0-9]+[:.\s]+.+$",
    r"^Appendix\s+[A-Za-z0-9]+[:.\s]+.+$",
    r"^[A-Z]\.\s+[A-Z][A-Za-z0-9 ,:()/%&+\-]+$",
    r"^[IVX]+\.\s+[A-Z][A-Za-z0-9 ,:()/%&+\-]+$",
    r"^Section\s+\d+[:.\s]+.+$",
]


def is_heading(line: str) -> bool:
    line = line.strip()
    if len(line) < 4 or len(line) > 150:
        return False

    # Avoid treating full sentences as headings.
    if line.endswith(".") and len(line.split()) > 8:
        return False

    for pattern in HEADING_PATTERNS:
        if re.match(pattern, line, flags=re.IGNORECASE):
            return True

    # Fallback for uppercase short headings, e.g., DOCUMENTATION REQUIREMENTS.
    alpha = [c for c in line if c.isalpha()]
    if alpha:
        uppercase_ratio = sum(c.isupper() for c in alpha) / len(alpha)
        if uppercase_ratio > 0.78 and len(line.split()) <= 10:
            return True

    return False


def get_heading_level(heading: str) -> int:
    heading = heading.strip()
    match = re.match(r"^(\d+(\.\d+)*)", heading)
    if match:
        return min(match.group(1).count(".") + 1, 6)

    if re.match(r"^(chapter|annex|appendix|section)\b", heading, flags=re.IGNORECASE):
        return 1

    if re.match(r"^[IVX]+\.\s+", heading):
        return 1

    if re.match(r"^[A-Z]\.\s+", heading):
        return 2

    return 2


def detect_sections(pages: List[Dict]) -> List[Dict]:
    sections = []
    current = None
    section_counter = 0
    heading_stack = []

    def close_current():
        if current and current["text"].strip():
            current["text"] = current["text"].strip()
            sections.append(dict(current))

    for page in pages:
        lines = page.get("text", "").splitlines()
        for line in lines:
            clean_line = line.strip()
            if not clean_line:
                if current:
                    current["text"] += "\n"
                continue

            if is_heading(clean_line):
                close_current()
                section_counter += 1
                level = get_heading_level(clean_line)
                heading_stack = heading_stack[:level - 1]
                heading_stack.append(clean_line)
                section_path = " > ".join(heading_stack)

                current = {
                    "section_id": f'{page["doc_id"]}_S{section_counter:03d}',
                    "doc_id": page["doc_id"],
                    "title": page["title"],
                    "source_file": page.get("source_file", ""),
                    "section_path": section_path,
                    "heading": clean_line,
                    "heading_level": level,
                    "page_start": page["page_number"],
                    "page_end": page["page_number"],
                    "text": ""
                }
            else:
                if current is None:
                    section_counter += 1
                    current = {
                        "section_id": f'{page["doc_id"]}_S{section_counter:03d}',
                        "doc_id": page["doc_id"],
                        "title": page["title"],
                        "source_file": page.get("source_file", ""),
                        "section_path": "Document Start",
                        "heading": "Document Start",
                        "heading_level": 1,
                        "page_start": page["page_number"],
                        "page_end": page["page_number"],
                        "text": ""
                    }

                current["text"] += clean_line + "\n"
                current["page_end"] = page["page_number"]

    close_current()
    return sections
