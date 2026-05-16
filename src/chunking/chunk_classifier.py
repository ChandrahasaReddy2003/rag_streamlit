import re
from typing import Dict


def classify_chunk(text: str) -> Dict:
    lower = text.lower()

    contains_table = "|" in text or "\t" in text or bool(re.search(r"\b(column|row|table)\b", lower))
    contains_definition = bool(re.search(r"\b(is defined as|means|refers to|definition|defined as)\b", lower))
    contains_procedure = bool(re.search(r"\b(step|procedure|perform|workflow|sequence|responsible for|shall be|must be|required to)\b", lower))
    contains_requirement = bool(re.search(r"\b(shall|must|required|should|ensure|responsible|mandatory|expectation)\b", lower))

    if contains_table:
        chunk_type = "table"
    elif bool(re.search(r"\b(q:|a:|question:|answer:)\b", lower)):
        chunk_type = "qa"
    elif contains_definition:
        chunk_type = "definition"
    elif contains_procedure:
        chunk_type = "procedure"
    elif bool(re.search(r"\b(warning|caution|critical|do not)\b", lower)):
        chunk_type = "warning"
    else:
        chunk_type = "section"

    return {
        "chunk_type": chunk_type,
        "contains_table": contains_table,
        "contains_definition": contains_definition,
        "contains_procedure": contains_procedure,
        "contains_requirement": contains_requirement,
    }
