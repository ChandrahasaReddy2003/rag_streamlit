from typing import Dict


def analyze_query(query: str) -> Dict:
    q = query.lower()

    exact_markers = [
        "section", "annex", "chapter", "clause", "ich", "fda", "who",
        "q7", "q8", "q9", "q10", "q11", "alcoa", "capa", "sop", "gmp"
    ]
    procedure_markers = [
        "steps", "procedure", "how to", "process", "workflow", "sequence",
        "perform", "handle", "implement", "what should be done"
    ]
    comparison_markers = [
        "compare", "difference", "differentiate", "versus", "vs", "similarities"
    ]
    definition_markers = [
        "define", "definition", "meaning", "what is", "what are", "refers to"
    ]
    no_answer_markers = [
        "approved batch number", "specific lot", "employee id", "password", "token"
    ]

    return {
        "query": query,
        "is_exact": any(marker in q for marker in exact_markers),
        "is_procedure": any(marker in q for marker in procedure_markers),
        "is_comparison": any(marker in q for marker in comparison_markers),
        "is_definition": any(marker in q for marker in definition_markers),
        "may_be_no_answer": any(marker in q for marker in no_answer_markers),
    }
