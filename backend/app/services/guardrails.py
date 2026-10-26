import re
from typing import List, Dict, Any

MAX_QUESTION_LENGTH = 2000

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?prior\s+instructions",
    r"reveal\s+(the\s+)?system\s+prompt",
    r"show\s+(me\s+)?(the\s+)?hidden\s+instructions",
    r"override\s+(all\s+)?safety\s+(rules|instructions)",
    r"act\s+as\s+(an?\s+)?(unrestricted|unfiltered)\s+ai",
    r"you\s+are\s+now\s+in\s+DAN\s+mode",
    r"bypass\s+all\s+(content\s+)?filters",
]


def check_input(question: str) -> dict:
    """Validate a question and flag common prompt-injection attempts."""
    if not isinstance(question, str):
        return {
            "allowed": False,
            "reason": "Question must be text.",
            "risk": "invalid_input",
        }

    question = question.strip()

    if not question:
        return {
            "allowed": False,
            "reason": "Question cannot be empty.",
            "risk": "invalid_input",
        }

    if len(question) > MAX_QUESTION_LENGTH:
        return {
            "allowed": False,
            "reason": "Question exceeds the 2000-character limit.",
            "risk": "input_too_long",
        }

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, question, flags=re.IGNORECASE):
            return {
                "allowed": False,
                "reason": "Potential prompt injection detected.",
                "risk": "prompt_injection",
            }

    return {
        "allowed": True,
        "reason": "Input passed basic security checks.",
        "risk": "low",
    }


def check_output(answer: str, passages: list[dict]) -> dict:
    """Perform basic output and citation consistency checks."""
    if not isinstance(answer, str) or not answer.strip():
        return {
            "passed": False,
            "reason": "The generated answer is empty.",
            "warnings": ["empty_answer"],
            "cited_pages": [],
            "available_pages": [str(p["page"]) for p in passages] if passages else [],
        }

    if not passages:
        return {
            "passed": False,
            "reason": "No document evidence was retrieved.",
            "warnings": ["missing_evidence"],
            "cited_pages": [],
            "available_pages": [],
        }

    available_pages = {
        str(passage["page"]) for passage in passages
    }

    cited_pages = set(
        re.findall(r"\[Page\s+(\d+)\]", answer, flags=re.IGNORECASE)
    )

    invalid_citations = cited_pages - available_pages
    warnings = []

    if not cited_pages:
        warnings.append("no_page_citations")

    if invalid_citations:
        warnings.append("citation_not_in_retrieved_sources")

    return {
        "passed": not invalid_citations,
        "reason": (
            "Output passed basic checks."
            if not invalid_citations
            else "Answer contains citations outside retrieved source pages."
        ),
        "warnings": warnings,
        "cited_pages": sorted(cited_pages),
        "available_pages": sorted(available_pages),
    }
