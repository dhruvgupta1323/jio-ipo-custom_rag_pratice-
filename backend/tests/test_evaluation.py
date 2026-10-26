
from backend.app.services.evaluation import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    keyword_coverage,
)


def test_precision_at_k():
    assert precision_at_k([1, 2, 3], [1, 3], k=3) == 2 / 3


def test_recall_at_k():
    assert recall_at_k([1, 2, 3], [1, 3], k=3) == 1.0


def test_mrr():
    assert mean_reciprocal_rank([2, 1, 3], [1]) == 0.5


def test_keyword_coverage():
    score = keyword_coverage(
        "The effective date was 31/08/2026.",
        ["31/08/2026", "cancelled"],
    )
    assert score == 0.5


def test_no_relevant_pages():
    assert recall_at_k([1, 2], [], k=3) == 0.0
