import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


def precision_at_k(
    retrieved_pages: List[int],
    relevant_pages: List[int],
    k: int = 3,
) -> float:
    retrieved = retrieved_pages[:k]
    if not retrieved:
        return 0.0

    relevant = set(relevant_pages)
    if not relevant:
        return 1.0 if not retrieved else 0.0

    hits = sum(1 for page in retrieved if page in relevant)
    return hits / len(retrieved)


def recall_at_k(
    retrieved_pages: List[int],
    relevant_pages: List[int],
    k: int = 3,
) -> float:
    relevant = set(relevant_pages)
    if not relevant:
        return 1.0 if not retrieved_pages[:k] else 0.0

    retrieved = set(retrieved_pages[:k])
    hits = len(retrieved.intersection(relevant))
    return hits / len(relevant)


def mean_reciprocal_rank(
    retrieved_pages: List[int],
    relevant_pages: List[int],
) -> float:
    relevant = set(relevant_pages)
    if not relevant:
        return 1.0 if not retrieved_pages else 0.0

    for rank, page in enumerate(retrieved_pages, start=1):
        if page in relevant:
            return 1.0 / rank

    return 0.0


def keyword_coverage(
    answer: str,
    expected_keywords: List[str],
) -> float:
    if not expected_keywords:
        return 1.0

    answer_lower = answer.lower()
    matched = sum(
        1 for keyword in expected_keywords
        if keyword.lower() in answer_lower
    )
    return matched / len(expected_keywords)


def load_dataset(
    path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    dataset_path = Path(path) if path else settings.evaluation_dataset_abs
    if not dataset_path.exists():
        fallback_path = settings.BASE_DIR / "evaluation" / "test_dataset.json"
        if fallback_path.exists():
            dataset_path = fallback_path
        else:
            raise FileNotFoundError(f"Evaluation dataset not found at {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as file:
        return json.load(file)


def run_evaluation_suite(
    retriever,
    generator,
    dataset_path: Optional[str] = None,
    k: int = 3,
) -> Dict[str, Any]:
    """Execute end-to-end RAG evaluation across all benchmark queries."""
    from backend.app.services.guardrails import check_input, check_output

    dataset = load_dataset(dataset_path)
    results = []

    start_total_time = time.perf_counter()

    for item in dataset:
        q_id = item["id"]
        question = item["question"]
        relevant_pages = item.get("relevant_pages", [])
        expected_keywords = item.get("expected_keywords", [])

        t0 = time.perf_counter()

        # Guardrails check
        input_check = check_input(question)
        if not input_check["allowed"]:
            results.append({
                "id": q_id,
                "question": question,
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "mrr": 0.0,
                "keyword_coverage": 0.0,
                "retrieved_pages": [],
                "relevant_pages": relevant_pages,
                "guardrail_passed": False,
                "generated_answer": f"Blocked: {input_check['reason']}",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            })
            continue

        # Retrieval
        retrieved_passages = retriever.retrieve(question, top_k=k)
        retrieved_pages = [p["page"] for p in retrieved_passages]

        # Generation
        gen_result = generator.generate_answer(question, retrieved_passages)
        answer = gen_result.get("answer", "")
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Output guardrails
        output_check = check_output(answer, retrieved_passages)

        p_at_k = precision_at_k(retrieved_pages, relevant_pages, k=k)
        r_at_k = recall_at_k(retrieved_pages, relevant_pages, k=k)
        mrr = mean_reciprocal_rank(retrieved_pages, relevant_pages)
        kw_cov = keyword_coverage(answer, expected_keywords)

        results.append({
            "id": q_id,
            "question": question,
            "precision_at_k": round(p_at_k, 4),
            "recall_at_k": round(r_at_k, 4),
            "mrr": round(mrr, 4),
            "keyword_coverage": round(kw_cov, 4),
            "retrieved_pages": retrieved_pages,
            "relevant_pages": relevant_pages,
            "guardrail_passed": output_check["passed"],
            "generated_answer": answer,
            "latency_ms": latency_ms,
        })

    total_time = round(time.perf_counter() - start_total_time, 2)
    n = len(results) or 1

    summary = {
        "total_queries": len(results),
        "mean_precision_at_k": round(sum(r["precision_at_k"] for r in results) / n, 4),
        "mean_recall_at_k": round(sum(r["recall_at_k"] for r in results) / n, 4),
        "mean_mrr": round(sum(r["mrr"] for r in results) / n, 4),
        "mean_keyword_coverage": round(sum(r["keyword_coverage"] for r in results) / n, 4),
        "guardrail_pass_rate": round(sum(1 for r in results if r["guardrail_passed"]) / n, 4),
        "evaluation_time_seconds": total_time,
        "results": results,
    }

    # Save results to disk
    try:
        results_dir = settings.evaluation_results_abs
        results_dir.mkdir(parents=True, exist_ok=True)
        out_file = results_dir / f"eval_run_{int(time.time())}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save evaluation results: {e}")

    return summary
