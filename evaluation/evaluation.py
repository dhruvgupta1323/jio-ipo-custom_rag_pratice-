import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.evaluation import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    keyword_coverage,
    load_dataset,
    run_evaluation_suite,
)
from backend.app.services.retrieval import retriever
from backend.app.services.generation import generator


def main():
    print("=" * 60)
    print("Running SecureRAG Evaluation on Jio Platforms Limited IPO")
    print("=" * 60)

    summary = run_evaluation_suite(retriever, generator)

    print(f"\nTotal Benchmark Queries: {summary['total_queries']}")
    print(f"Mean Precision@3:       {summary['mean_precision_at_k'] * 100:.2f}%")
    print(f"Mean Recall@3:          {summary['mean_recall_at_k'] * 100:.2f}%")
    print(f"Mean MRR:               {summary['mean_mrr']:.4f}")
    print(f"Mean Keyword Coverage:  {summary['mean_keyword_coverage'] * 100:.2f}%")
    print(f"Guardrail Pass Rate:    {summary['guardrail_pass_rate'] * 100:.2f}%")
    print(f"Evaluation Run Time:    {summary['evaluation_time_seconds']}s")
    print("\n" + "=" * 60)

    for res in summary["results"]:
        status = "PASS" if res["guardrail_passed"] else "FLAGGED"
        print(f"[{res['id']}] [{status}] {res['question']}")
        print(f"  P@3: {res['precision_at_k']} | R@3: {res['recall_at_k']} | MRR: {res['mrr']} | Keywords: {res['keyword_coverage'] * 100:.0f}%")
        print(f"  Pages Retrieved: {res['retrieved_pages']} | Relevant: {res['relevant_pages']}")
        print(f"  Answer Snippet: {res['generated_answer'][:120]}...\n")


if __name__ == "__main__":
    main()
