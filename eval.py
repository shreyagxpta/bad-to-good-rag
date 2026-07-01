"""Retrieval eval for RAG pipeline in good_rag.py.

Measures how well retrieval finds the right chunk for a set of labeled questions,
using two standard metrics:

  hit_rate@k : fraction of questions where a relevant chunk lands in the top k.
               Answers "did we retrieve the right thing at all?"
  MRR        : mean reciprocal rank of the first relevant chunk (1/rank, averaged).
               Answers "how highly did we rank it?" and rewards putting the right
               chunk first, which is exactly what reranking improves.

The golden set below is the ground truth: each question is paired with the chunk
id(s) that actually contain its answer. Only answerable questions appear here;
abstention on unanswerable questions is an answer-quality concern, measured later.

Usage:
    python eval.py
"""

import numpy as np

import good_rag as g   # reuse its chunk index, embedding, cosine, and reranker


# Golden set: question -> set of chunk ids that contain the answer. Any one of them
# counts as a relevant hit. Ids match good_rag's "path#i" chunk-source labels.
GOLDEN_SET = [
    {
        "question": "What kind of water should I use for brewing?",
        "relevant": {"docs/brewing_guide.md#3"},
    },
    {
        "question": "What grind size is best for espresso?",
        "relevant": {"docs/grind.md#0", "docs/brewing_guide.md#2"},
    },
    {
        "question": "How much caffeine is left in decaf coffee?",
        "relevant": {"docs/decaf.md#0"},
    },
    {
        "question": "Which roast keeps floral notes?",
        "relevant": {"docs/roast.md#0"},
    },
    {
        "question": "What coffee-to-water ratio and water temperature for a strong pour-over?",
        "relevant": {
            "docs/ratios.md#0", "docs/temperature.md#0",
            "docs/brewing_guide.md#0", "docs/brewing_guide.md#1",
        },
    },
]


def retrieve(question):
    """Return the ordered chunk source ids the good pipeline retrieves for a question.

    Mirrors good_rag's retrieval path (wide embedding retrieval, then rerank, then
    keep TOP_K) and exposes the ranked source ids the metrics need. If good_rag's
    retrieval logic changes, this would need to change with it; a cleaner design
    would factor a shared retrieve() out of good_rag.answer.

    Args:
        question: the question string.

    Returns:
        list of chunk source ids (e.g. "docs/brewing_guide.md#3"), best first.
    """
    q_vec = g.embed([question])[0]
    sims = g.cosine(q_vec, g.chunk_vecs)
    candidate_idx = np.argsort(sims)[::-1][:g.RETRIEVE_K]
    ranked = g.rerank(question, candidate_idx)[:g.TOP_K]
    return [g.chunk_sources[i] for i, _ in ranked]


def first_relevant_rank(retrieved, relevant):
    """Rank (1-based) of the first retrieved id that is relevant, or None if none are.

    Args:
        retrieved: ordered list of retrieved chunk ids, best first.
        relevant: set of chunk ids that count as correct.

    Returns:
        int rank starting at 1, or None if no relevant id was retrieved.
    """
    for rank, source in enumerate(retrieved, start=1):
        if source in relevant:
            return rank
    return None


def main():
    hits = 0
    reciprocal_ranks = []

    for item in GOLDEN_SET:
        retrieved = retrieve(item["question"])
        rank = first_relevant_rank(retrieved, item["relevant"])

        hits += rank is not None
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        status = f"hit @{rank}" if rank else "MISS"
        print(f"{status:8} {item['question']}")
        print(f"         retrieved: {retrieved}")

    n = len(GOLDEN_SET)
    print()
    print(f"hit_rate@{g.TOP_K}: {hits / n:.2f}  ({hits}/{n} questions retrieved a relevant chunk)")
    print(f"MRR:         {sum(reciprocal_ranks) / n:.2f}  (avg reciprocal rank of first relevant chunk)")


if __name__ == "__main__":
    main()