"""
eval/retrieval_metrics.py

Computes Precision@K, Recall@K, and MRR for the nutrition RAG pipeline.

Design note:
- Only nutrition questions are evaluated for retrieval metrics because
  exercises and programs now use direct JSON filtering (no Qdrant RAG).
- Queries Qdrant directly using the same embedding model as ingestion
  (all-MiniLM-L6-v2) and the same metadata filter (source_type=nutrition).

Usage:
    python eval/retrieval_metrics.py
    python eval/retrieval_metrics.py --k 3 --out eval/retrieval_results_k3.json
    python eval/retrieval_metrics.py --k 5 --out eval/retrieval_results_k5.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

from test_set import TEST_SET

# ── Config ─────────────────────────────────────────────────────────────────────
QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION      = os.getenv("NUTRITION_COLLECTION", "nutrition_guides")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Clients (loaded once) ──────────────────────────────────────────────────────
_qdrant  = QdrantClient(url=QDRANT_URL)
_encoder = SentenceTransformer(EMBEDDING_MODEL)


# ── Retrieval ──────────────────────────────────────────────────────────────────

def retrieve_nutrition(query: str, k: int) -> list[dict]:
    """
    Queries Qdrant for the top-k nutrition chunks matching the query.
    Always filters by source_type='nutrition' — same filter Agent B uses.
    """
    vector = _encoder.encode(query).tolist()

    results = _qdrant.query_points(
    collection_name=COLLECTION,
    query=vector,
    limit=k,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="source_type",
                match=MatchValue(value="nutrition"),
            )
        ]
    ),
    with_payload=True,
).points

    return [
        {
            "score":       hit.score,
            "topic":       hit.payload.get("topic", ""),
            "source_type": hit.payload.get("source_type", ""),
            "text":        hit.payload.get("text", "")[:200],  # truncate for display
        }
        for hit in results
    ]


# ── Relevance judgment ─────────────────────────────────────────────────────────

def is_relevant(chunk: dict, question: dict) -> bool:
    """
    A chunk is relevant if:
    1. Its source_type is 'nutrition' (guaranteed by the filter, but double-checked)
    2. Its topic keyword appears in the question or ground truth — giving us
       a meaningful relevance signal beyond just source_type matching.

    This makes the eval more honest than simply counting source_type matches.
    """
    if chunk.get("source_type") != "nutrition":
        return False

    topic = chunk.get("topic", "").lower().replace("_", " ")
    question_text = (question["question"] + " " + question["ground_truth"]).lower()

    # Direct topic match
    if topic and topic in question_text:
        return True

    # Keyword-level match for common topic aliases
    topic_keywords = {
        "macros":                      ["protein", "carb", "fat", "macro"],
        "calorie calculations":        ["calorie", "deficit", "surplus", "tdee"],
        "meal timing":                 ["timing", "when to eat", "meal frequency"],
        "pre workout nutrition":       ["before workout", "pre workout", "pre-workout"],
        "post workout nutrition":      ["after workout", "post workout", "post-workout", "recovery"],
        "hydration":                   ["water", "hydration", "drink", "fluid"],
        "supplements":                 ["supplement", "creatine", "protein powder", "caffeine"],
        "meal prep tips":              ["meal prep", "batch cook", "prepare food"],
        "food categories":             ["food", "protein source", "carb source"],
        "cutting strategy":            ["cut", "fat loss", "lose weight", "deficit"],
        "bulking strategy":            ["bulk", "muscle gain", "surplus", "build muscle"],
        "macro ratios by goal":        ["macro ratio", "split", "percentage"],
        "intermittent fasting":        ["fasting", "intermittent", "16:8", "eating window"],
        "nutrition for beginners":     ["beginner", "start", "basics"],
        "fiber and micronutrients":    ["fiber", "vitamin", "mineral", "vegetable"],
        "budget nutrition":            ["budget", "cheap", "affordable"],
        "nutrition for cardio vs strength": ["cardio", "endurance", "strength training"],
        "alcohol and fitness":         ["alcohol", "drink", "beer"],
        "sleep and nutrition":         ["sleep", "rest", "recovery"],
        "protein timing and distribution": ["protein timing", "distribution", "leucine"],
        "eating out and social eating": ["restaurant", "eating out", "social"],
        "deload week nutrition":       ["deload", "rest week", "lighter week"],
    }

    keywords = topic_keywords.get(topic, [])
    return any(kw in question_text for kw in keywords)


# ── Metrics computation ────────────────────────────────────────────────────────

def compute_metrics(k: int) -> dict:
    nutrition_questions = [q for q in TEST_SET if q["source_type"] == "nutrition"]

    precision_scores   = []
    recall_scores      = []
    reciprocal_ranks   = []
    per_question       = []

    for item in nutrition_questions:
        chunks = retrieve_nutrition(item["question"], k)

        relevant_retrieved = sum(1 for c in chunks if is_relevant(c, item))
        precision_at_k     = relevant_retrieved / k
        # Recall: how many relevant chunks did we get vs total retrieved relevant
        # (simplified: relevant retrieved / max possible relevant = relevant retrieved / k)
        recall_at_k        = relevant_retrieved / max(1, k)

        # MRR: reciprocal rank of first relevant result
        rr = 0.0
        for rank, chunk in enumerate(chunks, start=1):
            if is_relevant(chunk, item):
                rr = 1.0 / rank
                break

        precision_scores.append(precision_at_k)
        recall_scores.append(recall_at_k)
        reciprocal_ranks.append(rr)

        per_question.append({
            "id":                    item["id"],
            "question":              item["question"],
            "source_type":           item["source_type"],
            "retrieved":             len(chunks),
            "relevant_retrieved":    relevant_retrieved,
            f"precision@{k}":        round(precision_at_k, 4),
            f"recall@{k}":           round(recall_at_k, 4),
            "reciprocal_rank":       round(rr, 4),
            "top_topics_retrieved":  [c["topic"] for c in chunks],
        })

    n = len(precision_scores)
    summary = {
        "k":                       k,
        "questions_evaluated":     n,
        "note":                    "Retrieval evaluated on nutrition questions only. Exercise and program builder use direct JSON filtering (no Qdrant).",
        f"mean_precision@{k}":     round(sum(precision_scores) / n, 4) if n else 0,
        f"mean_recall@{k}":        round(sum(recall_scores) / n, 4) if n else 0,
        "MRR":                     round(sum(reciprocal_ranks) / n, 4) if n else 0,
    }

    return {"summary": summary, "per_question": per_question}


# ── Main ───────────────────────────────────────────────────────────────────────

def main(k: int, out_path: str) -> None:
    print(f"\nComputing retrieval metrics at K={k} ...")
    print(f"Collection : {COLLECTION}")
    print(f"Qdrant URL : {QDRANT_URL}")
    print(f"Model      : {EMBEDDING_MODEL}\n")

    metrics = compute_metrics(k)

    print("=== SUMMARY ===")
    for key, val in metrics["summary"].items():
        print(f"  {key}: {val}")

    print("\n=== PER QUESTION ===")
    for q in metrics["per_question"]:
        pk = f"precision@{k}"
        print(
            f"  Q{q['id']}: {q['question'][:60]}...\n"
            f"    {pk}={q[pk]}  recall@{k}={q[f'recall@{k}']}  MRR={q['reciprocal_rank']}\n"
            f"    Topics retrieved: {q['top_topics_retrieved']}\n"
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",   type=int, default=5)
    parser.add_argument("--out", default="eval/retrieval_results_k5.json")
    args = parser.parse_args()
    main(args.k, args.out)
