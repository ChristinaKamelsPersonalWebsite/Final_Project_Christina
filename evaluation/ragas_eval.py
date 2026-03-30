from __future__ import annotations

import argparse
import json
import re
import time
import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:1.5b"


JUDGE_SYSTEM_PROMPT = (
    "You are a strict but fair evaluator of AI fitness coaching responses. "
    "Score only based on the provided criteria. "
    "Respond ONLY with valid JSON and no extra text."
)

JUDGE_PROMPT_TEMPLATE = """Evaluate this AI fitness coach response.

Question: {question}
Ground Truth Answer: {ground_truth}
AI Response: {response}

Score each dimension from 0 to 5:
- faithfulness: Does the response avoid hallucinating facts not in the ground truth? (5 = fully grounded, 0 = completely fabricated)
- correctness: Is the information factually correct compared to the ground truth? (5 = fully correct, 0 = wrong)
- relevance: Does the response actually answer the question asked? (5 = directly answers it, 0 = off-topic)

Reply with ONLY this JSON (no markdown, no explanation):
{{"faithfulness": <0-5>, "correctness": <0-5>, "relevance": <0-5>, "reasoning": "<one sentence>"}}"""


def _call_ollama(prompt: str, retries: int = 3, timeout: int = 180) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_ctx": 2048,
        },
    }

    last_error = None

    for attempt in range(retries):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"].strip()
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(3)

    raise RuntimeError(f"Ollama call failed after {retries} retries: {last_error}")


def judge_response(question: str, ground_truth: str, response: str) -> dict:
    if not response or len(response.strip()) < 5:
        return {
            "faithfulness": 0,
            "correctness": 0,
            "relevance": 0,
            "reasoning": "Empty response.",
        }

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        response=response[:1200],
    )

    raw = ""
    try:
        raw = _call_ollama(prompt)
        clean = re.sub(r"```json|```", "", raw).strip()
        return json.loads(clean)

    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass

        return {
            "faithfulness": -1,
            "correctness": -1,
            "relevance": -1,
            "reasoning": f"Parse error: {raw[:150]}",
        }

    except Exception as e:
        return {
            "faithfulness": -1,
            "correctness": -1,
            "relevance": -1,
            "reasoning": str(e),
        }


def main(results_path: str, out_path: str) -> None:
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    print(f"Evaluating {len(results)} responses with LLM-as-judge...\n")

    scored = []
    faithfulness_scores = []
    correctness_scores = []
    relevance_scores = []

    total = len(results)

    for i, item in enumerate(results, 1):
        print(f"[{i:02d}/{total:02d}] Judging: {item['question'][:60]}...")

        scores = judge_response(
            question=item["question"],
            ground_truth=item["ground_truth"],
            response=item["actual_response"],
        )

        row = {**item, "scores": scores}
        scored.append(row)

        if scores.get("faithfulness", -1) >= 0:
            faithfulness_scores.append(scores["faithfulness"])
            correctness_scores.append(scores["correctness"])
            relevance_scores.append(scores["relevance"])

        print(
            f"       F={scores.get('faithfulness')} "
            f"C={scores.get('correctness')} "
            f"R={scores.get('relevance')} | "
            f"{scores.get('reasoning', '')[:80]}"
        )

        time.sleep(2)

    n = len(faithfulness_scores)
    summary = {
        "questions_evaluated": len(results),
        "questions_scored": n,
        "avg_faithfulness": round(sum(faithfulness_scores) / n, 3) if n else 0,
        "avg_correctness": round(sum(correctness_scores) / n, 3) if n else 0,
        "avg_relevance": round(sum(relevance_scores) / n, 3) if n else 0,
        "avg_overall": round(
            (
                sum(faithfulness_scores)
                + sum(correctness_scores)
                + sum(relevance_scores)
            ) / (3 * n),
            3,
        ) if n else 0,
    }

    output = {
        "summary": summary,
        "per_question": scored,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n=== GENERATION EVALUATION SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results.json")
    parser.add_argument("--out", default="generation_scores.json")
    args = parser.parse_args()
    main(args.results, args.out)