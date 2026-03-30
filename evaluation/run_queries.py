from __future__ import annotations

import argparse
import json
import time
import uuid

import requests

from test_set import TEST_SET
AGENT_A_URL = "http://localhost:8000"
DEFAULT_OUT = "eval/results.json"


def run_query(base_url: str, question: str, session_id: str) -> dict:
    payload = {
        "message": question,
        "session_id": session_id,
    }
    try:
        resp = requests.post(f"{base_url}/chat", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        return {"response": "", "route": "timeout_error", "session_id": session_id}
    except Exception as e:
        return {"response": "", "route": f"error: {str(e)}", "session_id": session_id}


def main(base_url: str, out_path: str) -> None:
    results = []
    print(f"Running {len(TEST_SET)} queries against {base_url}/chat ...\n")

    for item in TEST_SET:
        session_id = str(uuid.uuid4())
        print(f"[{item['id']:02d}/20] {item['question'][:70]}...")

        start = time.time()
        response = run_query(base_url, item["question"], session_id)
        elapsed = round(time.time() - start, 2)

        results.append({
            "id": item["id"],
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "source_type": item["source_type"],
            "expected_route": item["expected_route"],
            "actual_response": response.get("response", ""),
            "actual_route": response.get("route", ""),
            "latency_s": elapsed,
        })
        print(f"       route={response.get('route')} | latency={elapsed}s")
        time.sleep(0.5)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=AGENT_A_URL)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.url, args.out)