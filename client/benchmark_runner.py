#!/usr/bin/env python3

import os
import time
import json
import asyncio
import aiohttp
import statistics
from pathlib import Path

MODEL_NAME = os.environ["MODEL_NAME"]
TEST_TYPE = os.environ["TEST_TYPE"]          # diarization | clinical
BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "./results"))
CONFIG_DIR = Path(os.environ.get("CONFIG_DIR", "./configs"))

CONCURRENCY = int(os.environ.get("CONCURRENCY", 4))
REQUESTS = int(os.environ.get("REQUESTS", 20))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 512))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Workload generation
# ------------------------------------------------------------

def generate_clinical_prompt(i: int) -> str:
    return f"""
You are a clinical documentation assistant.

Patient information:
- Age: 67
- Sex: Male
- Chief complaint: Shortness of breath
- History: Hypertension, Type 2 Diabetes
- Vitals: BP 145/90, HR 102, SpO2 92%

Task:
Write a concise but complete clinical note including:
- Assessment
- Differential diagnosis
- Plan

Case ID: {i}
""".strip()


def generate_diarization_messages(i: int):
    return [
        {
            "role": "system",
            "content": "You are a medical transcription assistant."
        },
        {
            "role": "user",
            "content": f"""
Speaker 1: Hello, how are you feeling today?
Speaker 2: I have been feeling tired and short of breath.
Speaker 1: Any chest pain?
Speaker 2: No chest pain, just fatigue.

Task:
Summarize this conversation into a clean clinical transcription.

Conversation ID: {i}
""".strip()
        }
    ]


# ------------------------------------------------------------
# Request builders
# ------------------------------------------------------------

def build_request_payload(i: int):
    if TEST_TYPE == "clinical":
        return {
            "model": MODEL_NAME,
            "prompt": generate_clinical_prompt(i),
            "max_tokens": MAX_TOKENS,
            "temperature": 0.2,
        }
    else:
        return {
            "model": MODEL_NAME,
            "messages": generate_diarization_messages(i),
            "max_tokens": MAX_TOKENS,
            "temperature": 0.2,
        }


def get_endpoint():
    if TEST_TYPE == "clinical":
        return "/v1/completions"
    else:
        return "/v1/chat/completions"


# ------------------------------------------------------------
# Benchmark logic
# ------------------------------------------------------------

async def run_single_request(session, sem, i, timings, token_counts, errors):
    async with sem:
        payload = build_request_payload(i)
        endpoint = get_endpoint()
        url = BASE_URL + endpoint

        start = time.time()
        try:
            async with session.post(url, json=payload, timeout=300) as resp:
                if resp.status != 200:
                    txt = await resp.text()
                    errors.append(f"{resp.status}: {txt}")
                    return

                data = await resp.json()
                end = time.time()

                timings.append(end - start)

                if TEST_TYPE == "clinical":
                    token_counts.append(
                        data.get("usage", {}).get("completion_tokens", 0)
                    )
                else:
                    token_counts.append(
                        data.get("usage", {}).get("total_tokens", 0)
                    )

        except Exception as e:
            errors.append(str(e))


async def run_benchmark():
    sem = asyncio.Semaphore(CONCURRENCY)
    timings = []
    token_counts = []
    errors = []

    async with aiohttp.ClientSession() as session:
        tasks = [
            run_single_request(session, sem, i, timings, token_counts, errors)
            for i in range(REQUESTS)
        ]
        await asyncio.gather(*tasks)

    return timings, token_counts, errors


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

async def main():
    print(f">>> Generating Workload Data...")
    workload_path = CONFIG_DIR / "workloads.json"
    workload_path.write_text(
        json.dumps(
            {
                "test_type": TEST_TYPE,
                "model": MODEL_NAME,
                "requests": REQUESTS,
                "concurrency": CONCURRENCY,
            },
            indent=2,
        )
    )
    print(f">>> Synthetic Data Generated at {workload_path}")

    print(">>> Waiting for vLLM to initialize...")
    time.sleep(2)

    start_time = time.time()
    timings, token_counts, errors = await run_benchmark()
    total_time = time.time() - start_time

    successful = len(timings)
    failed = REQUESTS - successful

    result = {
        "model": MODEL_NAME,
        "test_type": TEST_TYPE,
        "total_requests": REQUESTS,
        "successful": successful,
        "failed": failed,
        "total_duration_sec": total_time,
        "throughput_tokens_per_sec": (
            sum(token_counts) / total_time if total_time > 0 else 0
        ),
        "avg_latency_sec": statistics.mean(timings) if timings else 0,
        "p95_latency_ms": (
            statistics.quantiles(timings, n=20)[18] * 1000
            if len(timings) >= 20 else 0
        ),
        "errors": [{"error": e} for e in errors[:5]],
    }

    out_file = RESULTS_DIR / f"{TEST_TYPE}_{MODEL_NAME.replace('/', '_')}_{int(time.time())}.json"
    out_file.write_text(json.dumps(result, indent=2))
    print(f">>> Report saved: {out_file}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
