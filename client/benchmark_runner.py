import os
import json
import time
import math
import asyncio
import aiohttp
from collections import Counter
from typing import Dict, Any

from generate_data import generate_configs


# =========================================================
# Environment
# =========================================================
MODEL_NAME = os.environ["MODEL_NAME"]
MODEL_MODE = os.environ.get("MODEL_MODE", "chat")  # chat | completion
TEST_TYPE = os.environ["TEST_TYPE"]               # diarization | clinical
BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "./results")

API_URL = f"{BASE_URL}/v1/chat/completions" if MODEL_MODE == "chat" else f"{BASE_URL}/v1/completions"


# =========================================================
# Load workload config (SINGLE SOURCE OF TRUTH)
# =========================================================
WORKLOADS = generate_configs()
CONFIG = WORKLOADS[TEST_TYPE]

SYSTEM_PROMPT = CONFIG["system_prompt"]
USER_PROMPT = CONFIG["user_prompt"]
MAX_TOKENS = CONFIG["max_tokens"]
TEMPERATURE = CONFIG["temperature"]
CONCURRENCY = CONFIG["concurrency"]
REQUESTS_COUNT = CONFIG["requests_count"]


# =========================================================
# Helpers
# =========================================================
def entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values())


def normalize_label(text: str) -> str:
    return text.strip().split()[0].replace(".", "").replace(",", "")


# =========================================================
# Request Builder
# =========================================================
def build_payload() -> Dict[str, Any]:
    if MODEL_MODE == "chat":
        return {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        }
    else:
        return {
            "model": MODEL_NAME,
            "prompt": SYSTEM_PROMPT + "\n\n" + USER_PROMPT,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        }


# =========================================================
# Async worker
# =========================================================
async def run_request(session: aiohttp.ClientSession, sem: asyncio.Semaphore, stats: dict):
    async with sem:
        start = time.perf_counter()
        try:
            async with session.post(API_URL, json=build_payload(), timeout=300) as resp:
                data = await resp.json()
                latency = time.perf_counter() - start

                if MODEL_MODE == "chat":
                    output = data["choices"][0]["message"]["content"]
                else:
                    output = data["choices"][0]["text"]

                stats["samples"].append({
                    "prompt": USER_PROMPT,
                    "output": output
                })

                stats["latencies"].append(latency)
                stats["successful"] += 1
                stats["tokens_in"] += data.get("usage", {}).get("prompt_tokens", 0)
                stats["tokens_out"] += data.get("usage", {}).get("completion_tokens", 0)

        except Exception as e:
            stats["failed"] += 1
            stats["errors"].append(str(e))


# =========================================================
# Main runner
# =========================================================
async def main():
    sem = asyncio.Semaphore(CONCURRENCY)

    stats = {
        "model": MODEL_NAME,
        "model_mode": MODEL_MODE,
        "test_type": TEST_TYPE,
        "total_requests": REQUESTS_COUNT,
        "successful": 0,
        "failed": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "latencies": [],
        "samples": [],
        "errors": [],
    }

    start_time = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        tasks = [
            run_request(session, sem, stats)
            for _ in range(REQUESTS_COUNT)
        ]
        await asyncio.gather(*tasks)

    total_duration = time.perf_counter() - start_time

    # =====================================================
    # Metrics
    # =====================================================
    latencies = stats["latencies"]
    latencies_sorted = sorted(latencies)

    result = {
        "model": MODEL_NAME,
        "model_mode": MODEL_MODE,
        "test_type": TEST_TYPE,
        "total_requests": REQUESTS_COUNT,
        "successful": stats["successful"],
        "failed": stats["failed"],
        "total_duration_sec": total_duration,
        "avg_latency_sec": sum(latencies) / len(latencies) if latencies else 0.0,
        "p95_latency_ms": latencies_sorted[int(0.95 * len(latencies))] * 1000 if latencies else 0.0,
        "tokens_in": stats["tokens_in"],
        "tokens_out": stats["tokens_out"],
        "throughput_tokens_per_sec": stats["tokens_out"] / total_duration if total_duration > 0 else 0.0,
        "tokens_per_gpu_per_sec": stats["tokens_out"] / total_duration if total_duration > 0 else 0.0,
        "samples": stats["samples"],
        "errors": stats["errors"],
        "status": "OK" if stats["failed"] == 0 else "PARTIAL",
    }

    # =====================================================
    # Diarization-specific metrics
    # =====================================================
    if TEST_TYPE == "diarization":
        labels = [normalize_label(s["output"]) for s in stats["samples"]]
        label_counts = Counter(labels)

        result["diarization"] = {
            "label_distribution": dict(label_counts),
            "entropy": entropy(label_counts),
            "consistency": max(label_counts.values()) / sum(label_counts.values()) if label_counts else 0.0
        }

    # =====================================================
    # Save report
    # =====================================================
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = int(time.time())
    out_file = f"{RESULTS_DIR}/{TEST_TYPE}_{MODEL_NAME.replace('/', '_')}_{ts}.json"

    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f">>> Report saved: {out_file}")
    print(json.dumps(result, indent=2))


# =========================================================
# Entrypoint
# =========================================================
if __name__ == "__main__":
    asyncio.run(main())
