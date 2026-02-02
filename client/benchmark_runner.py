import os
import json
import time
import asyncio
import aiohttp
import numpy as np
from pathlib import Path

# =========================================================
# Configuration
# =========================================================

MODEL_NAME = os.environ["MODEL_NAME"]
TEST_TYPE = os.environ["TEST_TYPE"]
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
MODEL_MODE = os.environ.get("MODEL_MODE", "chat")  # chat | completion

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "./results"))
CONFIG_DIR = Path(os.environ.get("CONFIG_DIR", "./configs"))

CONCURRENCY = 4
TOTAL_REQUESTS = 20
MAX_TOKENS = 512
TEMPERATURE = 0.2

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# Workload generation
# =========================================================

def generate_prompts():
    prompts = []
    for i in range(TOTAL_REQUESTS):
        prompts.append(
            f"You are a clinical language model.\n\n"
            f"Create a structured clinical note from the following transcript.\n\n"
            f"Transcript:\n"
            f"Patient reports chest pain for the last {i+1} days, "
            f"worse on exertion, with mild shortness of breath.\n\n"
            f"Clinical Note:\n"
        )
    return prompts

# =========================================================
# Request helpers
# =========================================================

def build_request_payload(prompt: str):
    if MODEL_MODE == "chat":
        return {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a clinical language model."},
                {"role": "user", "content": prompt},
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "stream": False,
        }, "/v1/chat/completions"

    # completion mode
    return {
        "model": MODEL_NAME,
        "prompt": prompt,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }, "/v1/completions"

def extract_text(response_json):
    try:
        if MODEL_MODE == "chat":
            return response_json["choices"][0]["message"]["content"]
        return response_json["choices"][0]["text"]
    except Exception:
        return ""

def extract_usage(response_json):
    usage = response_json.get("usage", {})
    return (
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
    )

# =========================================================
# Single request
# =========================================================

async def run_single(session, prompt):
    payload, endpoint = build_request_payload(prompt)
    url = f"{VLLM_BASE_URL}{endpoint}"

    start = time.perf_counter()
    try:
        async with session.post(url, json=payload, timeout=120) as resp:
            data = await resp.json()
            latency = time.perf_counter() - start

            text = extract_text(data)
            tin, tout = extract_usage(data)

            return {
                "ok": True,
                "latency": latency,
                "tokens_in": tin,
                "tokens_out": tout,
                "prompt": prompt,
                "output": text,
            }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
        }

# =========================================================
# Benchmark runner
# =========================================================

async def run_benchmark():
    prompts = generate_prompts()

    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=None)

    results = []
    start_time = time.perf_counter()

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        sem = asyncio.Semaphore(CONCURRENCY)

        async def wrapped(prompt):
            async with sem:
                return await run_single(session, prompt)

        tasks = [wrapped(p) for p in prompts]
        for coro in asyncio.as_completed(tasks):
            results.append(await coro)

    total_time = time.perf_counter() - start_time

    ok_results = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]

    latencies = [r["latency"] for r in ok_results]
    tokens_in = sum(r["tokens_in"] for r in ok_results)
    tokens_out = sum(r["tokens_out"] for r in ok_results)

    report = {
        "model": MODEL_NAME,
        "model_mode": MODEL_MODE,
        "test_type": TEST_TYPE,
        "total_requests": TOTAL_REQUESTS,
        "successful": len(ok_results),
        "failed": len(failed),
        "total_duration_sec": total_time,
        "throughput_tokens_per_sec": (
            tokens_out / total_time if total_time > 0 else 0
        ),
        "avg_latency_sec": float(np.mean(latencies)) if latencies else 0,
        "p95_latency_ms": float(np.percentile(latencies, 95) * 1000)
        if len(latencies) >= 2 else 0,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_per_gpu_per_sec": (
            (tokens_out / total_time) / 8 if total_time > 0 else 0
        ),
        "samples": [
            {"prompt": r["prompt"], "output": r["output"]}
            for r in ok_results[:3]
        ],
        "errors": failed,
        "status": "OK" if not failed else "PARTIAL",
    }

    out_file = RESULTS_DIR / f"{TEST_TYPE}_{MODEL_NAME.replace('/', '_')}_{int(time.time())}.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f">>> Report saved: {out_file}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(run_benchmark())
