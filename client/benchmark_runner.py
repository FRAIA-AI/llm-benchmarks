#!/usr/bin/env python3

import asyncio
import aiohttp
import json
import os
import time
import uuid
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------
MODEL_NAME = os.environ["MODEL_NAME"]
TEST_TYPE = os.environ["TEST_TYPE"]
BASE_URL = os.environ["VLLM_BASE_URL"]
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "./results"))
CONFIG_DIR = Path(os.environ.get("CONFIG_DIR", "./configs"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------
CONCURRENCY = int(os.environ.get("CONCURRENCY", 4))
REQUESTS = int(os.environ.get("REQUESTS", 20))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 512))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.2))
GPUS = int(os.environ.get("GPUS", 8))

# ---------------------------------------------------------
# Prompt builders (CRITICAL)
# ---------------------------------------------------------
def build_prompt(test_type: str, sample: str) -> str:
    if test_type == "clinical":
        return (
            "You are a clinical language model.\n\n"
            "Create a structured clinical note from the following transcript.\n\n"
            "Transcript:\n"
            f"{sample}\n\n"
            "Clinical Note:\n"
        )
    elif test_type == "diarization":
        return (
            "Identify speakers and rewrite the following transcript "
            "with speaker labels.\n\n"
            f"{sample}\n\n"
            "Diarized Transcript:\n"
        )
    else:
        return sample


# ---------------------------------------------------------
# Workload generation
# ---------------------------------------------------------
def generate_workload() -> List[str]:
    samples = []
    for i in range(REQUESTS):
        samples.append(
            f"Patient reports chest pain for the last {i+1} days, "
            f"worse on exertion, with mild shortness of breath."
        )
    return samples


# ---------------------------------------------------------
# Single request
# ---------------------------------------------------------
async def run_single(
    session: aiohttp.ClientSession,
    prompt: str,
) -> Dict[str, Any]:

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }

    start = time.perf_counter()

    try:
        async with session.post(
            f"{BASE_URL}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:

            ttft = time.perf_counter() - start
            data = await resp.json()
            end = time.perf_counter()

            if "choices" not in data or not data["choices"]:
                return {"ok": False, "error": "No choices returned"}

            text = data["choices"][0].get("text", "")
            usage = data.get("usage", {})

            return {
                "ok": True,
                "latency": end - start,
                "ttft": ttft,
                "tpot": (end - ttft) / max(len(text.split()), 1),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "text": text,
            }

    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------
async def run_benchmark():

    workload = generate_workload()
    prompts = [build_prompt(TEST_TYPE, s) for s in workload]

    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=None)

    results = []
    start_time = time.perf_counter()

    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout
    ) as session:

        sem = asyncio.Semaphore(CONCURRENCY)

        async def bounded(prompt):
            async with sem:
                return await run_single(session, prompt)

        tasks = [bounded(p) for p in prompts]
        results = await asyncio.gather(*tasks)

    total_time = time.perf_counter() - start_time

    successes = [r for r in results if r.get("ok")]
    failures = [r for r in results if not r.get("ok")]

    latencies = [r["latency"] for r in successes]
    ttft = [r["ttft"] for r in successes]
    tpot = [r["tpot"] for r in successes]

    prompt_tokens = sum(r["prompt_tokens"] for r in successes)
    completion_tokens = sum(r["completion_tokens"] for r in successes)
    total_tokens = prompt_tokens + completion_tokens

    report = {
        "model": MODEL_NAME,
        "test_type": TEST_TYPE,
        "total_requests": REQUESTS,
        "successful": len(successes),
        "failed": len(failures),
        "total_duration_sec": total_time,
        "throughput_tokens_per_sec": (
            total_tokens / total_time if total_tokens > 0 else 0.0
        ),
        "avg_latency_sec": float(np.mean(latencies)) if latencies else 0.0,
        "p95_latency_ms": float(np.percentile(latencies, 95) * 1000) if latencies else 0.0,
        "avg_ttft_ms": float(np.mean(ttft) * 1000) if ttft else 0.0,
        "avg_tpot_ms": float(np.mean(tpot) * 1000) if tpot else 0.0,
        "tokens_in": prompt_tokens,
        "tokens_out": completion_tokens,
        "tokens_per_gpu_per_sec": (
            total_tokens / total_time / GPUS if total_tokens > 0 else 0.0
        ),
        "samples": [
            {
                "prompt": prompts[i],
                "output": successes[i]["text"] if i < len(successes) else None,
            }
            for i in range(min(len(successes), 3))
        ],
        "errors": failures[:5],
        "status": "OK" if successes else "NO_SUCCESS",
    }

    out_file = RESULTS_DIR / f"{TEST_TYPE}_{MODEL_NAME.replace('/', '_')}_{int(time.time())}.json"
    out_file.write_text(json.dumps(report, indent=2))

    print(f">>> Report saved: {out_file}")
    print(json.dumps(report, indent=2))


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(run_benchmark())
