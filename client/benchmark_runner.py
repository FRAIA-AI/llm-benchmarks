import asyncio
import aiohttp
import time
import json
import os
import sys
import numpy as np
from uuid import uuid4
from datetime import datetime

# =========================================================
# Environment
# =========================================================

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
VLLM_URL = f"{VLLM_BASE_URL}/v1/chat/completions"
HEALTH_URL = f"{VLLM_BASE_URL}/health"

MODEL_NAME = os.getenv("MODEL_NAME")
TEST_TYPE = os.getenv("TEST_TYPE", "diarization")

RESULTS_DIR = os.getenv("RESULTS_DIR", "./results")
CONFIG_DIR = os.getenv("CONFIG_DIR", "./configs")

WORKLOAD_FILE = os.path.join(CONFIG_DIR, "workloads.json")

GPU_COUNT = int(os.getenv("RUNPOD_GPU_COUNT", "1"))

os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================================================
# Streaming Request Handler
# =========================================================

async def make_request(session, payload, concurrency_level):
    request_id = str(uuid4())

    start_ts = time.time()
    first_token_ts = None
    last_token_ts = None

    full_text = ""
    usage = None
    inter_token_latencies = []

    payload = payload.copy()
    payload["stream"] = True

    try:
        async with session.post(VLLM_URL, json=payload) as response:
            if response.status != 200:
                return {
                    "request_id": request_id,
                    "error": await response.text()
                }

            async for line in response.content:
                now = time.time()
                line = line.decode("utf-8").strip()

                if not line or line == "data: [DONE]":
                    continue

                if not line.startswith("data: "):
                    continue

                data = json.loads(line[6:])

                # Final usage block
                if "usage" in data:
                    usage = data["usage"]

                delta = data.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")

                if content:
                    if first_token_ts is None:
                        first_token_ts = now
                    else:
                        inter_token_latencies.append(now - last_token_ts)

                    last_token_ts = now
                    full_text += content

        end_ts = time.time()

        if not usage or not first_token_ts:
            return {
                "request_id": request_id,
                "error": "Missing usage or tokens"
            }

        decode_time = last_token_ts - first_token_ts

        record = {
            "request_id": request_id,
            "model": MODEL_NAME,
            "test_type": TEST_TYPE,
            "concurrency": concurrency_level,

            "timing": {
                "request_start_ts": start_ts,
                "first_token_ts": first_token_ts,
                "last_token_ts": last_token_ts,
                "total_latency_ms": (end_ts - start_ts) * 1000,
                "ttft_ms": (first_token_ts - start_ts) * 1000,
                "decode_time_ms": decode_time * 1000,
                "tpot_ms": (decode_time / usage["completion_tokens"]) * 1000
            },

            "tokens": {
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
                "tokens_per_second": usage["completion_tokens"] / decode_time
            },

            "output": {
                "raw_text": full_text,
                "truncated": False
            },

            "runtime": {
                "gpu_count": GPU_COUNT,
                "tokens_per_gpu_per_sec":
                    (usage["completion_tokens"] / decode_time) / GPU_COUNT
            }
        }

        # Attempt JSON parse (important for clinical)
        try:
            parsed = json.loads(full_text)
            record["output"]["parsed_json"] = parsed
            record["output"]["valid_json"] = True
        except Exception:
            record["output"]["valid_json"] = False

        return record

    except Exception as e:
        return {
            "request_id": request_id,
            "error": str(e)
        }

# =========================================================
# Benchmark Runner
# =========================================================

async def run_benchmark():
    print(">>> Generating Workload Data...")
    if os.system("python client/generate_data.py") != 0:
        print("!!! Data generation failed")
        sys.exit(1)

    with open(WORKLOAD_FILE) as f:
        config = json.load(f)[TEST_TYPE]

    concurrency = config["concurrency"]
    requests_count = config["requests_count"]

    print("=================================================")
    print(f" STARTING BENCHMARK: {TEST_TYPE.upper()}")
    print(f" Model: {MODEL_NAME}")
    print(f" Target: {VLLM_BASE_URL}")
    print(f" Concurrency: {concurrency}")
    print(f" Requests: {requests_count}")
    print(f" GPUs: {GPU_COUNT}")
    print("=================================================")

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": config["user_prompt"]}
        ],
        "max_tokens": config["max_tokens"],
        "temperature": config["temperature"]
    }

    async with aiohttp.ClientSession() as session:
        print(">>> Waiting for vLLM to initialize...")
        start_wait = time.time()

        while True:
            try:
                async with session.get(HEALTH_URL) as r:
                    if r.status == 200:
                        break
            except:
                pass

            if time.time() - start_wait > 900:
                print("!!! vLLM failed to become healthy")
                sys.exit(1)

            await asyncio.sleep(5)

        print(f">>> vLLM Ready (Waited {int(time.time() - start_wait)}s)")
        print(">>> Starting stress test...")

        start_bench = time.time()

        sem = asyncio.Semaphore(concurrency)

        async def bounded():
            async with sem:
                return await make_request(session, payload, concurrency)

        tasks = [bounded() for _ in range(requests_count)]
        results = await asyncio.gather(*tasks)

        total_duration = time.time() - start_bench

    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    total_completion_tokens = sum(r["tokens"]["completion_tokens"] for r in valid)
    total_prompt_tokens = sum(r["tokens"]["prompt_tokens"] for r in valid)

    # =====================================================
    # Aggregates
    # =====================================================

    summary = {
        "meta": {
            "model": MODEL_NAME,
            "test_type": TEST_TYPE,
            "timestamp": datetime.now().isoformat(),
            "concurrency": concurrency,
            "requests": requests_count,
            "gpus": GPU_COUNT
        },
        "results": {
            "successful": len(valid),
            "failed": len(errors),
            "total_duration_sec": total_duration,

            "throughput": {
                "tokens_per_sec": total_completion_tokens / total_duration,
                "tokens_per_sec_per_gpu":
                    (total_completion_tokens / total_duration) / GPU_COUNT,
                "tokens_per_hour":
                    (total_completion_tokens / total_duration) * 3600
            },

            "latency": {
                "avg_latency_ms":
                    np.mean([r["timing"]["total_latency_ms"] for r in valid]),
                "p95_latency_ms":
                    np.percentile(
                        [r["timing"]["total_latency_ms"] for r in valid], 95
                    ),
                "avg_ttft_ms":
                    np.mean([r["timing"]["ttft_ms"] for r in valid]),
                "avg_tpot_ms":
                    np.mean([r["timing"]["tpot_ms"] for r in valid])
            },

            "tokens": {
                "avg_prompt_tokens":
                    total_prompt_tokens / len(valid),
                "avg_completion_tokens":
                    total_completion_tokens / len(valid),
                "avg_total_tokens":
                    (total_prompt_tokens + total_completion_tokens) / len(valid)
            },

            "quality": {
                "valid_json_ratio":
                    sum(r["output"].get("valid_json", False) for r in valid) / len(valid)
            },

            "errors": errors[:5]
        }
    }

    safe_model = MODEL_NAME.replace("/", "_")
    raw_path = os.path.join(
        RESULTS_DIR, f"{TEST_TYPE}_{safe_model}_raw.jsonl"
    )
    summary_path = os.path.join(
        RESULTS_DIR, f"{TEST_TYPE}_{safe_model}_{int(time.time())}.json"
    )

    with open(raw_path, "w") as f:
        for r in valid:
            f.write(json.dumps(r) + "\n")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f">>> Raw results: {raw_path}")
    print(f">>> Summary: {summary_path}")
    print(json.dumps(summary["results"], indent=2))

# =========================================================

if __name__ == "__main__":
    asyncio.run(run_benchmark())
