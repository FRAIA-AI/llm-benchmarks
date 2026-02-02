import os
import json
import time
import asyncio
import aiohttp
import numpy as np
from collections import Counter
from typing import List, Dict

# =========================================================
# Environment
# =========================================================

MODEL_NAME = os.environ["MODEL_NAME"]
MODEL_MODE = os.environ.get("MODEL_MODE", "chat")  # chat | completion
TEST_TYPE = os.environ["TEST_TYPE"]               # diarization | clinical
VLLM_BASE_URL = os.environ["VLLM_BASE_URL"]
RESULTS_DIR = os.environ["RESULTS_DIR"]
CONFIG_DIR = os.environ["CONFIG_DIR"]

CONCURRENCY = 4
REQUESTS = 20

# Diarization labels (fixed universe)
DIARIZATION_LABELS = ["Patient", "Doctor"]

# =========================================================
# Prompt builders
# =========================================================

def build_diarization_prompt(text: str) -> str:
    return f"""Task: Speaker identification.

Identify who is speaking in the transcript below.
Respond with ONLY ONE label.

Possible speakers:
- Patient
- Doctor

Transcript:
{text}

Speaker:
""".strip()


def build_clinical_prompt(text: str) -> str:
    return f"""You are a clinical language model.

Create a structured clinical note from the following transcript.

Transcript:
{text}

Clinical Note:
""".strip()


# =========================================================
# Request payloads
# =========================================================

def build_payload(prompt: str) -> Dict:
    if MODEL_MODE == "chat":
        return {
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0 if TEST_TYPE == "diarization" else 0.7,
            "max_tokens": 8 if TEST_TYPE == "diarization" else 1024,
        }
    else:
        return {
            "model": MODEL_NAME,
            "prompt": prompt,
            "temperature": 0.0 if TEST_TYPE == "diarization" else 0.7,
            "max_tokens": 8 if TEST_TYPE == "diarization" else 1024,
        }


# =========================================================
# Sample generation
# =========================================================

def generate_samples() -> List[str]:
    if TEST_TYPE == "diarization":
        return [
            "Yes doctor, the pain started yesterday and worsens when I walk.",
            "Can you describe where exactly the pain is located?",
            "It feels like pressure in the middle of my chest.",
            "Do you feel short of breath when climbing stairs?",
            "Yes, slightly."
        ] * (REQUESTS // 5)
    else:
        return [
            f"Patient reports chest pain for the last {d} days, worse on exertion, with mild shortness of breath."
            for d in range(1, REQUESTS + 1)
        ]


# =========================================================
# HTTP call
# =========================================================

async def call_model(session, payload):
    url = f"{VLLM_BASE_URL}/v1/chat/completions" if MODEL_MODE == "chat" else f"{VLLM_BASE_URL}/v1/completions"
    start = time.time()
    async with session.post(url, json=payload) as resp:
        data = await resp.json()
    latency = time.time() - start
    return data, latency


# =========================================================
# Metrics
# =========================================================

def entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = [v / total for v in counter.values()]
    return -sum(p * np.log2(p) for p in probs)


# =========================================================
# Benchmark runner
# =========================================================

async def run_benchmark():
    samples = generate_samples()

    latencies = []
    tokens_in = 0
    tokens_out = 0
    outputs = []
    diarization_labels = []

    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(CONCURRENCY)

        async def run_one(sample):
            nonlocal tokens_in, tokens_out

            if TEST_TYPE == "diarization":
                prompt = build_diarization_prompt(sample)
            else:
                prompt = build_clinical_prompt(sample)

            payload = build_payload(prompt)

            async with sem:
                result, latency = await call_model(session, payload)

            latencies.append(latency)

            if MODEL_MODE == "chat":
                content = result["choices"][0]["message"]["content"]
            else:
                content = result["choices"][0]["text"]

            outputs.append({
                "prompt": prompt,
                "output": content
            })

            tokens_in += len(prompt.split())
            tokens_out += len(content.split())

            if TEST_TYPE == "diarization":
                label = content.strip().split()[0]
                diarization_labels.append(label)

        start = time.time()
        await asyncio.gather(*(run_one(s) for s in samples))
        total_duration = time.time() - start

    # =====================================================
    # Aggregate metrics
    # =====================================================

    report = {
        "model": MODEL_NAME,
        "model_mode": MODEL_MODE,
        "test_type": TEST_TYPE,
        "total_requests": REQUESTS,
        "successful": REQUESTS,
        "failed": 0,
        "total_duration_sec": total_duration,
        "avg_latency_sec": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95) * 1000),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "throughput_tokens_per_sec": tokens_out / total_duration if total_duration > 0 else 0,
        "tokens_per_gpu_per_sec": (tokens_out / total_duration) / int(os.environ.get("TP_SIZE", 1)),
        "samples": outputs[:3],
        "errors": [],
        "status": "OK"
    }

    # =====================================================
    # Diarization-specific metrics
    # =====================================================

    if TEST_TYPE == "diarization":
        counter = Counter(diarization_labels)
        report["diarization"] = {
            "label_distribution": dict(counter),
            "entropy": entropy(counter),
            "consistency": max(counter.values()) / sum(counter.values()) if counter else 0.0
        }

    # =====================================================
    # Save
    # =====================================================

    ts = int(time.time())
    fname = f"{RESULTS_DIR}/{TEST_TYPE}_{MODEL_NAME.replace('/', '_')}_{ts}.json"
    with open(fname, "w") as f:
        json.dump(report, f, indent=2)

    print(f">>> Report saved: {fname}")
    print(json.dumps(report, indent=2))


# =========================================================
# Entry
# =========================================================

if __name__ == "__main__":
    asyncio.run(run_benchmark())
