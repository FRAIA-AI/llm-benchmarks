# client/benchmark_runner.py

import asyncio
import aiohttp
import time
import json
import os
import sys
import numpy as np
from datetime import datetime

# Environment Config
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
VLLM_URL = f"{VLLM_BASE_URL}/v1/chat/completions"
HEALTH_URL = f"{VLLM_BASE_URL}/health"

# Path Config (Defaults to local paths for non-docker usage)
MODEL_NAME = os.getenv("MODEL_NAME")
TEST_TYPE = os.getenv("TEST_TYPE", "diarization")
RESULTS_DIR = os.getenv("RESULTS_DIR", "./results")
CONFIG_DIR = os.getenv("CONFIG_DIR", "./configs")
WORKLOAD_FILE = os.path.join(CONFIG_DIR, "workloads.json")

async def make_request(session, payload):
    start_time = time.time()
    ttft = None
    last_token_time = None
    inter_token_latencies = []
    output_tokens = 0
    
    # Force stream=True for high-res metrics
    payload["stream"] = True
    
    try:
        async with session.post(VLLM_URL, json=payload) as response:
            if response.status != 200:
                text = await response.text()
                return {"error": f"{response.status}: {text}"}
            
            async for line in response.content:
                chunk_time = time.time()
                line = line.decode('utf-8').strip()
                
                if not line or line == "data: [DONE]":
                    continue
                
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        
                        # Track first token (Prefill phase)
                        if ttft is None:
                            ttft = chunk_time - start_time
                            last_token_time = chunk_time
                        else:
                            # Track subsequent tokens (Decoding phase)
                            itl = chunk_time - last_token_time
                            inter_token_latencies.append(itl)
                            last_token_time = chunk_time
                        
                        delta = data.get('choices', [{}])[0].get('delta', {})
                        if delta.get('content'):
                            output_tokens += 1
                    except json.JSONDecodeError:
                        continue

            end_time = time.time()
            total_latency = end_time - start_time
            
            if output_tokens == 0:
                return {"error": "Empty response from model"}

            decoding_duration = end_time - (start_time + ttft) if ttft else 0
            tpot = decoding_duration / output_tokens if output_tokens > 0 else 0

            return {
                "total_latency": total_latency,
                "ttft": ttft,
                "tpot": tpot,
                "itl_avg": np.mean(inter_token_latencies) if inter_token_latencies else 0,
                "output_tokens": output_tokens,
                "input_tokens": 0 
            }
    except Exception as e:
        return {"error": str(e)}

async def run_benchmark():
    print(">>> Generating Workload Data...")
    # Run the generator script relative to current location
    exit_code = os.system(f"python client/generate_data.py")
    if exit_code != 0:
        print("!!! Data Generation Failed !!!")
        sys.exit(1)
    
    try:
        if not os.path.exists(WORKLOAD_FILE):
            raise FileNotFoundError(f"Missing {WORKLOAD_FILE}")
            
        with open(WORKLOAD_FILE, "r") as f:
            full_config = json.load(f)
            config = full_config[TEST_TYPE]
    except (KeyError, FileNotFoundError) as e:
        print(f"!!! Initialization Error: {str(e)} !!!")
        sys.exit(1)

    print(f"=================================================")
    print(f" STARTING BENCHMARK: {TEST_TYPE.upper()}")
    print(f" Model: {MODEL_NAME}")
    print(f" Target: {VLLM_BASE_URL}")
    print(f" Concurrency: {config['concurrency']}")
    print(f" Requests: {config['requests_count']}")
    print(f"=================================================")

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": config['system_prompt']},
            {"role": "user", "content": config['user_prompt']}
        ],
        "max_tokens": config['max_tokens'],
        "temperature": config['temperature'],
        "stream": True 
    }

    async with aiohttp.ClientSession() as session:
        print(">>> Waiting for vLLM to initialize...")
        start_wait = time.time()
        while True:
            try:
                async with session.get(HEALTH_URL) as r:
                    if r.status == 200:
                        print(f">>> vLLM Ready (Waited {int(time.time() - start_wait)}s)")
                        break
            except:
                pass
            
            if time.time() - start_wait > 900: # 15 min wait for local loading
                print("!!! vLLM Failed to Load Model in time !!!")
                sys.exit(1)
            await asyncio.sleep(5)

        print(">>> Starting stress test...")
        start_bench = time.time()
        tasks = []
        sem = asyncio.Semaphore(config['concurrency'])
        
        async def bound_request():
            async with sem:
                return await make_request(session, payload)

        for _ in range(config['requests_count']):
            tasks.append(bound_request())

        results = await asyncio.gather(*tasks)
        total_duration = time.time() - start_bench

    valid_results = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    
    total_output_tokens = sum(r['output_tokens'] for r in valid_results)
    
    metrics = {
        "meta": {
            "model": MODEL_NAME,
            "test_type": TEST_TYPE,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "environment": "local" if "localhost" in VLLM_BASE_URL else "docker"
        },
        "results": {
            "total_requests": len(results),
            "successful": len(valid_results),
            "failed": len(errors),
            "total_duration_sec": total_duration,
            "throughput_tokens_per_sec": total_output_tokens / total_duration if total_duration > 0 else 0,
            "avg_latency_sec": np.mean([r['total_latency'] for r in valid_results]) if valid_results else 0,
            "avg_ttft_ms": np.mean([r['ttft'] for r in valid_results]) * 1000 if valid_results else 0,
            "avg_tpot_ms": np.mean([r['tpot'] for r in valid_results]) * 1000 if valid_results else 0,
            "p95_latency_ms": np.percentile([r['total_latency'] for r in valid_results], 95) * 1000 if valid_results else 0,
            "avg_itl_ms": np.mean([r['itl_avg'] for r in valid_results]) * 1000 if valid_results else 0,
            "errors": errors[:5] 
        }
    }

    safe_model_name = MODEL_NAME.replace("/", "_")
    filename = f"{TEST_TYPE}_{safe_model_name}_{int(time.time())}.json"
    filepath = os.path.join(RESULTS_DIR, filename)
    
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f">>> Report saved: {filepath}")
    print(json.dumps(metrics['results'], indent=2))

if __name__ == "__main__":
    asyncio.run(run_benchmark())