# client/benchmark_runner.py

import asyncio
import aiohttp
import time
import json
import os
import sys
import numpy as np
from datetime import datetime

# Path Configuration matching Docker volume mounts
VLLM_URL = "http://vllm:8000/v1/chat/completions"
HEALTH_URL = "http://vllm:8000/health"
MODEL_NAME = os.getenv("MODEL_NAME")
TEST_TYPE = os.getenv("TEST_TYPE", "diarization")
RESULTS_DIR = "/app/results"
WORKLOAD_FILE = "/app/configs/workloads.json"

async def make_request(session, payload):
    start_time = time.time()
    try:
        async with session.post(VLLM_URL, json=payload) as response:
            if response.status != 200:
                text = await response.text()
                return {"error": f"{response.status}: {text}"}
            
            result = await response.json()
            end_time = time.time()
            
            usage = result.get('usage', {})
            output_tokens = usage.get('completion_tokens', 0)
            input_tokens = usage.get('prompt_tokens', 0)
            
            latency = end_time - start_time
            tpot = (latency / output_tokens) if output_tokens > 0 else 0
            
            return {
                "latency": latency,
                "output_tokens": output_tokens,
                "input_tokens": input_tokens,
                "tpot": tpot
            }
    except Exception as e:
        return {"error": str(e)}

async def run_benchmark():
    print(">>> Generating Workload Data...")
    exit_code = os.system("python generate_data.py")
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
        "stream": False 
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
            
            if time.time() - start_wait > 900: 
                print("!!! vLLM Failed to Load Model in time !!!")
                sys.exit(1)
            await asyncio.sleep(5)

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
            "config": config
        },
        "results": {
            "total_requests": len(results),
            "successful": len(valid_results),
            "failed": len(errors),
            "total_duration_sec": total_duration,
            "throughput_tokens_per_sec": total_output_tokens / total_duration if total_duration > 0 else 0,
            "avg_latency_sec": np.mean([r['latency'] for r in valid_results]) if valid_results else 0,
            "p95_latency_sec": np.percentile([r['latency'] for r in valid_results], 95) if valid_results else 0,
            "avg_tpot_sec": np.mean([r['tpot'] for r in valid_results]) if valid_results else 0,
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