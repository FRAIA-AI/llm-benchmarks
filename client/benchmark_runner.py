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

# ... [make_request function remains unchanged] ...

async def run_benchmark():
    # 1. Generate Data (Ensures workloads.json exists in the mounted volume)
    print(">>> Generating Workload Data...")
    exit_code = os.system("python generate_data.py")
    if exit_code != 0:
        print("!!! Data Generation Failed !!!")
        sys.exit(1)
    
    # 2. Load Config from established path
    try:
        if not os.path.exists(WORKLOAD_FILE):
            raise FileNotFoundError(f"Missing {WORKLOAD_FILE}")
            
        with open(WORKLOAD_FILE, "r") as f:
            full_config = json.load(f)
            config = full_config[TEST_TYPE]
    except (KeyError, FileNotFoundError) as e:
        print(f"!!! Initialization Error: {str(e)} !!!")
        sys.exit(1)

    # ... [vLLM healthcheck and Execution loop remain unchanged] ...