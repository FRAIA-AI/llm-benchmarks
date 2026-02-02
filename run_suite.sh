#!/bin/bash

# run_suite.sh

# 1. Setup & Checks
if [ -f .env ]; then
  export $(echo $(cat .env | sed 's/#.*//g' | xargs) | envsubst)
fi

# Enforce cache dirs for all subprocesses
export HF_HOME
export HF_HUB_CACHE
export TRANSFORMERS_CACHE
export XDG_CACHE_HOME
export VLLM_CACHE_DIR
export TORCH_HOME

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is missing. Please create a .env file."
    exit 1
fi

RESULTS_DIR="./results"
CACHE_DIR="./hf_cache"
CONFIGS_DIR="./configs"
ERROR_LOG="./results/error_log.txt"

mkdir -p $RESULTS_DIR
mkdir -p $CACHE_DIR
mkdir -p $CONFIGS_DIR
touch $ERROR_LOG

MODE="docker"
if [[ "$1" == "--local" ]]; then
    MODE="local"
    echo ">>> Running in LOCAL mode (Direct host execution)"
else
    echo ">>> Running in DOCKER mode"
fi

echo "======================================================="
echo "   H100 BENCHMARK SUITE ($MODE)"
echo "======================================================="

# Helper for Local Execution
run_local() {
    local model=$1
    local tp=$2
    local type=$3
    local timeout=$4

    echo ">>> [LOCAL] Starting vLLM for $model (TP=$tp)..."
    
    # 1. Start vLLM in background
    # Note: We use nohup to prevent it dying if shell disconnects, 
    # but we trap cleanup below.
    # NCCL_P2P_LEVEL=NVL ensures NVLink usage on H100
    NCCL_P2P_LEVEL=NVL nohup python3 -m vllm.entrypoints.openai.api_server \
        --model $model \
        --tensor-parallel-size $tp \
        --max-model-len 16384 \
        --trust-remote-code \
        --disable-log-requests \
        --gpu-memory-utilization 0.90 \
        --max-num-seqs 128 \
        --port 8000 > vllm.log 2>&1 &
    
    SERVER_PID=$!
    echo ">>> vLLM PID: $SERVER_PID. Logs in vllm.log"

    # 2. Wait for health (handled by client logic basically, but we do a loop here to be safe)
    # The python client also has a health check, but we need to ensure the process actually started.
    sleep 10
    if ! ps -p $SERVER_PID > /dev/null; then
        echo "!!! vLLM failed to start immediately. Check vllm.log"
        cat vllm.log
        return 1
    fi

    # 3. Run Benchmark Client
    export MODEL_NAME=$model
    export TEST_TYPE=$type
    export VLLM_BASE_URL="http://localhost:8000"
    export RESULTS_DIR="./results"
    export CONFIG_DIR="./configs"
    
    echo ">>> Starting Benchmarker Client..."
    # We use 'timeout' on the client execution to enforce the time limit
    timeout "${timeout}s" python3 client/benchmark_runner.py
    CLIENT_STATUS=$?
    
    # 4. Cleanup
    echo ">>> Stopping vLLM..."
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null

    if [ $CLIENT_STATUS -eq 124 ]; then
         echo "!!! TIMEOUT: Client exceeded ${timeout}s." | tee -a "$ERROR_LOG"
    elif [ $CLIENT_STATUS -ne 0 ]; then
         echo "!!! FAILURE: Client failed." | tee -a "$ERROR_LOG"
    else
         echo ">>> SUCCESS: Benchmark complete."
    fi
}

# Helper for Docker Execution
run_docker() {
    local model=$1
    local tp=$2
    local type=$3
    local timeout=$4

    export MODEL_NAME=$model
    export TENSOR_PARALLEL_SIZE=$tp
    export TEST_TYPE=$type
    
    timeout "${timeout}s" docker compose up --build --abort-on-container-exit
    docker compose down
}

# Check Dependencies if Local
if [[ "$MODE" == "local" ]]; then
    if ! python3 -c "import vllm" &> /dev/null; then
        echo ">>> Installing vLLM and dependencies..."
        pip install --upgrade pip
        pip install vllm aiohttp numpy pandas tqdm huggingface_hub
    fi
    # Ensure logged in for gated models
    huggingface-cli login --token $HF_TOKEN
fi

# =========================================================
# MODEL DEFINITIONS
# =========================================================

# PHASE 1: DIARIZATION
DIARIZATION_MODELS=(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    "Qwen/Qwen2.5-14B-Instruct"
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
#    "mistralai/Mistral-Nemo-Instruct-2407"
    "google/gemma-2-9b-it"
)

echo ">>> STARTING PHASE 1: DIARIZATION JUDGE"

for model in "${DIARIZATION_MODELS[@]}"; do
    if [[ "$MODE" == "local" ]]; then
        run_local "$model" 1 "diarization" 480
    else
        run_docker "$model" 1 "diarization" 480
    fi
    sleep 5
done

# PHASE 2: CLINICAL NOTES
CLINICAL_MODELS=(
    "epfl-llm/meditron-70b"
    "clinicalnlplab/me-llama-70B-chat"
    "m42-health/Llama3-Med42-70B"
    "aaditya/Llama3-OpenBioLLM-70B"
    "abacusai/Dracarys2-72B-Instruct"
    "meta-llama/Llama-3.3-70B-Instruct"
    "Qwen/Qwen2.5-72B-Instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
)

echo ">>> STARTING PHASE 2: CLINICAL DEEP DIVE"

for model in "${CLINICAL_MODELS[@]}"; do
    if [[ "$MODE" == "local" ]]; then
        run_local "$model" 8 "clinical" 900
    else
        run_docker "$model" 8 "clinical" 900
    fi
    sleep 10
done

echo "======================================================="
echo "   BENCHMARK COMPLETE"
echo "======================================================="

tar -czf benchmark_results.tar.gz results/ vllm.log
echo ">>> Results saved to benchmark_results.tar.gz"