// run_suite.sh

#!/bin/bash

# 1. Setup & Checks
if [ -f .env ]; then
  export $(echo $(cat .env | sed 's/#.*//g' | xargs) | envsubst)
fi

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is missing. Please create a .env file with your Hugging Face token."
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

echo "======================================================="
echo "   PEOPLES DOCTOR - H100 BENCHMARK SUITE"
echo "======================================================="
echo "Date: $(date)"
echo "Output Directory: $RESULTS_DIR"
echo "Cache Directory: $CACHE_DIR"
echo "======================================================="

# Function to run benchmark with resilience
execute_benchmark() {
    local model=$1
    local tp=$2
    local type=$3
    local timeout=$4

    echo "----------------------------------------------------"
    echo ">>> TESTING: $model ($type)"
    echo "----------------------------------------------------"
    
    export MODEL_NAME=$model
    export TENSOR_PARALLEL_SIZE=$tp
    export TEST_TYPE=$type
    
    # Execute with timeout and capture exit status
    timeout "${timeout}s" docker compose up --build --abort-on-container-exit
    local status=$?

    if [ $status -eq 124 ]; then
        echo "!!! TIMEOUT: $model exceeded ${timeout}s limit. Skipping..." | tee -a "$ERROR_LOG"
    elif [ $status -ne 0 ]; then
        echo "!!! FAILURE: $model failed with exit code $status. Skipping..." | tee -a "$ERROR_LOG"
    else
        echo ">>> SUCCESS: $model completed."
    fi
    
    # Ensure cleanup occurs regardless of success or failure
    docker compose down
}

# -----------------------------------------------------------------------------
# PHASE 1: DIARIZATION JUDGE (Small/Medium Models)
# -----------------------------------------------------------------------------

DIARIZATION_MODELS=(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    "Qwen/Qwen2.5-14B-Instruct"
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
    "mistralai/Mistral-Nemo-Instruct-2407"
    "google/gemma-2-9b-it"
)

echo ">>> STARTING PHASE 1: DIARIZATION JUDGE"

for model in "${DIARIZATION_MODELS[@]}"; do
    execute_benchmark "$model" 1 "diarization" 480
    sleep 5
done

# -----------------------------------------------------------------------------
# PHASE 2: CLINICAL NOTES (Large/Medical Models)
# -----------------------------------------------------------------------------

CLINICAL_MODELS=(
    "epfl-llm/meditron-70b"
    "clinical-llama/Clinical-Llama-3-70B"
    "m42-health/med42-v2-70b"
    "aaditya/OpenBioLLM-Llama3-70B"
    "dracarys-llm/MedGemma-2-27b"
    "meta-llama/Llama-3.3-70B-Instruct"
    "Qwen/Qwen2.5-72B-Instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
)

echo ">>> STARTING PHASE 2: CLINICAL DEEP DIVE (SOAP Notes)"

for model in "${CLINICAL_MODELS[@]}"; do
    execute_benchmark "$model" 8 "clinical" 900
    sleep 10
done

echo "======================================================="
echo "   BENCHMARK COMPLETE - COMPRESSING RESULTS"
echo "======================================================="

if [ -s "$ERROR_LOG" ]; then
    echo "Warning: Some models failed. Check $ERROR_LOG for details."
fi

tar -czf benchmark_results.tar.gz results/
echo ">>> Results saved to benchmark_results.tar.gz"