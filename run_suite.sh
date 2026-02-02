#!/bin/bash

# run_suite.sh

# 1. Setup & Checks
if [ -f .env ]; then
  export $(echo $(cat .env | sed 's/#.*//g' | xargs) | envsubst)
fi

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is missing. Please create a .env file with your Hugging Face token."
    exit 1
fi

RESULTS_DIR="./results"
mkdir -p $RESULTS_DIR

echo "======================================================="
echo "   PEOPLES DOCTOR - H100 BENCHMARK SUITE"
echo "======================================================="
echo "Date: $(date)"
echo "Output Directory: $RESULTS_DIR"
echo "======================================================="

# -----------------------------------------------------------------------------
# PHASE 1: DIARIZATION JUDGE (Small/Medium Models)
# Workload: Short Context, Logic Heavy, High Concurrency
# Time Allocation: ~30 Minutes
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
    echo "----------------------------------------------------"
    echo ">>> TESTING: $model"
    echo "----------------------------------------------------"
    
    export MODEL_NAME=$model
    # TP=1 allows vLLM to manage batching on a single GPU (or replicate across GPUs)
    export TENSOR_PARALLEL_SIZE=1 
    export TEST_TYPE="diarization"
    
    # Run Benchmark (Timeout: 6 minutes)
    timeout 360s docker compose up --build --abort-on-container-exit
    
    # Cleanup to free VRAM
    docker compose down
    sleep 5
done

# -----------------------------------------------------------------------------
# PHASE 2: CLINICAL NOTES (Large/Medical Models)
# Workload: Long Context (16k), Medical Reasoning, Low Concurrency
# Time Allocation: ~80 Minutes
# -----------------------------------------------------------------------------

CLINICAL_MODELS=(
    # --- The Medical Specialists ---
    "epfl-llm/meditron-70b"
    "clinical-llama/Clinical-Llama-3-70B"
    "m42-health/med42-v2-70b"
    "aaditya/OpenBioLLM-Llama3-70B"
    "dracarys-llm/MedGemma-2-27b" # Efficiency Test

    # --- The Generalist Baselines ---
    "meta-llama/Llama-3.3-70B-Instruct"
    "Qwen/Qwen2.5-72B-Instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
)

echo ">>> STARTING PHASE 2: CLINICAL DEEP DIVE (SOAP Notes)"

for model in "${CLINICAL_MODELS[@]}"; do
    echo "----------------------------------------------------"
    echo ">>> TESTING: $model"
    echo "----------------------------------------------------"
    
    export MODEL_NAME=$model
    # TP=8 forces the model to split across all 8 GPUs using NVLink/Infinity Fabric
    export TENSOR_PARALLEL_SIZE=8 
    export TEST_TYPE="clinical"
    
    # Run Benchmark (Timeout: 11 minutes - allows for longer load times)
    timeout 650s docker compose up --build --abort-on-container-exit
    
    # Force Cleanup
    docker compose down
    sleep 10
done

echo "======================================================="
echo "   BENCHMARK COMPLETE - COMPRESSING RESULTS"
echo "======================================================="

tar -czf benchmark_results.tar.gz results/
echo ">>> Results saved to benchmark_results.tar.gz"