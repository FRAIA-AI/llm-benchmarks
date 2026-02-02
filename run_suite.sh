#!/bin/bash
set -euo pipefail

# =========================================================
# run_suite.sh — H100 Benchmark Suite
# =========================================================

# ---------------------------------------------------------
# Load environment
# ---------------------------------------------------------
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is missing. Please create a .env file."
    exit 1
fi

# ---------------------------------------------------------
# Mode detection
# ---------------------------------------------------------
MODE="docker"
if [[ "${1:-}" == "--local" ]]; then
    MODE="local"
    echo ">>> Running in LOCAL mode (Direct host execution)"
else
    echo ">>> Running in DOCKER mode"
fi

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
VENV_PATH="./venv"
RESULTS_DIR="./results"
CONFIGS_DIR="./configs"
ERROR_LOG="./results/error_log.txt"
HF_CACHE_ROOT="./hf_cache"

# ---------------------------------------------------------
# Local-only setup
# ---------------------------------------------------------
if [[ "$MODE" == "local" ]]; then
    if [ ! -d "$VENV_PATH" ]; then
        echo "ERROR: Virtualenv not found at $VENV_PATH"
        echo "Create it once with:"
        echo "  python3 -m venv ./venv"
        exit 1
    fi
    source "$VENV_PATH/bin/activate"

    export HF_HOME="$HF_CACHE_ROOT/hf_home"
    export HF_HUB_CACHE="$HF_CACHE_ROOT/hub"
    export TRANSFORMERS_CACHE="$HF_CACHE_ROOT/transformers"
    export XDG_CACHE_HOME="$HF_CACHE_ROOT/xdg"
    export VLLM_CACHE_DIR="$HF_CACHE_ROOT/vllm"
    export TORCH_HOME="$HF_CACHE_ROOT/torch"

    mkdir -p \
        "$HF_HOME" \
        "$HF_HUB_CACHE" \
        "$TRANSFORMERS_CACHE" \
        "$XDG_CACHE_HOME" \
        "$VLLM_CACHE_DIR" \
        "$TORCH_HOME"

    if ! python3 -c "import vllm" &>/dev/null; then
        echo ">>> Installing vLLM and dependencies..."
        pip install --upgrade pip
        pip install vllm aiohttp numpy pandas tqdm huggingface_hub
    fi

    if ! hf auth whoami &>/dev/null; then
        hf auth login --token "$HF_TOKEN" --add-to-git-credential
    fi

    cleanup() {
        pkill -f vllm.entrypoints.openai.api_server || true
    }
    trap cleanup EXIT INT TERM
fi

# ---------------------------------------------------------
# Common setup
# ---------------------------------------------------------
mkdir -p "$RESULTS_DIR" "$CONFIGS_DIR"
touch "$ERROR_LOG"

echo "======================================================="
echo "   H100 BENCHMARK SUITE ($MODE)"
echo "======================================================="

# ---------------------------------------------------------
# Model-specific max context
# ---------------------------------------------------------
get_max_len() {
  case "$1" in
    m42-health/Llama3-Med42-70B) echo 8192 ;;
    m42-health/Llama3-Med42-8B) echo 8192 ;;
    aaditya/Llama3-OpenBioLLM-70B) echo 8192 ;;
    abacusai/Dracarys2-72B-Instruct) echo 16384 ;;
    Qwen/Qwen2.5-72B-Instruct) echo 16384 ;;
    meta-llama/Llama-3.3-70B-Instruct) echo 16384 ;;
    deepseek-ai/DeepSeek-R1-Distill-Llama-70B) echo 16384 ;;
    *) echo 8192 ;;
  esac
}

# ---------------------------------------------------------
# Model capability: chat vs completion
# ---------------------------------------------------------
get_model_mode() {
  case "$1" in
   some-completion-model-identifier*)
      echo "completion"
      ;;
    *)
      echo "chat"
      ;;
  esac
}

# ---------------------------------------------------------
# Local execution helper
# ---------------------------------------------------------
run_local() {
    local model=$1
    local tp=$2
    local type=$3
    local timeout=$4

    export MODEL_MODE
    MODEL_MODE="$(get_model_mode "$model")"

    echo ">>> [LOCAL] Starting vLLM for $model (TP=$tp, MODE=$MODEL_MODE)..."

    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((tp-1)))

    NCCL_P2P_LEVEL=NVL nohup python3 -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --served-model-name "$model" \
        --tensor-parallel-size "$tp" \
        --max-model-len "$(get_max_len "$model")" \
        --trust-remote-code \
        --disable-log-requests \
        --gpu-memory-utilization 0.90 \
        --max-num-seqs 128 \
        --port 8000 > vllm.log 2>&1 &

    SERVER_PID=$!
    echo ">>> vLLM PID: $SERVER_PID"

    READY=false
    for _ in {1..60}; do
        if curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
            READY=true
            echo ">>> vLLM is responding"
            break
        fi
        sleep 5
    done

    if [[ "$READY" != "true" ]]; then
        echo "!!! vLLM failed to become ready"
        cat vllm.log
        kill "$SERVER_PID" || true
        wait "$SERVER_PID" 2>/dev/null || true
        return 1
    fi

    export MODEL_NAME="$model"
    export TEST_TYPE="$type"
    export VLLM_BASE_URL="http://localhost:8000"
    export RESULTS_DIR
    export CONFIG_DIR="$CONFIGS_DIR"

    echo ">>> Starting Benchmarker Client..."
    timeout "${timeout}s" python3 client/benchmark_runner.py || true

    echo ">>> Stopping vLLM..."
    kill "$SERVER_PID"
    wait "$SERVER_PID" 2>/dev/null || true
}

# ---------------------------------------------------------
# Docker execution helper
# ---------------------------------------------------------
run_docker() {
    local model=$1
    local tp=$2
    local type=$3
    local timeout=$4

    export MODEL_NAME="$model"
    export TENSOR_PARALLEL_SIZE="$tp"
    export TEST_TYPE="$type"

    timeout "${timeout}s" docker compose up --build --abort-on-container-exit
    docker compose down
}

# =========================================================
# PHASE 1: DIARIZATION JUDGE
# =========================================================
export PHASE_NAME="diarization"
DIARIZATION_MODELS=(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    "Qwen/Qwen2.5-14B-Instruct"
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
    "mistralai/Mistral-Nemo-Instruct-2407"
    "m42-health/Llama3-Med42-8B"
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

# =========================================================
# PHASE 2: CLINICAL NOTES
# =========================================================
export PHASE_NAME="clinical_c4"

CLINICAL_MODELS=(
    "m42-health/Llama3-Med42-70B"
    "m42-health/Llama3-Med42-8B"
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

# =========================================================
# PHASE 3: CLINICAL NOTES (HIGH CONCURRENCY)
# =========================================================
echo ">>> STARTING PHASE 3: CLINICAL DEEP DIVE (CONCURRENCY=$CLINICAL_CONCURRENT)"

export CLINICAL_CONCURRENCY_OVERRIDE="8"
export PHASE_NAME="clinical_c8"

CLINICAL_MODELS=(
    "m42-health/Llama3-Med42-70B"
    "m42-health/Llama3-Med42-8B"
)

for model in "${CLINICAL_MODELS[@]}"; do
    if [[ "$MODE" == "local" ]]; then
        run_local "$model" 8 "clinical" 900
    else
        run_docker "$model" 8 "clinical" 900
    fi
    sleep 10
done

unset CLINICAL_CONCURRENCY_OVERRIDE

# =========================================================
# PHASE 4: CLINICAL NOTES (SingleGPU small model)
# =========================================================
echo ">>> STARTING PHASE 4: CLINICAL DEEP DIVE (CONCURRENCY=$CLINICAL_CONCURRENT)"

export CLINICAL_CONCURRENCY_OVERRIDE="15"
export PHASE_NAME="clinical_gpu1_c15"

CLINICAL_MODELS=(
    "m42-health/Llama3-Med42-8B"
)

for model in "${CLINICAL_MODELS[@]}"; do
    if [[ "$MODE" == "local" ]]; then
        run_local "$model" 1 "clinical" 900
    else
        run_docker "$model" 1 "clinical" 900
    fi
    sleep 10
done

unset CLINICAL_CONCURRENCY_OVERRIDE

echo "======================================================="
echo "   BENCHMARK COMPLETE"
echo "======================================================="

tar -czf benchmark_results.tar.gz results/ vllm.log
echo ">>> Results saved to benchmark_results.tar.gz"
