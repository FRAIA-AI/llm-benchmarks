# H100 Benchmark Suite

Standardized performance evaluation for 8× NVIDIA H100 / A100 GPU clusters using vLLM.

This suite benchmarks **real-world LLM inference workloads**, with a strong focus on:
- High-concurrency diarization-style inference
- Long-context clinical note generation
- Token-level latency and throughput metrics
- Per-GPU efficiency on multi-H100 systems

It is designed to run reliably on **RunPod** and similar GPU providers.

---

## Modes of Operation

### Docker Mode (default)
- Uses docker compose
- Requires Docker-in-Docker support
- Not recommended for RunPod unless preconfigured

### Local Mode (`--local`) — Recommended
- Runs directly on the host container
- Uses a **persistent Python virtual environment**
- Avoids reinstalling dependencies after pod restarts

---

## RunPod Storage Model (Critical)

RunPod provides two storage layers:

| Location | Persistence |
|--------|-------------|
| Container filesystem (`/root`, `/usr/local`) | Ephemeral (wiped on stop) |
| Volume disk (`/workspace`) | Persistent |

**All important data must live under `/workspace`.**

This includes:
- Python virtualenv
- Hugging Face caches
- vLLM caches
- Model downloads
- Benchmark results

---

## One-Time Setup (RunPod)

### 1. Clone the repository into the persistent volume

```bash
cd /workspace
git clone <your-repo-url> llm-benchmarks
cd llm-benchmarks
```

---

### 2. Create a persistent Python virtualenv (ONE TIME)

```bash
cd /workspace
python3 -m venv venv
```

This creates:

```
/workspace/venv
```

The virtualenv **survives pod restarts**.

---

### 3. Activate the virtualenv

```bash
source /workspace/venv/bin/activate
```

You should see:

```
(venv) root@...
```

---

### 4. Install Python dependencies (ONE TIME)

```bash
pip install --upgrade pip
pip install \
  vllm \
  transformers \
  huggingface_hub \
  accelerate \
  aiohttp \
  numpy \
  pandas \
  tqdm
```

After this, **no pip installs are needed again** unless you change dependencies.

---

## Hugging Face Authentication (Required)

Most clinical and LLaMA-based models are gated.

### 1. Create a `.env` file

```bash
cd /workspace/llm-benchmarks
nano .env
```

Add:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
```

---

### 2. Login once using the Hugging Face CLI

```bash
huggingface-cli login --token $HF_TOKEN
```

This enables access to:
- Llama 3.x
- Med42
- Clinical fine-tuned models

---

## Cache Configuration (Mandatory)

The benchmark enforces **persistent cache directories** so models and kernels are reused.

The following environment variables are set by `run_suite.sh`:

```
HF_HOME
HF_HUB_CACHE
TRANSFORMERS_CACHE
XDG_CACHE_HOME
VLLM_CACHE_DIR
TORCH_HOME
```

All of them point to subdirectories under:

```
/workspace/llm-benchmarks/
```

This prevents:
- Disk exhaustion
- Redownloading models
- Rebuilding kernels on restart

---

## Running the Benchmark Suite

### Always activate the virtualenv first

```bash
source /workspace/venv/bin/activate
```

---

### Run in LOCAL mode (recommended)

```bash
./run_suite.sh --local
```

What this does:
- Launches vLLM directly on the host
- Uses tensor parallelism for multi-GPU models
- Runs benchmarks sequentially
- Saves detailed per-model JSON reports

IMPORTANT:
- Run inside `tmux` or `screen`
- Full runs may take 1–2 hours on 8× H100

---

## Benchmark Phases

### Phase 1 — Diarization
- Smaller models (8–14B)
- High concurrency
- Short prompts
- Throughput-focused

### Phase 2 — Clinical Deep Dive
- Large models (70B+)
- Long context
- Structured clinical note generation
- Latency and token efficiency focused

---

## Output Artifacts

After completion:

```
results/
├── diarization_*.json
├── clinical_*.json
├── error_log.txt
└── benchmark_results.tar.gz
```

Each JSON result includes:
- Total requests
- Successful / failed requests
- Total runtime
- Tokens per second
- Average latency per request
- TTFT (Time To First Token)
- TPOT (Time Per Output Token)
- p95 latency
- Token-normalized metrics
- Per-GPU efficiency metrics

---

## Fast Restart Workflow (After Pod Stop)

When restarting the pod:

```bash
cd /workspace/llm-benchmarks
./run_suite.sh --local
```

No reinstall required:
- Python packages persist
- Models persist
- Caches persist

