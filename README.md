# H100 Benchmark Suite

Standardized performance evaluation for **8× NVIDIA H100 / A100 GPU clusters** using **vLLM**.

This suite benchmarks **real-world LLM inference workloads**, focusing on:

- High-concurrency diarization-style inference  
- Long-context clinical note generation  
- Token-level latency and throughput metrics  
- Per-GPU efficiency on multi-H100 systems  

---

## Modes of Operation

The benchmark suite supports **two fully separate execution modes**.

---

## Docker Mode (default)

**Intended use**
- CI environments
- Preconfigured Docker setups
- Non-RunPod systems

**Characteristics**
- Uses `docker compose`
- Requires Docker-in-Docker support
- Dependencies are installed inside containers
- Slower cold starts
- Not recommended on RunPod unless already set up

**How to run**
```bash
./run_suite.sh
```

---

## Local Mode (`--local`) — Recommended for RunPod

**Intended use**
- RunPod
- Bare-metal GPU nodes
- Long-running benchmark sessions

**Characteristics**
- Runs directly on the host container
- Uses a **persistent Python virtual environment**
- Automatically activates the venv
- Uses persistent cache directories
- Avoids reinstalling dependencies after pod restarts
- Best performance and stability

**How to run**
```bash
./run_suite.sh --local
```

You do **not** need to activate the virtualenv manually —  
this is handled entirely by `run_suite.sh`.

---

## Hugging Face Authentication (Required)

Most clinical and LLaMA-based models are **gated**.  
Authentication is required in **both Docker and Local modes**.

### 1. Create a `.env` file

From the repository root:

```bash
cd /workspace/llm-benchmarks
nano .env
```

Add:

```env
HF_TOKEN=your_huggingface_token_here
```

That is **all you need to provide**.

---

### 2. Login once (Local Mode only)

When running in `--local` mode, the script uses the modern Hugging Face CLI:

```bash
hf auth login --token $HF_TOKEN
```

This enables access to:
- Llama 3.x
- Med42
- Clinical fine-tuned models

This only needs to be done **once per pod**.

---

## Cache & Persistence (Local Mode)

In **Local mode**, the script automatically configures **persistent cache directories**.

The following environment variables are set internally by `run_suite.sh`:

```
HF_HOME
HF_HUB_CACHE
TRANSFORMERS_CACHE
XDG_CACHE_HOME
VLLM_CACHE_DIR
TORCH_HOME
```

All caches live under:

```
/workspace/llm-benchmarks/.cache/
```

This ensures:
- Models are not re-downloaded
- CUDA kernels are reused
- Disk usage stays predictable
- Pod restarts are fast

Docker mode does **not** use these persistent caches.

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
- Latency and token-efficiency focused

Phases can be enabled or disabled by commenting models in `run_suite.sh`.

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

Each JSON report contains:

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

## Fast Restart Workflow (RunPod)

After stopping and restarting a pod:

```bash
cd /workspace/llm-benchmarks
./run_suite.sh --local
```

No manual steps required:
- Virtualenv persists
- Python packages persist
- Models persist
- Caches persist
