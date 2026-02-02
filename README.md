# H100 Benchmark Suite

Standardized performance evaluation for **8× NVIDIA H100 / A100 GPU clusters** using **vLLM**.

This suite benchmarks **real-world LLM inference workloads**, with a focus on:

- High-concurrency diarization inference
- Long-context clinical note generation
- Token-level latency and throughput
- Per-GPU efficiency on multi-GPU systems


---
## Huggingface Models that Require Access / Licence Acceptance

Meta LLaMA 3.1 8B Instruct
👉 https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

Llama-3.3-70B-Instruct
👉 https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct

Llama3-Med42-70B
👉 https://huggingface.co/m42-health/Llama3-Med42-70B

Llama3-Med42-8B
👉 https://huggingface.co/m42-health/Llama3-Med42-8B

---

## Recommended Way to Run

👉 **Use Local Mode (`--local`)**

👉 **When running Local Mode, the recommended project root is `/workspace/llm-benchmarks`**

Docker mode exists but is **untested**.

---

## Modes of Operation

### Local Mode (`--local`) — **Recommended**

**Prerequisites:**
- ~1TB disk space for all models

- Create a .env from the env_example. You will only have to provide your huggingface access token, if your project root is `/workspace/llm-benchmarks`

- Create a python venv in the project root
```bash
python3 -m venv ./venv
```

- Depending on the distro, you may need to install this:
```bash
apt install gettext-base
```

**What it does:**
- Runs directly on the host container
- Uses a persistent Python virtual environment
- Automatically checks for and activates the venv
- Uses persistent cache directories


**Run it:**
```bash
./run_suite.sh --local
