# H100 Benchmark Suite

Standardized performance evaluation for **8× NVIDIA H100 / A100 GPU clusters** using **vLLM**.

This suite benchmarks **real-world LLM inference workloads**, with a focus on:

- High-concurrency diarization inference  
- Long-context clinical note generation  
- Token-level latency and throughput  
- Per-GPU efficiency on multi-GPU systems  

---

## Recommended Way to Run

👉 **Use Local Mode (`--local`)**

👉 **When running Local Mode, the recommended project root is `/workspace/llm-benchmarks`**

Docker mode exists but is **untested, slower, and not recommended**.

---

## Modes of Operation

### Local Mode (`--local`) — **Recommended**

**Use this for:**
- RunPod
- Bare-metal GPU nodes
- Long benchmark runs

**What it does:**
- Runs directly on the host container
- Uses a persistent Python virtual environment
- Automatically activates the venv
- Uses persistent cache directories
- Avoids reinstalling dependencies after pod restarts
- Provides best performance and stability


**Run it:**
```bash
./run_suite.sh --local
