# PeoplesDoctor H100 Benchmark Suite

Standardized performance evaluation for 8x H100 GPU clusters, focusing on medical reasoning and conversational logic.

## Project Structure
- `client/`: Python benchmark engine and data generators.
- `hf_cache/`: Local model weight storage (created on first run).
- `results/`: JSON performance reports and error logs.
- `configs/`: Generated workload parameters.

## Setup Instructions

1.  **Environment:**
    Create a `.env` file in the root directory:
    ```ini
    HF_TOKEN=hf_your_token_here
    ```

2.  **Permissions:**
    Ensure the project directory is writable by your user (Docker will map these to the container):
    ```bash
    chmod +x run_suite.sh
    ```

3.  **Execution:**
    Run the suite in a persistent background session:
    ```bash
    screen -S benchmark
    ./run_suite.sh
    ```

## Benchmarked Tasks

### 1. Diarization Judge (Small Models)
- **Objective:** Semantic speaker identification using short-context logic.
- **Payload:** Messy transcripts with unknown labels.
- **Metric:** Tokens per second at high concurrency (64).

### 2. Clinical Deep Dive (Large Models)
- **Objective:** High-accuracy SOAP note generation.
- **Payload:** 16k context including EHR history and 15-minute consultation transcripts.
- **Metric:** TTFT (Prefill) and TPOT (Decoding) latencies.

## Troubleshooting

**VRAM Cleanup:**
If a process is interrupted, the GPUs may remain locked. Use:
```bash
docker compose down
sudo fuser -v /dev/nvidia*
```

**vLLM Loading Issues:**
Check `results/error_log.txt`. Most failures are due to `HF_TOKEN` lacking permissions for gated models (Llama 3.1/3.3).