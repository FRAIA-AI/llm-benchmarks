# PeoplesDoctor H100 Benchmark Suite

This repository contains a standardized benchmarking suite designed to test LLM performance on **8x H100 clusters**. 

It evaluates two specific workloads critical to the PeoplesDoctor platform:
1.  **Diarization Logic:** High-speed correction of speaker transcripts.
2.  **Clinical Documentation:** Long-context SOAP note generation.

## Prerequisites

*   **OS:** Linux (Ubuntu 22.04+ recommended)
*   **Hardware:** 8x NVIDIA H100 GPUs (80GB VRAM) with NVLink.
*   **Software:** Docker, Docker Compose, NVIDIA Container Toolkit.

## Quick Start

1.  **Clone this repository** to the GPU node.
2.  **Configure Environment:**
    Create a `.env` file and add your token:
    ```ini
    HF_TOKEN=hf_your_token_here
    ```
3.  **Run the Benchmark:**
    ```bash
    chmod +x run_suite.sh
    ./run_suite.sh
    ```
    *Note: The suite takes approximately 2 hours. Use `tmux` or `screen`.*

4.  **Retrieve Results:**
    Send the generated `benchmark_results.tar.gz` back to the engineering team.

## Architecture

*   **Model Cache:** Weights are stored locally in `./hf_cache` within the project folder to ensure write permissions and portability.
*   **Phase 1 (Diarization):** Tests small models (7B-14B) for conversational logic throughput.
*   **Phase 2 (Clinical):** Tests large models (27B-70B+) for medical reasoning and long-context (16k) SOAP note accuracy.

## Troubleshooting

**OOM Errors:**
If a run crashes, clean the VRAM manually:
```bash
docker compose down
sudo fuser -k /dev/nvidia*
```

**Permission Issues:**
The suite automatically creates `./hf_cache` and `./results`. If Docker cannot write to these, ensure the current user has ownership of the project directory:
```bash
sudo chown -R $USER:$USER .
```