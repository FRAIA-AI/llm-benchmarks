# H100 Benchmark Suite

Standardized performance evaluation for 8x H100/A100 GPU clusters.

## Modes of Operation

1.  **Docker Mode (Default):** Uses `docker-compose`. Requires Docker-in-Docker support.
2.  **Local Mode (`--local`):** Runs directly on the host. Required several packages to be installed.

## Setup (RunPod)

1.  **Clone this repository** to your Pod.
2.  **Configure Environment:**
    ```bash
    cp .env.example .env
    # Edit .env with your HF_TOKEN
    ```
3.  **Run Benchmark:**
    Run inside a `tmux` session as this takes ~2 hours.
    
    ```bash
    chmod +x run_suite.sh
    ./run_suite.sh --local
    ```
    *The script will automatically install vLLM and dependencies on the first run.*

## Output
Results are saved in `benchmark_results.tar.gz`.

## Workloads
*   **Diarization:** Small models (7-14B), High concurrency.
*   **Clinical Notes:** Large models (70B+), Long context (16k), Medical accuracy.