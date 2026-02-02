# client/Dockerfile

FROM python:3.10-slim
WORKDIR /app

# Copy and install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application scripts
COPY . .

# Execute the benchmark runner
CMD ["python", "benchmark_runner.py"]