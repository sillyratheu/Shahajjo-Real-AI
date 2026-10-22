FROM python:3.9-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Requirements file copy kora
COPY requirements.txt .

# CPU-only PyTorch ebong baki package gulo install kora
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --index-url https://download.pytorch.org/whl/cpu

# Baki shob file copy kora
COPY . .

# Render-er default port $PORT use korar jonno
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}
