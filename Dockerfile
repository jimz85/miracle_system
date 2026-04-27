# Miracle 2.0 - 自主学习量化交易系统 Dockerfile
# ================================================
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install sentence-transformers (heavy, do early for caching)
RUN pip install --no-cache-dir sentence-transformers

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser

# Create logs and data directories
RUN mkdir -p /app/logs /app/data && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Healthcheck using Python
HEALTHCHECK --interval=60s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health', timeout=5)" || exit 1

# Default command - run miracle in daemon mode
CMD ["python", "miracle.py", "--daemon"]
