# Miracle 2.0 - 自主学习量化交易系统 Dockerfile
# ===============================================
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

# ============================================================
# Prometheus Metrics + Health Endpoints
# Port 8080 exposes:
#   GET /metrics  - Prometheus exposition format
#   GET /health   - Enhanced health check with component status
# ============================================================

# Enhanced healthcheck with component-level checks
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=3 \
    CMD python -c "
import sys
try:
    import requests
    r = requests.get('http://localhost:8080/health', timeout=5)
    if r.status_code >= 500:
        sys.exit(1)
    data = r.json()
    # Block on critical component errors
    for comp in data.get('components', []):
        if comp.get('status') == 'error' and comp.get('critical', False):
            sys.exit(1)
except Exception as e:
    print(f'Health check failed: {e}', file=sys.stderr)
    sys.exit(1)
"

# Run metrics server + main app
CMD ["python", "-c", "
import threading, logging, sys
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('startup')

# Start Flask metrics/health server in background
def run_metrics():
    try:
        from core.metrics import create_metrics_app
        app = create_metrics_app()
        if app:
            logger.info('Starting metrics server :8080 (/metrics, /health)')
            app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
        else:
            logger.warning('Flask not available, metrics disabled')
    except Exception as e:
        logger.error(f'Metrics server error: {e}')

t = threading.Thread(target=run_metrics, daemon=True)
t.start()
import time; time.sleep(2)

# Run main miracle app
sys.path.insert(0, '/app')
from miracle import main
main()
"]
