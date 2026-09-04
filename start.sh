#!/bin/bash
set -e

PUBLIC_PORT=${PORT:-8000}
ML_PORT=8050

echo "==================================================="
echo " Starting AI Cloud Resource Optimization Platform"
echo " Public Gateway & UI Port: ${PUBLIC_PORT}"
echo " Internal ML Engine Port:   ${ML_PORT}"
echo "==================================================="

export PYTHONPATH=/app:${PYTHONPATH}

echo "-> Launching ML Service on port ${ML_PORT}..."
python -m uvicorn ml_service.main:app --host 127.0.0.1 --port ${ML_PORT} &
ML_PID=$!

echo "-> Waiting for ML Service initialization..."
for i in $(seq 1 45); do
    if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8050/health')" 2>/dev/null; then
        echo "? ML Service is healthy and operational (PID: ${ML_PID})."
        break
    fi
    sleep 1
done

echo "-> Starting API Gateway on port ${PUBLIC_PORT}..."
exec python -m uvicorn app.main:app --host 0.0.0.0 --port ${PUBLIC_PORT}
