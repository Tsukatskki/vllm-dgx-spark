#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[vLLM] Starting service..."
docker compose --env-file config.env up -d --build

echo "[vLLM] Waiting for health check (port $(grep VLLM_PORT config.env | cut -d= -f2))..."
PORT=$(grep VLLM_PORT config.env | cut -d= -f2)
RETRIES=40
for i in $(seq 1 $RETRIES); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[vLLM] Service is healthy!"
    exit 0
  fi
  echo "[vLLM] Attempt $i/$RETRIES - not ready yet, waiting 15s..."
  sleep 15
done

echo "[vLLM] ERROR: Service did not become healthy after $((RETRIES * 15))s"
docker compose --env-file config.env logs --tail=50
exit 1
