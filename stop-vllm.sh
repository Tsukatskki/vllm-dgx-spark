#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[vLLM] Stopping service..."
docker compose --env-file config.env down

echo "[vLLM] Service stopped."
