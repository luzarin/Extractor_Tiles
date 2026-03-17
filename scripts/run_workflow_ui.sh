#!/bin/bash
PORT=${1:-8090}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

cd "$ROOT_DIR" || exit 1
uv run uvicorn api.workflow_ui_api:app --app-dir "$ROOT_DIR" --host 127.0.0.1 --port "$PORT" --reload
