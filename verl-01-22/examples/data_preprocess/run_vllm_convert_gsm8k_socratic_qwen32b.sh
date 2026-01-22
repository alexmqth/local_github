#!/usr/bin/env bash
set -euo pipefail

# One-shot runner:
# 1) start local vLLM OpenAI-compatible server (Qwen2.5-32B-Instruct)
# 2) wait for readiness
# 3) run GSM8K Socratic -> step_verify_agent parquet conversion

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}/v1}"
export PORT HOST BASE_URL

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-32B-Instruct}"
SERVED_NAME="${SERVED_NAME:-qwen2.5-32b}"

SPLIT="${SPLIT:-train}"          # train | test
MAX_SAMPLES="${MAX_SAMPLES:-0}"  # 0 means all
OUT_PARQUET="${OUT_PARQUET:-data/gsm8k_socratic_step_verify/${SPLIT}.parquet}"

# vLLM quirks: on some clusters FlashInfer JIT may fail due to nvcc arch mismatch
export VLLM_USE_FLASHINFER="${VLLM_USE_FLASHINFER:-0}"

mkdir -p "$(dirname "${OUT_PARQUET}")"

echo "[INFO] Starting vLLM server on ${HOST}:${PORT} (model=${MODEL_PATH}, served_name=${SERVED_NAME})"
python -m vllm.entrypoints.openai.api_server \
  --host "${HOST}" \
  --port "${PORT}" \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --trust-remote-code \
  > vllm_server.log 2>&1 &

VLLM_PID=$!
echo "[INFO] vLLM pid=${VLLM_PID} (log=vllm_server.log)"

cleanup() {
  kill "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

sleep 2
if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
  echo "[ERROR] vLLM process exited early. Showing last 200 lines of vllm_server.log:"
  tail -n 200 vllm_server.log || true
  exit 1
fi

VLLM_WAIT_SEC="${VLLM_WAIT_SEC:-900}"
python - <<PY
import os, time, requests, sys
base = os.environ["BASE_URL"].rstrip("/")
wait_s = int(os.environ.get("VLLM_WAIT_SEC","900"))
health = base.replace("/v1","") + "/health" if base.endswith("/v1") else base + "/health"
models = base + "/models" if base.endswith("/v1") else base + "/v1/models"
for t in range(wait_s):
    try:
        r = requests.get(health, timeout=2)
        if r.status_code == 200:
            r2 = requests.get(models, timeout=2)
            if r2.status_code == 200:
                print("vLLM ready:", r2.json())
                sys.exit(0)
    except Exception:
        pass
    time.sleep(1)
raise SystemExit(f"vLLM not ready after {wait_s}s")
PY

echo "[INFO] Converting GSM8K socratic (${SPLIT}) -> ${OUT_PARQUET}"
python -u examples/data_preprocess/gsm8k_socratic_to_step_verify_agent_loop.py \
  --split "${SPLIT}" \
  --max_samples "${MAX_SAMPLES}" \
  --out_parquet "${OUT_PARQUET}" \
  --base_url "${BASE_URL}" \
  --model "${SERVED_NAME}" \
  --api_key "EMPTY" \
  --workers "${WORKERS:-4}" \
  --max_tokens "${MAX_TOKENS:-768}" \
  --max_steps "${MAX_STEPS:-12}" \
  --max_chars_each_step "${MAX_CHARS_EACH_STEP:-500}"

echo "[OK] Done. Output: ${OUT_PARQUET}"


