#!/usr/bin/env bash
set -euo pipefail

# Start a local vLLM OpenAI-compatible server for the BACKWARD agent (trained),
# then run MATH-500 evaluation while keeping FORWARD as remote.
#
# Default backward model path (as requested):
#   /local/scratch/zqin30/checkpoints/math/agent_loop/qwen2_5_7b_base_mathdata_backward_v1
#
# Requirements:
# - Linux shell with bash
# - vLLM installed (either `vllm serve` or `python -m vllm.entrypoints.openai.api_server`)
# - python env with `datasets` and `aiohttp` to run scripts/eval_backward_agent_math500.py
#
# Usage:
#   bash scripts/run_eval_backward_local_vllm.sh \
#     --forward_base_url http://REMOTE_FORWARD:8000/v1 \
#     --forward_model YOUR_FORWARD_SERVED_NAME \
#     --max_samples 200 \
#     --backward_port 9000 \
#     --backward_max_concurrency 8 \
#     --forward_max_concurrency 8
#
# Optional:
#   --keep_server 1   # do not kill vLLM after evaluation
#   --backward_model_path /path/to/ckpt
#   --backward_served_name backward_local
#   --host 127.0.0.1

BACKWARD_MODEL_PATH="/local/scratch/zqin30/checkpoints/math/agent_loop/qwen2_5_7b_base_mathdata_backward_v1"
BACKWARD_SERVED_NAME="backward_local"
HOST="127.0.0.1"
BACKWARD_PORT="9000"
BACKWARD_MAX_CONCURRENCY="8"

FORWARD_BASE_URL=""
FORWARD_MODEL=""
FORWARD_API_KEY=""
FORWARD_MAX_CONCURRENCY="8"

MAX_SAMPLES="200"
MAX_ROUNDS="24"
SPLIT="test"
OUT_JSONL="outputs/math500_backward_eval.jsonl"

KEEP_SERVER="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backward_model_path) BACKWARD_MODEL_PATH="$2"; shift 2;;
    --backward_served_name) BACKWARD_SERVED_NAME="$2"; shift 2;;
    --host) HOST="$2"; shift 2;;
    --backward_port) BACKWARD_PORT="$2"; shift 2;;
    --backward_max_concurrency) BACKWARD_MAX_CONCURRENCY="$2"; shift 2;;

    --forward_base_url) FORWARD_BASE_URL="$2"; shift 2;;
    --forward_model) FORWARD_MODEL="$2"; shift 2;;
    --forward_api_key) FORWARD_API_KEY="$2"; shift 2;;
    --forward_max_concurrency) FORWARD_MAX_CONCURRENCY="$2"; shift 2;;

    --max_samples) MAX_SAMPLES="$2"; shift 2;;
    --max_rounds) MAX_ROUNDS="$2"; shift 2;;
    --split) SPLIT="$2"; shift 2;;
    --out_jsonl) OUT_JSONL="$2"; shift 2;;
    --keep_server) KEEP_SERVER="$2"; shift 2;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${FORWARD_BASE_URL}" || -z "${FORWARD_MODEL}" ]]; then
  echo "ERROR: You must provide --forward_base_url and --forward_model" >&2
  exit 2
fi

mkdir -p "$(dirname "${OUT_JSONL}")"

BACKWARD_BASE_URL="http://${HOST}:${BACKWARD_PORT}/v1"
PID_FILE="/tmp/vllm_backward_${BACKWARD_PORT}.pid"
LOG_FILE="/tmp/vllm_backward_${BACKWARD_PORT}.log"

cleanup() {
  if [[ "${KEEP_SERVER}" == "1" ]]; then
    echo "[run] KEEP_SERVER=1, leaving vLLM running (pid file: ${PID_FILE})"
    return
  fi
  if [[ -f "${PID_FILE}" ]]; then
    local pid
    pid="$(cat "${PID_FILE}" || true)"
    if [[ -n "${pid}" ]]; then
      echo "[run] stopping vLLM (pid=${pid})"
      kill "${pid}" >/dev/null 2>&1 || true
    fi
    rm -f "${PID_FILE}" || true
  fi
}
trap cleanup EXIT

echo "[run] starting local vLLM backward server"
echo "[run] model_path=${BACKWARD_MODEL_PATH}"
echo "[run] served_name=${BACKWARD_SERVED_NAME}"
echo "[run] base_url=${BACKWARD_BASE_URL}"

if command -v vllm >/dev/null 2>&1; then
  # Newer vLLM provides `vllm serve` with OpenAI-compatible server.
  # NOTE: flags vary across versions; this is a common subset.
  nohup vllm serve "${BACKWARD_MODEL_PATH}" \
    --served-model-name "${BACKWARD_SERVED_NAME}" \
    --host "${HOST}" \
    --port "${BACKWARD_PORT}" \
    >"${LOG_FILE}" 2>&1 &
  echo $! > "${PID_FILE}"
else
  # Fallback: classic module entrypoint.
  nohup python -m vllm.entrypoints.openai.api_server \
    --model "${BACKWARD_MODEL_PATH}" \
    --served-model-name "${BACKWARD_SERVED_NAME}" \
    --host "${HOST}" \
    --port "${BACKWARD_PORT}" \
    >"${LOG_FILE}" 2>&1 &
  echo $! > "${PID_FILE}"
fi

echo "[run] waiting for vLLM to become ready..."
for i in $(seq 1 120); do
  if curl -fsS "${BACKWARD_BASE_URL}/models" >/dev/null 2>&1; then
    echo "[run] vLLM ready"
    break
  fi
  sleep 1
  if [[ "${i}" == "120" ]]; then
    echo "[run] vLLM did not become ready in time. Last logs:" >&2
    tail -n 80 "${LOG_FILE}" >&2 || true
    exit 1
  fi
done

echo "[run] running evaluation (backward=local vLLM, forward=remote)"
python scripts/eval_backward_agent_math500.py \
  --out_jsonl "${OUT_JSONL}" \
  --split "${SPLIT}" \
  --max_samples "${MAX_SAMPLES}" \
  --max_rounds "${MAX_ROUNDS}" \
  --trace_jsonl "outputs/math500_backward_traces.jsonl" \
  --backward_base_url "${BACKWARD_BASE_URL}" \
  --backward_model "${BACKWARD_SERVED_NAME}" \
  --backward_max_concurrency "${BACKWARD_MAX_CONCURRENCY}" \
  --forward_base_url "${FORWARD_BASE_URL}" \
  --forward_model "${FORWARD_MODEL}" \
  --forward_max_concurrency "${FORWARD_MAX_CONCURRENCY}" \
  ${FORWARD_API_KEY:+--forward_api_key "${FORWARD_API_KEY}"}

echo "[run] done. results jsonl: ${OUT_JSONL}"


