#!/bin/bash
#SBATCH --job-name=get_parquetlora          # Job name, replace with your job name
#SBATCH --error=get_parquet-%j.log
#SBATCH --output=get_parquet-%j.log
#SBATCH --ntasks=1                      # Number of tasks (usually set to 1 for single node jobs)
#SBATCH --cpus-per-task=16              # Number of CPU cores per task
#SBATCH --mem=64G                      # Total memory per node (128GB)
#SBATCH --gres=gpu:1
# nvidia-smi
set -euo pipefail


export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
export HOME=/local/scratch/zqin30
export XDG_CACHE_HOME=/local/scratch/zqin30/.cache
source /local/scratch/zqin30/miniconda3/etc/profile.d/conda.sh

# ---- CUDA toolkit selection ----
# Use a CUDA toolkit that supports Hopper (compute_90 / compute_90a) if you are on H100 nodes.
# This also ensures nvcc is taken from the expected toolkit path.
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# ---- Create/activate a dedicated vLLM env (recommended: Python 3.10/3.11; avoid 3.12 for many ML stacks) ----
ENV_NAME="verl_vllm"
PY_VER="3.10"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] Creating conda env: ${ENV_NAME} (python=${PY_VER})"
  conda create -y -n "${ENV_NAME}" "python=${PY_VER}" pip

  # Install PyTorch CUDA 12.4 wheels + vLLM + deps used by our scripts.
  # Notes:
  # - If your cluster uses a different CUDA driver/toolkit combo, adjust cu124 accordingly.
  # - You may want to pin versions to your cluster policy.
  conda run -n "${ENV_NAME}" python -m pip install -U pip setuptools wheel
  conda run -n "${ENV_NAME}" python -m pip install \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch torchvision torchaudio
  conda run -n "${ENV_NAME}" python -m pip install \
    vllm \
    transformers \
    accelerate \
    datasets \
    requests
fi

conda activate "${ENV_NAME}"

export HF_HOME=/local/scratch/zqin30/.cache/huggingface 
cd /local/scratch/zqin30/projects/repo/verl/examples/data_preprocess

# python mathprm.py
# python build_rffg_prm_agentloop.py

# ---------- Start a local vLLM OpenAI-compatible server in THIS Slurm job ----------
# NOTE:
# - This is required because dedupe_subproblems_with_llm.py calls /v1/chat/completions via HTTP.
# - We start the server in background and poll readiness before running the dedupe script.
PORT=8000
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
SERVED_NAME="local-qwen"

# Workaround: vLLM v0.11 may enable FlashInfer sampling by default.
# On some clusters, the installed nvcc does NOT support Hopper arch 'compute_90a',
# which makes FlashInfer JIT compilation fail (nvcc fatal: Unsupported gpu architecture 'compute_90a').
# Disable FlashInfer so vLLM falls back to non-FlashInfer sampling ops.
export VLLM_USE_FLASHINFER=0
# Optional: if your environment defaults to V1 engine and it's problematic, you can also try:
# export VLLM_USE_V1=0

python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port ${PORT} \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --trust-remote-code \
  > vllm_server.log 2>&1 &

VLLM_PID=$!
echo "vLLM server pid=${VLLM_PID}"

# Fail fast if the process already died (port in use / import error / etc.)
sleep 2
if ! kill -0 ${VLLM_PID} 2>/dev/null; then
  echo "[ERROR] vLLM process exited immediately. Showing last 200 lines of vllm_server.log:"
  tail -n 200 vllm_server.log || true
  exit 1
fi

# Wait for readiness (model load can take long on first run). Increase if needed.
VLLM_WAIT_SEC=${VLLM_WAIT_SEC:-900}
python - <<PY
import time, requests, os, sys
port = int(os.environ.get("PORT","8000"))
wait_s = int(os.environ.get("VLLM_WAIT_SEC","900"))
base = f"http://127.0.0.1:{port}"
health = base + "/health"
models = base + "/v1/models"
for t in range(wait_s):
    try:
        r = requests.get(health, timeout=2)
        if r.status_code == 200:
            # extra: confirm /v1/models also works
            r2 = requests.get(models, timeout=2)
            if r2.status_code == 200:
                print("vLLM ready:", r2.json())
                sys.exit(0)
    except Exception:
        pass
    time.sleep(1)
raise SystemExit(f"vLLM not ready after {wait_s}s (health/models not responding)")
PY

python -u dedupe_subproblems_with_llm.py \
  --in_parquet /local/scratch/zqin30/projects/repo/verl/examples/data_preprocess/math_rffg_agent/math_train_solutiongt.parquet \
  --out_parquet /local/scratch/zqin30/projects/repo/verl/examples/data_preprocess/math_rffg_agent/math_train_solutiongt_fixed.parquet \
  --base_url http://127.0.0.1:8000/v1 \
  --model ${SERVED_NAME} \
  --api_key EMPTY \
  --fail_on_error \
  --max_tokens 128 \
  --max_chars_each 400 \
  --sleep_s 0.0

# Cleanup
kill ${VLLM_PID} 2>/dev/null || true