#!/usr/bin/env bash
set -euo pipefail

# =========================
# User config (CAN OVERRIDE BY ENV)
# =========================
WORKDIR="${WORKDIR:-/home/projects/hku-medai/larrypl/code/mq/causal-reasoning/verl-01-22/infer}"

# 兼容你可能的文件名：causal_inf.py / causal_inf(1).py / causal_inf (1).py
if [[ -n "${PY:-}" ]]; then
  PY="${PY}"
else
  if [[ -f "${WORKDIR}/causal_inf (1).py" ]]; then
    PY="${WORKDIR}/causal_inf (1).py"
  elif [[ -f "${WORKDIR}/causal_inf(1).py" ]]; then
    PY="${WORKDIR}/causal_inf(1).py"
  else
    PY="${WORKDIR}/causal_inf.py"
  fi
fi

# 模型路径 (Merged checkpoints)
POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-/home/projects/hku-medai/larrypl/code/mq/causal-reasoning/verl-01-22/rl_models/qwen2_5_7b_ins_gsm8k}"

# Tokenizer 路径 (如果 ckpt 目录缺 tokenizer 文件，请改为 base model 路径)
# 若你设置了 MODEL_DIR，则默认每个模型使用其自身目录；如需统一 tokenizer，设置 POLICY_TOKENIZER_PATH
POLICY_TOKENIZER_PATH="${POLICY_TOKENIZER_PATH:-}"

# 批量测试：给出模型文件夹，自动遍历其子目录
MODEL_DIR="${MODEL_DIR:-}"

MODEL_NAME="${MODEL_NAME:-$(basename "${POLICY_MODEL_PATH}")}"

OUT_DIR="${OUT_DIR:-/home/projects/hku-medai/larrypl/code/mq/causal-reasoning/verl-01-22/causal_outputs}"

# OUT_BASE 既可以传“完整 jsonl 路径”，也可以不传（自动生成）
# 注意：当使用 MODEL_DIR 批量测试时，会忽略 OUT_BASE，按每个模型自动生成
OUT_BASE="${OUT_BASE:-${OUT_DIR}/${MODEL_NAME}/${MODEL_NAME}_test.jsonl}"

# ✅ 你现在要测的 GSM8K parquet（可覆盖）
DATASET_NAME="${DATASET_NAME:-/home/projects/hku-medai/larrypl/code/mq/causal-reasoning/verl-01-22/datasets/GSM8K/gsm8k_test.parquet}"
SPLIT="${SPLIT:-test}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"

POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
# 生产环境建议设为 cpu，避免和 Policy 模型抢显存
JUDGE_DEVICE="${JUDGE_DEVICE:-cpu}"

GPUS="${GPUS:-2}"   # 每个模型使用的 GPU 数
TOTAL_GPUS="${TOTAL_GPUS:-8}"  # 总可用 GPU 数
GPUS_PER_MODEL="${GPUS_PER_MODEL:-1}"
MAX_PARALLEL_MODELS="${MAX_PARALLEL_MODELS:-2}"  # 0=自动(=TOTAL_GPUS/GPUS_PER_MODEL)

# Speed knobs (关键参数)
STEP_MAX_NEW_TOKENS="${STEP_MAX_NEW_TOKENS:-512}"
MAX_STEPS="${MAX_STEPS:-10}"
MAX_RETRIES="${MAX_RETRIES:-1}"
FORCE_FINISH_AFTER_STEPS="${FORCE_FINISH_AFTER_STEPS:-6}"

# Profiling
PROFILE="${PROFILE:-1}"         # 1=on, 0=off
PROFILE_EVERY="${PROFILE_EVERY:-25}"

# =========================
# CPU threading (per-process) for CPU judge
# =========================
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Conda env
ENV_NAME="${ENV_NAME:-verlrl}"
CONDA_SH="${CONDA_SH:-/home/projects/hku-medai/larrypl/anaconda3/etc/profile.d/conda.sh}"

# =========================
# Conda activate
# =========================
if [[ -f "${CONDA_SH}" ]]; then
  set +u
  source "${CONDA_SH}"
  conda activate "${ENV_NAME}" || true
  set -u
fi

# =========================
# Sanitize CRLF & Paths
# =========================
WORKDIR="$(printf "%s" "$WORKDIR" | tr -d '\r')"
PY="$(printf "%s" "$PY" | tr -d '\r')"
POLICY_MODEL_PATH="$(printf "%s" "$POLICY_MODEL_PATH" | tr -d '\r')"
POLICY_TOKENIZER_PATH="$(printf "%s" "$POLICY_TOKENIZER_PATH" | tr -d '\r')"
MODEL_DIR="$(printf "%s" "$MODEL_DIR" | tr -d '\r')"
OUT_BASE="$(printf "%s" "$OUT_BASE" | tr -d '\r')"
DATASET_NAME="$(printf "%s" "$DATASET_NAME" | tr -d '\r')"

cd "${WORKDIR}"

# =========================
# Positional arg support for MODEL_DIR
# =========================
if [[ -z "${MODEL_DIR}" ]] && [[ "${1:-}" != "" ]] && [[ -d "${1}" ]]; then
  MODEL_DIR="${1}"
fi

# =========================
# Preflight checks
# =========================
[[ -f "$PY" ]] || { echo "[FATAL] PY not found: $PY"; exit 1; }
run_one_model() {
  local model_path="$1"
  local model_name="${2:-$(basename "$model_path")}"
  local gpu_ids_csv="${3:-}"
  local tokenizer_path
  if [[ -n "${POLICY_TOKENIZER_PATH:-}" ]]; then
    tokenizer_path="${POLICY_TOKENIZER_PATH}"
  else
    tokenizer_path="${model_path}"
  fi

  local out_dir="${OUT_DIR}/${model_name}"
  mkdir -p "${out_dir}"
  local out_base
  if [[ -n "${MODEL_DIR:-}" ]]; then
    out_base="${out_dir}/${model_name}_test.jsonl"
  else
    out_base="${OUT_BASE}"
  fi

  echo "[INFO] WORKDIR=${WORKDIR}"
  echo "[INFO] GPUS=${GPUS} POLICY_DEVICE=${POLICY_DEVICE} JUDGE_DEVICE=${JUDGE_DEVICE}"
  if [[ -n "${gpu_ids_csv}" ]]; then
    echo "[INFO] GPU_IDS=${gpu_ids_csv}"
  fi
  echo "[INFO] POLICY_MODEL_PATH=${model_path}"
  echo "[INFO] DATASET_NAME=${DATASET_NAME}"
  echo "[INFO] OUT_BASE=${out_base}"
  echo "[INFO] OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"

  [[ -d "$model_path" ]] || { echo "[FATAL] POLICY_MODEL_PATH not found: $model_path"; return 1; }

  # 清理旧的 shard 文件
  rm -f "${out_base}.rank"*.jsonl "${out_base}.rank"*.log || true

  # =========================
  # Launch: 1 proc per GPU
  # =========================
  pids=()
  local gpu_ids=()
  if [[ -n "${gpu_ids_csv}" ]]; then
    IFS=',' read -r -a gpu_ids <<< "${gpu_ids_csv}"
  fi
  for ((r=0; r<GPUS; r++)); do
    echo "[INFO] Launch rank=${r}/${GPUS} on GPU=${r}"
    (
      if [[ -n "${gpu_ids_csv}" ]]; then
        export CUDA_VISIBLE_DEVICES="${gpu_ids[$r]}"
      else
        export CUDA_VISIBLE_DEVICES="${r}"
      fi
      export RANK="${r}"
      export WORLD_SIZE="${GPUS}"

      cmd=(python "${PY}"
        --policy_model_path "${model_path}"
        --policy_tokenizer_path "${tokenizer_path}"
        --dataset_name "${DATASET_NAME}"
        --dataset_type gsm8k
        --split "${SPLIT}"
        --max_examples "${MAX_EXAMPLES}"
        --policy_device "${POLICY_DEVICE}"
        --judge_device "${JUDGE_DEVICE}"
        --step_max_new_tokens "${STEP_MAX_NEW_TOKENS}"
        --max_steps "${MAX_STEPS}"
        --max_retries "${MAX_RETRIES}"
        --force_finish_after_steps "${FORCE_FINISH_AFTER_STEPS}"
        --profile_every "${PROFILE_EVERY}"
        --out_jsonl "${out_base}.rank${r}.jsonl"
      )

      if [[ "${PROFILE}" == "1" ]]; then
        cmd+=(--profile)
      fi

      echo "[INFO][rank ${r}] ${cmd[*]}"
      "${cmd[@]}"
    ) > "${out_base}.rank${r}.log" 2>&1 &

    pids+=("$!")
  done

  # =========================
  # Wait for completion
  # =========================
  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      fail=1
    fi
  done

  if [[ "${fail}" -ne 0 ]]; then
    echo "[ERROR] One or more ranks failed. Check logs: ${out_base}.rank*.log"
    return 1
  fi

  # =========================
  # Merge shards
  # =========================
  export OUT_BASE="${out_base}"
  python3 - <<'EOF'
import glob, json, os
out_base = os.environ["OUT_BASE"]
files = sorted(glob.glob(out_base + ".rank*.jsonl"))
rows = []
for fn in files:
    with open(fn, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
rows.sort(key=lambda r: int(r.get("idx", 0)))

with open(out_base, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[INFO] merged {len(rows)} rows -> {out_base}")

# Simple stat check
n=0; c=0
with open(out_base, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        r=json.loads(line)
        n+=1
        c+=1 if r.get("correct", False) else 0
print(f"total={n} correct={c} acc={c/n if n else 0:.4f}")
EOF

  echo "[DONE] Logs  : ${out_base}.rank*.log"
  echo "[DONE] Output: ${out_base}"
}

if [[ -n "${MODEL_DIR:-}" ]]; then
  if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "[FATAL] MODEL_DIR not found: ${MODEL_DIR}"
    exit 1
  fi
  shopt -s nullglob
  model_dirs=( "${MODEL_DIR}"/*/ )
  if [[ "${#model_dirs[@]}" -eq 0 ]]; then
    echo "[FATAL] MODEL_DIR has no subdirectories: ${MODEL_DIR}"
    exit 1
  fi
  # Build GPU groups
  gpu_pool=()
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a gpu_pool <<< "${CUDA_VISIBLE_DEVICES}"
  else
    for ((i=0; i<TOTAL_GPUS; i++)); do
      gpu_pool+=("${i}")
    done
  fi
  if (( ${#gpu_pool[@]} < TOTAL_GPUS )); then
    echo "[FATAL] TOTAL_GPUS=${TOTAL_GPUS} exceeds available CUDA_VISIBLE_DEVICES (${#gpu_pool[@]})"
    exit 1
  fi
  if (( TOTAL_GPUS % GPUS_PER_MODEL != 0 )); then
    echo "[FATAL] TOTAL_GPUS=${TOTAL_GPUS} must be divisible by GPUS_PER_MODEL=${GPUS_PER_MODEL}"
    exit 1
  fi
  if (( GPUS_PER_MODEL <= 0 )); then
    echo "[FATAL] GPUS_PER_MODEL must be > 0"
    exit 1
  fi
  max_parallel=$(( TOTAL_GPUS / GPUS_PER_MODEL ))
  if (( MAX_PARALLEL_MODELS > 0 )); then
    if (( MAX_PARALLEL_MODELS < max_parallel )); then
      max_parallel="${MAX_PARALLEL_MODELS}"
    fi
  fi
  echo "[INFO] TOTAL_GPUS=${TOTAL_GPUS} GPUS_PER_MODEL=${GPUS_PER_MODEL} MAX_PARALLEL=${max_parallel} (MAX_PARALLEL_MODELS=${MAX_PARALLEL_MODELS})"

  pids=()
  slot=0
  for d in "${model_dirs[@]}"; do
    d="${d%/}"
    gpu_ids_csv=""
    start=$(( slot * GPUS_PER_MODEL ))
    for ((i=0; i<GPUS_PER_MODEL; i++)); do
      idx=$(( start + i ))
      gpu_id="${gpu_pool[$idx]}"
      if [[ -z "${gpu_ids_csv}" ]]; then
        gpu_ids_csv="${gpu_id}"
      else
        gpu_ids_csv="${gpu_ids_csv},${gpu_id}"
      fi
    done

    echo "============================================================"
    echo "[MODEL] ${d}"
    echo "============================================================"
    (
      GPUS="${GPUS_PER_MODEL}"
      run_one_model "${d}" "" "${gpu_ids_csv}"
    ) &
    pids+=("$!")

    slot=$(( slot + 1 ))
    if (( slot >= max_parallel )); then
      if ! wait -n; then
        echo "[ERROR] One or more models failed."
      fi
      # shrink active list
      new_pids=()
      for pid in "${pids[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
          new_pids+=("${pid}")
        fi
      done
      pids=("${new_pids[@]}")
      slot=$(( ${#pids[@]} ))
    fi
  done
  # wait remaining
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      echo "[ERROR] Model failed."
    fi
  done
else
  if [[ -z "${POLICY_TOKENIZER_PATH:-}" ]]; then
    POLICY_TOKENIZER_PATH="${POLICY_MODEL_PATH}"
  fi
  run_one_model "${POLICY_MODEL_PATH}" "${MODEL_NAME}"
fi


