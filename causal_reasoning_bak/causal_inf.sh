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
POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-/home/projects/hku-medai/larrypl/code/mq/causal-reasoning/verl-01-22/rl_models/qwen2_5_7b_ins_gsm8k-01-26-0}"

# Tokenizer 路径 (如果 ckpt 目录缺 tokenizer 文件，请改为 base model 路径)
POLICY_TOKENIZER_PATH="${POLICY_TOKENIZER_PATH:-${POLICY_MODEL_PATH}}"

MODEL_NAME="${MODEL_NAME:-$(basename "${POLICY_MODEL_PATH}")}"

OUT_DIR="${OUT_DIR:-/home/projects/hku-medai/larrypl/code/mq/causal-reasoning/causal_outputs}"
mkdir -p "${OUT_DIR}/${MODEL_NAME}"

# OUT_BASE 既可以传“完整 jsonl 路径”，也可以不传（自动生成）
OUT_BASE="${OUT_BASE:-${OUT_DIR}/${MODEL_NAME}/${MODEL_NAME}_gsm8k_test.jsonl}"

# ✅ 你现在要测的 GSM8K parquet（可覆盖）
DATASET_NAME="${DATASET_NAME:-/home/projects/hku-medai/larrypl/code/mq/causal-reasoning/verl-01-22/datasets/GSM8K/gsm8k_test.parquet}"
SPLIT="${SPLIT:-test}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"

POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
# 生产环境建议设为 cpu，避免和 Policy 模型抢显存
JUDGE_DEVICE="${JUDGE_DEVICE:-cpu}"

GPUS="${GPUS:-8}"   # 8xH800

# Speed knobs (关键参数)
STEP_MAX_NEW_TOKENS="${STEP_MAX_NEW_TOKENS:-256}"
MAX_STEPS="${MAX_STEPS:-10}"
MAX_RETRIES="${MAX_RETRIES:-1}"
FORCE_FINISH_AFTER_STEPS="${FORCE_FINISH_AFTER_STEPS:-6}"

# Profiling
PROFILE="${PROFILE:-1}"         # 1=on, 0=off
PROFILE_EVERY="${PROFILE_EVERY:-25}"

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
OUT_BASE="$(printf "%s" "$OUT_BASE" | tr -d '\r')"
DATASET_NAME="$(printf "%s" "$DATASET_NAME" | tr -d '\r')"

cd "${WORKDIR}"

# =========================
# Preflight checks
# =========================
[[ -f "$PY" ]] || { echo "[FATAL] PY not found: $PY"; exit 1; }
[[ -d "$POLICY_MODEL_PATH" ]] || { echo "[FATAL] POLICY_MODEL_PATH not found: $POLICY_MODEL_PATH"; exit 1; }

echo "[INFO] WORKDIR=${WORKDIR}"
echo "[INFO] GPUS=${GPUS} POLICY_DEVICE=${POLICY_DEVICE} JUDGE_DEVICE=${JUDGE_DEVICE}"
echo "[INFO] POLICY_MODEL_PATH=${POLICY_MODEL_PATH}"
echo "[INFO] DATASET_NAME=${DATASET_NAME}"
echo "[INFO] OUT_BASE=${OUT_BASE}"

# 清理旧的 shard 文件
rm -f "${OUT_BASE}.rank"*.jsonl "${OUT_BASE}.rank"*.log || true

# =========================
# Launch: 1 proc per GPU
# =========================
pids=()
for ((r=0; r<GPUS; r++)); do
  echo "[INFO] Launch rank=${r}/${GPUS} on GPU=${r}"
  (
    export CUDA_VISIBLE_DEVICES="${r}"
    export RANK="${r}"
    export WORLD_SIZE="${GPUS}"

    cmd=(python "${PY}"
      --policy_model_path "${POLICY_MODEL_PATH}"
      --policy_tokenizer_path "${POLICY_TOKENIZER_PATH}"
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
      --out_jsonl "${OUT_BASE}.rank${r}.jsonl"
    )

    if [[ "${PROFILE}" == "1" ]]; then
      cmd+=(--profile)
    fi

    echo "[INFO][rank ${r}] ${cmd[*]}"
    "${cmd[@]}"
  ) > "${OUT_BASE}.rank${r}.log" 2>&1 &

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
  echo "[ERROR] One or more ranks failed. Check logs: ${OUT_BASE}.rank*.log"
  exit 1
fi

# =========================
# Merge shards
# =========================
export OUT_BASE="${OUT_BASE}"
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

echo "[DONE] Logs  : ${OUT_BASE}.rank*.log"
echo "[DONE] Output: ${OUT_BASE}"

