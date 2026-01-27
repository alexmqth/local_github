#!/usr/bin/env bash
set -euo pipefail

# =========================
# User config (EDIT THESE)
# =========================
WORKDIR="/home/projects/hku-medai/larrypl/code/mq/causal-reasoning/verl-01-22/infer"
PY="${WORKDIR}/causal_inf.py"

# 模型路径 (Merged checkpoints)
POLICY_MODEL_PATH="/home/projects/hku-medai/larrypl/code/mq/causal-reasoning/verl-01-22/rl_models/qwen2_5_7b_ins_math_all"

# Tokenizer 路径 (如果 ckpt 目录缺 tokenizer 文件，请改为 base model 路径)
POLICY_TOKENIZER_PATH="${POLICY_TOKENIZER_PATH:-${POLICY_MODEL_PATH}}"

MODEL_NAME="qwen2_5_7b_ins"

OUT_DIR="/home/projects/hku-medai/larrypl/code/mq/causal-reasoning/causal_outputs"
mkdir -p "${OUT_DIR}/${MODEL_NAME}"
OUT_BASE="${OUT_DIR}/${MODEL_NAME}/${MODEL_NAME}_math_all.jsonl"

DATASET_NAME="/home/projects/hku-medai/larrypl/code/mq/causal-reasoning/verl-01-22/datasets/MATH-500"
SPLIT="test"
MAX_EXAMPLES=500

POLICY_DEVICE="cuda"
# 生产环境建议设为 cpu，避免和 Policy 模型抢显存；或者设为 cuda 也可以
JUDGE_DEVICE="${JUDGE_DEVICE:-cuda}"

GPUS="${GPUS:-8}"   # 8xH800

# Speed knobs (关键参数)
# 256 是我们修复低分问题的关键 (之前是 64)
STEP_MAX_NEW_TOKENS="${STEP_MAX_NEW_TOKENS:-256}"
MAX_STEPS="${MAX_STEPS:-10}"
MAX_RETRIES="${MAX_RETRIES:-1}"
FORCE_FINISH_AFTER_STEPS="${FORCE_FINISH_AFTER_STEPS:-6}"

# Profiling
PROFILE="${PROFILE:-1}"         # 1=on, 0=off
PROFILE_EVERY="${PROFILE_EVERY:-25}"

# Conda env
ENV_NAME="verlrl"
CONDA_SH="/home/projects/hku-medai/larrypl/anaconda3/etc/profile.d/conda.sh"

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
# 移除路径中可能存在的 Windows 换行符
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
# Wait for completion (修复了这里缺失的逻辑)
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
        if not line.strip(): continue
        r=json.loads(line)
        n+=1
        c+=1 if r.get("correct", False) else 0
print(f"total={n} correct={c} acc={c/n if n else 0:.4f}")
EOF

echo "[DONE] Logs  : ${OUT_BASE}.rank*.log"
echo "[DONE] Output: ${OUT_BASE}"
