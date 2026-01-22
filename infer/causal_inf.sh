#!/usr/bin/env bash
set -euo pipefail

# =========================
# User config (EDIT THESE)
# =========================
WORKDIR="/home/projects/hku-medai/larrypl/code/mq/causal-reasoning/verl-01-22/infer"
PY="${WORKDIR}/causal_inf.py"

#POLICY_MODEL_PATH="/home/projects/hku-medai/larrypl/code/mq/causal-reasoning/checkpoints/qwen2_5_7b_ins_step_causal_verifier_h800_optimized/global_step_29_merged"
POLICY_MODEL_PATH="/home/projects/hku-medai/larrypl/code/mq/causal-reasoning/verl-01-22/rl_models/qwen2_5_7b_math_all_rerl"

# 如果你的 ckpt 目录里没有 tokenizer 文件，把下面改成你 base 模型 tokenizer 的本地目录
# 例如："/home/projects/.../Qwen2.5-7B-Instruct"
POLICY_TOKENIZER_PATH="${POLICY_TOKENIZER_PATH:-${POLICY_MODEL_PATH}}"

MODEL_NAME="qwen2_5_7b"

OUT_DIR="/home/projects/hku-medai/larrypl/code/mq/causal-reasoning/causal_outputs"

mkdir -p "${OUT_DIR}/${MODEL_NAME}"

OUT_BASE="${OUT_DIR}/${MODEL_NAME}/${MODEL_NAME}_math_all.jsonl"

DATASET_NAME="/home/projects/hku-medai/larrypl/code/mq/causal-reasoning/verl-01-22/datasets/MATH-500"
SPLIT="test"
MAX_EXAMPLES=500

POLICY_DEVICE="cuda"
# 想更快更稳：export JUDGE_DEVICE=cpu 再运行（避免 judge 抢 GPU）
JUDGE_DEVICE="${JUDGE_DEVICE:-cuda}"

GPUS="${GPUS:-8}"   # 8×H800

# Speed knobs
STEP_MAX_NEW_TOKENS="${STEP_MAX_NEW_TOKENS:-64}"
MAX_STEPS="${MAX_STEPS:-10}"
MAX_RETRIES="${MAX_RETRIES:-1}"
FORCE_FINISH_AFTER_STEPS="${FORCE_FINISH_AFTER_STEPS:-6}"

# Profiling
PROFILE="${PROFILE:-1}"         # 1=on, 0=off
PROFILE_EVERY="${PROFILE_EVERY:-25}"

# Optional conda env (你如果已经在 verlrl 里，可直接把这段注释掉)
ENV_NAME="verlrl"
CONDA_SH="/home/projects/hku-medai/larrypl/anaconda3/etc/profile.d/conda.sh"

# =========================
# Conda activate (avoid set -u conflict)
# =========================
if [[ -f "${CONDA_SH}" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${ENV_NAME}" || true
  set -u
fi

# =========================
# sanitize CRLF (VERY IMPORTANT)
# =========================
WORKDIR="$(printf "%s" "$WORKDIR" | tr -d '\r')"
PY="$(printf "%s" "$PY" | tr -d '\r')"
POLICY_MODEL_PATH="$(printf "%s" "$POLICY_MODEL_PATH" | tr -d '\r')"
POLICY_TOKENIZER_PATH="$(printf "%s" "$POLICY_TOKENIZER_PATH" | tr -d '\r')"
OUT_BASE="$(printf "%s" "$OUT_BASE" | tr -d '\r')"
DATASET_NAME="$(printf "%s" "$DATASET_NAME" | tr -d '\r')"
SPLIT="$(printf "%s" "$SPLIT" | tr -d '\r')"

cd "${WORKDIR}"

# =========================
# preflight checks
# =========================
[[ -f "$PY" ]] || { echo "[FATAL] PY not found: $PY"; exit 1; }
[[ -d "$POLICY_MODEL_PATH" ]] || { echo "[FATAL] POLICY_MODEL_PATH not found: $POLICY_MODEL_PATH"; exit 1; }
if [[ ! -d "$POLICY_TOKENIZER_PATH" ]]; then
  echo "[WARN] POLICY_TOKENIZER_PATH not a local dir: $POLICY_TOKENIZER_PATH"
  echo "       If ckpt has no tokenizer files, set POLICY_TOKENIZER_PATH to base model tokenizer directory."
fi

echo "[INFO] WORKDIR=${WORKDIR}"
echo "[INFO] GPUS=${GPUS} POLICY_DEVICE=${POLICY_DEVICE} JUDGE_DEVICE=${JUDGE_DEVICE}"
echo "[INFO] POLICY_MODEL_PATH=${POLICY_MODEL_PATH}"
echo "[INFO] POLICY_TOKENIZER_PATH=${POLICY_TOKENIZER_PATH}"
echo "[INFO] OUT_BASE=${OUT_BASE}"

rm -f "${OUT_BASE}.rank"*.jsonl "${OUT_BASE}.rank"*.log || true

# =========================
# Launch: 1 proc per GPU, shard by rank::world_size
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

# Wait
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
# Merge shards (sorted by idx)
# =========================
export OUT_BASE="${OUT_BASE}"
python - <<'PY'
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

n=0; c=0
with open(out_base, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip(): 
            continue
        r=json.loads(line)
        n+=1
        c+=1 if r.get("correct", False) else 0
print(f"total={n} correct={c} acc={c/n if n else 0:.4f}")
PY

echo "[DONE] Logs  : ${OUT_BASE}.rank*.log"
echo "[DONE] Output: ${OUT_BASE}"



