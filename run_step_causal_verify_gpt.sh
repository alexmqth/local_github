#!/usr/bin/env bash
set -euo pipefail
# 默认打开 xtrace（和你原脚本一致）；如果不想刷屏：export DEBUG=0
if [[ "${DEBUG:-1}" == "1" ]]; then
  set -x
fi

############################################
# 0) Fixed config for your machine
############################################
ENV_NAME="verlrl"
CONDA_SH="/home/projects/hku-medai/larrypl/anaconda3/etc/profile.d/conda.sh"

CACHE_ROOT="/home/projects/hku-medai/larrypl/cache"
PROJECT_ROOT="/home/projects/hku-medai/larrypl/code/mq/causal-reasoning/verl-01-22"
RUN_DIR="$PROJECT_ROOT/verl/experimental/agent_loop"

############################################
# 1) Train/Eval data + paths
############################################
TRAIN_PARQUET="${TRAIN_PARQUET:-$PROJECT_ROOT/data/pns_agentloop_gsm8k.parquet}"
VAL_PARQUET="${VAL_PARQUET:-$TRAIN_PARQUET}"

# verifier config
verifier_yaml="${verifier_yaml:-$RUN_DIR/step_causal_verifier.yaml}"
#verifier_yaml="${verifier_yaml:-$RUN_DIR/step_causal_verifier_gsm8k.yaml}"

# algo
adv_estimator="${adv_estimator:-grpo_token_level}"

# HF base models root (按你的目录结构来)
HF_MODELS_ROOT="${HF_MODELS_ROOT:-/home/projects/hku-medai/larrypl/code/mq/causal-reasoning/hf_models/Qwen}"

# checkpoints / merged models
CKPT_ROOT="${CKPT_ROOT:-$PROJECT_ROOT/checkpoints}"
RL_MODELS_ROOT="${RL_MODELS_ROOT:-$PROJECT_ROOT/rl_models}"

# evaluation
EVAL_DATASET="${EVAL_DATASET:-$PROJECT_ROOT/datasets/GSM8K/gsm8k_test.parquet}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-/home/projects/hku-medai/larrypl/code/mq/causal-reasoning/causal_outputs}"

# 兼容你可能的文件名（有的文件带 (1)）
EVAL_SH_DEFAULT_1="$PROJECT_ROOT/infer/causal_inf (1).sh"
EVAL_SH_DEFAULT_2="$PROJECT_ROOT/infer/causal_inf.sh"
if [[ -n "${EVAL_SH:-}" ]]; then
  EVAL_SH="$EVAL_SH"
elif [[ -f "$EVAL_SH_DEFAULT_1" ]]; then
  EVAL_SH="$EVAL_SH_DEFAULT_1"
else
  EVAL_SH="$EVAL_SH_DEFAULT_2"
fi

############################################
# 2) Queue / Wait settings
############################################
IDLE_CHECK_INTERVAL="${IDLE_CHECK_INTERVAL:-60}"   # 每隔多少秒检查一次
IDLE_GRACE_SECONDS="${IDLE_GRACE_SECONDS:-300}"    # 检测到空闲后再等 5 分钟
BETWEEN_MODELS_SLEEP="${BETWEEN_MODELS_SLEEP:-180}" # 每个模型完成后等 3 分钟再开始下一个

# 你想把哪些进程视为“训练还在跑”
BUSY_PATTERN="${BUSY_PATTERN:-verl\.trainer\.main_ppo|step_causal_verifier_agent|ray::|vllm}"

# 如果你希望结合 slurm squeue 检测（按 job name 过滤），可改这个正则
SQUEUE_BUSY_PATTERN="${SQUEUE_BUSY_PATTERN:-verl|vllm|ray|ppo|step_causal}"

# 如果你希望结合 nvidia-smi 检测 GPU 上的 python/ray/vllm 进程
CHECK_GPU_BUSY="${CHECK_GPU_BUSY:-1}"

############################################
# 3) Models queue
############################################
# 用法1：直接把模型名当作参数传入：
#   bash run_step_causal_verify_gpt(1).sh Qwen2.5-3B Qwen2.5-7B-Instruct
# 用法2：用环境变量（逗号分隔）：
#   export MODEL_QUEUE="Qwen2.5-3B,Qwen2.5-3B-Instruct,Qwen2.5-7B,Qwen3-8B"
#   bash run_step_causal_verify_gpt(1).sh
#
# 额外训练参数（Hydra overrides）支持用 -- 分隔：
#   bash run_step...sh Qwen2.5-3B Qwen2.5-7B -- trainer.total_epochs=2 trainer.save_freq=200
MODELS=()
EXTRA_TRAIN_ARGS=()
parse_extra=0
for a in "$@"; do
  if [[ "$a" == "--" ]]; then
    parse_extra=1
    continue
  fi
  if [[ "$parse_extra" == "0" ]]; then
    MODELS+=("$a")
  else
    EXTRA_TRAIN_ARGS+=("$a")
  fi
done

if [[ "${#MODELS[@]}" -eq 0 ]]; then
  if [[ -n "${MODEL_QUEUE:-}" ]]; then
    IFS=',' read -r -a MODELS <<< "${MODEL_QUEUE}"
  else
    # 默认队列（你给的示例）
    MODELS=("Qwen2.5-3B" "Qwen2.5-3B-Instruct" "Qwen2.5-7B" "Qwen3-8B")
  fi
fi

############################################
# 4) Env setup (和原脚本保持一致)
############################################
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  source "$CONDA_SH"
  conda activate "$ENV_NAME"
fi

export XDG_CACHE_HOME="$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export VLLM_CACHE_DIR="$CACHE_ROOT/vllm"
export FLASHINFER_JIT_DIR="$CACHE_ROOT/flashinfer"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$TORCH_EXTENSIONS_DIR" "$VLLM_CACHE_DIR" "$FLASHINFER_JIT_DIR"

export TMPDIR="/tmp/$USER/tmp"
export RAY_TMPDIR="/tmp/$USER/ray"
mkdir -p "$TMPDIR" "$RAY_TMPDIR"

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export NVCC="$CUDACXX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LIBRARY_PATH:-}"
export LDFLAGS="-L$CONDA_PREFIX/lib -L$CONDA_PREFIX/lib64"
export TORCH_CUDA_ARCH_LIST="9.0a"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export RAY_DEDUP_LOGS=0
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export VLLM_USE_FLASHINFER=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export HYDRA_FULL_ERROR=1

cd "$RUN_DIR"

############################################
# 5) Training hyperparams (原样保留)
############################################
n_resp_per_prompt=16
train_batch_size=64

ppo_mini_batch_size=64
micro_batch_size=32

max_turns=24
max_prompt_length=4096
max_response_length=64

infer_tp=1
train_sp=1
offload=False
actor_lr=1e-6

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 2 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 2 ))

############################################
# Helpers
############################################
trim() {
  local s="${1:-}"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf "%s" "$s"
}

slugify() {
  local s
  s="$(trim "${1:-}")"
  s="${s//\//_}"
  s="${s// /_}"
  s="${s//./_}"
  s="${s//:/_}"
  printf "%s" "$s"
}

is_busy_process() {
  pgrep -u "$USER" -f "$BUSY_PATTERN" >/dev/null 2>&1
}

is_busy_squeue() {
  if ! command -v squeue >/dev/null 2>&1; then
    return 1
  fi
  # 只看 RUNNING 状态；并且按 job name 过滤（避免把你其它无关 job 也算进来）
  if squeue -u "$USER" -h -t R -o "%j" 2>/dev/null | grep -Eq "$SQUEUE_BUSY_PATTERN"; then
    return 0
  fi
  return 1
}

is_busy_gpu() {
  if [[ "${CHECK_GPU_BUSY}" != "1" ]]; then
    return 1
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  # 查询 GPU 上的 pid；如果是你自己的 python/ray/vllm 进程，就认为忙
  local pid user comm
  while IFS=',' read -r pid comm; do
    pid="$(trim "$pid")"
    comm="$(trim "$comm")"
    [[ -z "$pid" ]] && continue
    user="$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    [[ "$user" != "$USER" ]] && continue
    if echo "$comm" | grep -Eqi 'python|ray|vllm'; then
      return 0
    fi
  done < <(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null || true)

  return 1
}

is_training_busy() {
  if is_busy_process; then return 0; fi
  if is_busy_squeue; then return 0; fi
  if is_busy_gpu; then return 0; fi
  return 1
}

wait_until_idle_then_grace() {
  # 关闭 xtrace，避免等待时刷屏
  local xtrace_on=0
  if [[ "${-}" == *x* ]]; then
    xtrace_on=1
    set +x
  fi

  local was_busy=0
  while is_training_busy; do
    was_busy=1
    echo "[WAIT] Detected training busy (pattern=${BUSY_PATTERN}). Re-check in ${IDLE_CHECK_INTERVAL}s..."
    # 打印一点点帮助排查（不会导致 set -e 退出）
    pgrep -u "$USER" -f -a "$BUSY_PATTERN" 2>/dev/null | head -n 5 || true
    sleep "${IDLE_CHECK_INTERVAL}"
  done

  if [[ "${was_busy}" == "1" ]]; then
    echo "[WAIT] Became idle. Grace sleep ${IDLE_GRACE_SECONDS}s then start..."
    sleep "${IDLE_GRACE_SECONDS}"
  else
    echo "[WAIT] Already idle. Start immediately."
  fi

  if [[ "$xtrace_on" == "1" ]]; then
    set -x
  fi
}

run_one_model() {
  local model_raw="${1:-}"
  local model_name
  model_name="$(trim "$model_raw")"
  if [[ -z "$model_name" ]]; then
    echo "[SKIP] empty model name"
    return 0
  fi

  local ts tag
  ts="$(date '+%Y-%m-%d-%H')"
  tag="$(slugify "$model_name")"

  # 你要求：project_name / experiment_name 都用 “模型名 + 当前时间(年-月-日-小时)”
  local project_name="${tag}-${ts}"
  local experiment_name="${tag}-${ts}"

  local MODEL_PATH="${HF_MODELS_ROOT}/${model_name}"
  if [[ -n "${MODEL_PATH_MAP:-}" ]]; then
    # 可选：允许你用映射覆盖模型路径，例如：
    # export MODEL_PATH_MAP="Qwen3-8B=/path/to/Qwen3-8B;Qwen2.5-7B=/path/to/Qwen2.5-7B"
    local kv k v
    local -a kvs=()
    IFS=';' read -r -a kvs <<< "${MODEL_PATH_MAP}"
    for kv in "${kvs[@]}"; do
      k="${kv%%=*}"; v="${kv#*=}"
      if [[ "$(trim "$k")" == "$model_name" ]]; then
        MODEL_PATH="$(trim "$v")"
        break
      fi
    done
  fi

  if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[ERROR] Base model not found: ${MODEL_PATH}"
    return 1
  fi

  local default_local_dir="${CKPT_ROOT}/${experiment_name}"
  mkdir -p "$default_local_dir"

  echo "============================================================"
  echo "[TRAIN START] $(date '+%F %T')"
  echo "model_name=${model_name}"
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "project_name=${project_name}"
  echo "experiment_name=${experiment_name}"
  echo "default_local_dir=${default_local_dir}"
  echo "============================================================"

  ############################################
  # 5.1) Train (原命令，仅把变量改成当前 model)
  ############################################
  python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator="$adv_estimator" \
    data.train_files="['$TRAIN_PARQUET']" \
    data.val_files="['$VAL_PARQUET']" \
    data.prompt_key=prompt \
    data.return_raw_chat=true \
    data.train_batch_size="$train_batch_size" \
    data.max_prompt_length="$max_prompt_length" \
    data.max_response_length="$max_response_length" \
    data.filter_overlong_prompts=true \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr="$actor_lr" \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.ppo_mini_batch_size="$ppo_mini_batch_size" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$micro_batch_size" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$actor_max_token_len_per_gpu" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="$train_sp" \
    actor_rollout_ref.actor.fsdp_config.param_offload="$offload" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="$offload" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$log_prob_max_token_len_per_gpu" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$infer_tp" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.prompt_length="$max_prompt_length" \
    actor_rollout_ref.rollout.response_length="$max_response_length" \
    actor_rollout_ref.rollout.n="$n_resp_per_prompt" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns="$max_turns" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$max_turns" \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.rollout.agent.default_agent_loop=step_causal_verifier_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${verifier_yaml}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$micro_batch_size" \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$project_name" \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_local_dir="$default_local_dir" \
    trainer.save_freq=500 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    "${EXTRA_TRAIN_ARGS[@]}"

  echo "=== Training finished. Starting Model Merge... ==="

  ############################################
  # 5.2) Merge weights (原逻辑)
  ############################################
  local TARGET_DIR="${RL_MODELS_ROOT}/${experiment_name}"

  local LATEST_CHECKPOINT
  LATEST_CHECKPOINT=$(ls -d "${default_local_dir}"/global_step_* 2>/dev/null | sort -V | tail -n 1)

  if [[ -z "${LATEST_CHECKPOINT:-}" ]]; then
    echo "[Error] No checkpoints found in ${default_local_dir}. Merge skipped."
    return 1
  fi

  local SOURCE_ACTOR_DIR="${LATEST_CHECKPOINT}/actor"
  if [[ ! -d "$SOURCE_ACTOR_DIR" ]]; then
    echo "[Error] Actor directory not found at $SOURCE_ACTOR_DIR. Merge skipped."
    return 1
  fi

  echo "Source Checkpoint: $SOURCE_ACTOR_DIR"
  echo "Target Directory:  $TARGET_DIR"

  mkdir -p "$TARGET_DIR"
  python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$SOURCE_ACTOR_DIR" \
    --target_dir "$TARGET_DIR"

  echo "=== Model merged successfully saved to: $TARGET_DIR ==="

  ############################################
  # 5.3) Eval (GSM8K parquet) + log.txt
  ############################################
  local eval_dir="${EVAL_OUT_DIR}/${experiment_name}"
  mkdir -p "${eval_dir}"
  local log_txt="${eval_dir}/log.txt"
  local out_jsonl="${eval_dir}/${experiment_name}_gsm8k_test.jsonl"

  {
    echo "============================================================"
    echo "[EVAL START] $(date '+%F %T')"
    echo "model_name=${model_name}"
    echo "project_name=${project_name}"
    echo "experiment_name=${experiment_name}"
    echo "merged_model_dir=${TARGET_DIR}"
    echo "tokenizer_base_dir=${MODEL_PATH}"
    echo "dataset=${EVAL_DATASET}"
    echo "out_jsonl=${out_jsonl}"
    echo "============================================================"
  } | tee -a "${log_txt}"

  # 用环境变量覆盖 eval 脚本里的默认值（需要配合我给你的 causal_inf(1).sh 改动）
  POLICY_MODEL_PATH="${TARGET_DIR}" \
  POLICY_TOKENIZER_PATH="${MODEL_PATH}" \
  MODEL_NAME="${experiment_name}" \
  OUT_DIR="${EVAL_OUT_DIR}" \
  OUT_BASE="${out_jsonl}" \
  DATASET_NAME="${EVAL_DATASET}" \
  SPLIT="test" \
  MAX_EXAMPLES="${EVAL_MAX_EXAMPLES:-0}" \
  GPUS="${EVAL_GPUS:-8}" \
  JUDGE_DEVICE="${JUDGE_DEVICE:-cpu}" \
  # bash "${EVAL_SH}" 2>&1 | tee -a "${log_txt}"

  echo "[EVAL DONE] $(date '+%F %T')" | tee -a "${log_txt}"

  ############################################
  # 5.4) Cooldown
  ############################################
  echo "[SLEEP] ${BETWEEN_MODELS_SLEEP}s before next model..."
  sleep "${BETWEEN_MODELS_SLEEP}"
}

############################################
# Main loop
############################################
echo "[QUEUE] Models (${#MODELS[@]}): ${MODELS[*]}"
echo "[INFO] EVAL_SH=${EVAL_SH}"
echo "[INFO] EVAL_DATASET=${EVAL_DATASET}"
echo "[INFO] RL_MODELS_ROOT=${RL_MODELS_ROOT}"
echo "[INFO] CKPT_ROOT=${CKPT_ROOT}"

FAIL_FAST="${FAIL_FAST:-0}"  # 1=某个模型失败就直接退出；0=继续跑后续

for m in "${MODELS[@]}"; do
  wait_until_idle_then_grace
  if ! run_one_model "$m"; then
    echo "[ERROR] Model failed: $m"
    if [[ "$FAIL_FAST" == "1" ]]; then
      exit 1
    fi
  fi
done

echo "[DONE] All queued models finished."

