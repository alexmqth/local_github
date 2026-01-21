#!/usr/bin/env bash
set -xeuo pipefail

############################################
# 0) Fixed config for your machine
############################################
ENV_NAME="verlrl"
CONDA_SH="/home/projects/hku-medai/larrypl/anaconda3/etc/profile.d/conda.sh"

CACHE_ROOT="/home/projects/hku-medai/larrypl/cache"
PROJECT_ROOT="/home/projects/hku-medai/larrypl/code/mq/causal-reasoning/verl"
RUN_DIR="$PROJECT_ROOT/verl/experimental/agent_loop"

TRAIN_PARQUET="${TRAIN_PARQUET:-$PROJECT_ROOT/verl/data/math/causal_output/pns_agentloop_gsm8k.parquet}"
VAL_PARQUET="${VAL_PARQUET:-$TRAIN_PARQUET}"
MODEL_PATH="${MODEL_PATH:-/home/projects/hku-medai/larrypl/code/mq/causal-reasoning/hf_models/Qwen/Qwen2.5-3B-Instruct}"

############################################
# 1) Activate conda env (auto)
############################################
if [ -z "${CONDA_PREFIX:-}" ]; then
  source "$CONDA_SH"
  conda activate "$ENV_NAME"
fi

echo "CONDA_PREFIX=$CONDA_PREFIX"
which python
python -V
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"

############################################
# 2) Cache dirs (do NOT change HOME)
############################################
export XDG_CACHE_HOME="$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export VLLM_CACHE_DIR="$CACHE_ROOT/vllm"
export FLASHINFER_JIT_DIR="$CACHE_ROOT/flashinfer"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$TORCH_EXTENSIONS_DIR" "$VLLM_CACHE_DIR" "$FLASHINFER_JIT_DIR"

# IMPORTANT: Ray socket path must be short (<107 chars). Use /tmp for ray+tmp only.
export TMPDIR="/tmp/$USER/tmp"
export RAY_TMPDIR="/tmp/$USER/ray"
mkdir -p "$TMPDIR" "$RAY_TMPDIR"

############################################
# 3) CUDA / nvcc / link env (conda)
############################################
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export NVCC="$CUDACXX"
[ -x "$CUDACXX" ] || { echo "[FATAL] nvcc not found at $CUDACXX"; exit 1; }

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LIBRARY_PATH:-}"
export LDFLAGS="-L$CONDA_PREFIX/lib -L$CONDA_PREFIX/lib64"

# H800 / Hopper (sm90a)
export TORCH_CUDA_ARCH_LIST="9.0a"

############################################
# 4) Runtime knobs (stable defaults)
############################################
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

export RAY_DEDUP_LOGS=0
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export WANDB_MODE=disabled
export WANDB_DISABLED=true

# vLLM knobs
# 强烈建议安装 flashinfer 以获得 H800 最大性能。如未安装则保持为 0
export VLLM_USE_FLASHINFER=0 

# CPU thread caps
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

############################################
# 5) Run (OPTIMIZED for 8x H800)
############################################
cd "$RUN_DIR"

adv_estimator="${adv_estimator:-grpo}"

# === [关键优化 1] Batch Size 大幅提升 ===
# 解释: H800 显存极大，原来 Batch=4 太小导致 GPU 空转。
# Global Batch Size (总采样数/步)
train_batch_size="${train_batch_size:-64}" 
# PPO Update Mini Batch
ppo_mini_batch_size="${ppo_mini_batch_size:-64}"
# 单卡计算时的 Micro Batch (填满显存的关键)
micro_batch_size="${micro_batch_size:-16}"

# === [关键优化 2] 增加并发采样数 ===
# 解释: 增加 n 可以让 Rollout 阶段并行度更高
n_resp_per_prompt="${n_resp_per_prompt:-64}"
n_resp_per_prompt_val="${n_resp_per_prompt_val:-1}"

# === [关键优化 3] 解决截断问题 ===
# 解释: 256 对于 CoT 根本不够。改为 2048 保证逻辑完整，同时增加计算密度。
max_turns="${max_turns:-2}"
max_prompt_length="${max_prompt_length:-4096}"
max_response_length="${max_response_length:-2048}" 

# === [关键优化 4] 并行策略调整 ===
# 解释: 7B 模型在 80G 卡上不需要 TP/SP，拆分反而增加通信开销。
# 改为 1 让每张卡独立跑，速度最快。
infer_tp="${infer_tp:-1}"
train_sp="${train_sp:-1}"
offload="${offload:-False}"
actor_lr="${actor_lr:-1e-6}"

# 自动计算 Buffer
actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 2 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 2 ))

project_name="${project_name:-step_causal_verifier}"
# 建议修改 experiment_name 避免和旧的垃圾 checkpoint 混淆
experiment_name="${experiment_name:-qwen2_5_3b_ins_step_causal_verifier_h800_optimized}"
default_local_dir="${default_local_dir:-/home/projects/hku-medai/larrypl/code/mq/causal-reasoning/checkpoints/${experiment_name}}"
mkdir -p "$default_local_dir"

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
  actor_rollout_ref.rollout.val_kwargs.n="$n_resp_per_prompt_val" \
  actor_rollout_ref.rollout.multi_turn.max_user_turns="$max_turns" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$max_turns" \
  actor_rollout_ref.rollout.agent.num_workers=32 \
  actor_rollout_ref.rollout.agent.default_agent_loop=step_causal_verifier_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$RUN_DIR/step_causal_verifier.yaml" \
  trainer.logger='["console","file"]' \
  trainer.project_name="$project_name" \
  trainer.experiment_name="$experiment_name" \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.default_local_dir="$default_local_dir" \
  trainer.save_freq=500 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  "$@"
