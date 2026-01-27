#!/usr/bin/env bash
#SBATCH --job-name=verl_rffg_step_verify          # Job name
#SBATCH --error=shell_logs/verl_rffg_step_verify-%j.log
#SBATCH --output=shell_logs/verl_rffg_step_verify-%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:2

set -xeuo pipefail

mkdir -p shell_logs

# ========================= env / cache =========================
export HF_TOKEN="${HF_TOKEN:-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}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"

export HOME=/local/scratch/zqin30
export XDG_CACHE_HOME=/local/scratch/zqin30/.cache

source /local/scratch/zqin30/miniconda3/etc/profile.d/conda.sh
conda activate verl

# CUDA toolkit
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDACXX="$CUDA_HOME/bin/nvcc"

nvcc --version || true
which nvcc || true

############ Clean JIT caches (important) ############
rm -rf "${XDG_CACHE_HOME:-$HOME/.cache}/flashinfer" "$HOME/.cache/flashinfer" 2>/dev/null || true

# Clean ROCm visibility vars (some clusters inject these)
unset ROCR_VISIBLE_DEVICES || true
unset HIP_VISIBLE_DEVICES || true
unset HSA_VISIBLE_DEVICES || true

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-}"
echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-}"

export HF_HOME=/local/scratch/zqin30/.cache/huggingface

# If you don't need OpenAI, leave these unset. (Don't hardcode keys in scripts.)
# export OPENAI_BASE_URL="https://api.openai.com/v1"
# export OPENAI_API_KEY="sk-..."

# vLLM compatibility knobs (avoid FlashInfer JIT issues on some clusters)
export VLLM_USE_FLASHINFER=0

ulimit -n 65535 || true

# ========================= project paths =========================
cd /local/scratch/zqin30/projects/repo/verl

# ========================= data / model =========================
# Your cleaned parquet produced by dedupe_subproblems_with_llm.py
TRAIN_PARQUET=${TRAIN_PARQUET:-/local/scratch/zqin30/projects/repo/verl/examples/data_preprocess/math_rffg_agent/math_train_solutiongt_fixed.parquet}
VAL_PARQUET=${VAL_PARQUET:-$TRAIN_PARQUET}

# Prefer local model path if exists, else HF hub path.
MODEL_PATH=${MODEL_PATH:-/local/scratch/zqin30/models/Qwen/Qwen2.5-3B-Instruct}
if [ ! -d "$MODEL_PATH" ]; then
  MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
fi

# ========================= wandb =========================
export WANDB_PROJECT="${WANDB_PROJECT:-verl_test}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

project_name=${project_name:-rffg_step_verify}
experiment_name=${experiment_name:-qwen2_5_3b_step_verify_2g}
default_local_dir=${default_local_dir:-/local/scratch/zqin30/checkpoints/${experiment_name}}

# ========================= algorithm knobs =========================
# Keep these conservative to start; tune later.
adv_estimator=${adv_estimator:-ppo}
train_batch_size=${train_batch_size:-64}
ppo_mini_batch_size=${ppo_mini_batch_size:-8}
n_resp_per_prompt=${n_resp_per_prompt:-2}
n_resp_per_prompt_val=${n_resp_per_prompt_val:-1}

# Multi-turn limits (should be >= max subproblems length; set a safe upper bound)
max_turns=${max_turns:-16}

max_prompt_length=${max_prompt_length:-1024}
max_response_length=${max_response_length:-2048}

# vLLM tensor parallel for inference (2 GPUs -> TP=2 is typical)
infer_tp=${infer_tp:-2}

# Training parallelism (2 GPUs -> sequence parallel size 2 is typical)
train_sp=${train_sp:-2}
offload=${offload:-true}

actor_lr=${actor_lr:-1e-6}

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 2 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 2 ))

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=$adv_estimator \
  data.train_files="['$TRAIN_PARQUET']" \
  data.val_files="['$VAL_PARQUET']" \
  data.prompt_key=prompt \
  data.return_raw_chat=true \
  data.train_batch_size=$train_batch_size \
  data.max_prompt_length=$max_prompt_length \
  data.max_response_length=$max_response_length \
  data.filter_overlong_prompts=true \
  data.truncation='error' \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.optim.lr=$actor_lr \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
  actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
  actor_rollout_ref.rollout.prompt_length=$max_prompt_length \
  actor_rollout_ref.rollout.response_length=$max_response_length \
  actor_rollout_ref.rollout.n=$n_resp_per_prompt \
  actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
  actor_rollout_ref.rollout.agent.num_workers=2 \
  actor_rollout_ref.rollout.agent.default_agent_loop=step_verify_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=recipe/langgraph_agent/example/step_verify_agent.yaml \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=$project_name \
  trainer.experiment_name=$experiment_name \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.default_local_dir="$default_local_dir" \
  trainer.save_freq=-1 \
  trainer.total_epochs=1 \
  "$@"


