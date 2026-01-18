#!/usr/bin/env bash
#SBATCH --job-name=verl_step_causal_verifier
#SBATCH --error=verl_step_causal_verifier-%j.log
#SBATCH --output=verl_step_causal_verifier-%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=168G
#SBATCH --gres=gpu:2

set -xeuo pipefail

mkdir -p shell_logs

# ========================= env / cache =========================
# Prefer exporting these in your environment; do NOT hardcode secrets in scripts.
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"

export HOME=/home/projects/hku-medai/larryp
export XDG_CACHE_HOME=/home/projects/hku-medai/larrypl/cache
export HF_HOME=/home/projects/hku-medai/larrypl/cache/huggingface

# source /local/scratch/zqin30/miniconda/etc/profile.d/conda.sh
# conda activate agent_loop

# CUDA toolkit
export CUDA_HOME=/usr/local/cuda-12.2
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDACXX=/home/projects/hku-medai/larrypl/cache/conda/envs/qwen3_vl/bin/nvcc

nvcc --version || true
which nvcc || true

############ Clean JIT caches (optional) ############
rm -rf "${XDG_CACHE_HOME:-$HOME/.cache}/flashinfer" "$HOME/.cache/flashinfer" 2>/dev/null || true

# Clean ROCm visibility vars (some clusters inject these)
unset ROCR_VISIBLE_DEVICES || true
unset HIP_VISIBLE_DEVICES || true
unset HSA_VISIBLE_DEVICES || true

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# vLLM compatibility knobs (avoid FlashInfer JIT issues on some clusters)
export VLLM_USE_FLASHINFER=0

ulimit -n 65535 || true

# ========================= project paths =========================
cd /home/projects/hku-medai/larrypl/code/mq/causal-reasoning/verl/verl/experimental/agent_loop

# ========================= data / model =========================
# AgentLoop parquet/jsonl produced by:
#   python examples/data_preprocess/convert_pns_jsonl_to_agent_loop_jsonl.py --mode interactive ...
TRAIN_PARQUET=${TRAIN_PARQUET:-/home/projects/hku-medai/larrypl/code/mq/causal-reasoning/verl/verl/data/math/causal_output/pns_agentloop.parquet}
VAL_PARQUET=${VAL_PARQUET:-$TRAIN_PARQUET}

# Prefer local model path if exists, else HF hub path.
MODEL_PATH=${MODEL_PATH:-/home/projects/hku-medai/larrypl/code/mq/causal-reasoning/hf_models/Qwen/Qwen2.5-7B-Instruct}
if [ ! -d "$MODEL_PATH" ]; then
  MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
fi

# ========================= judge (PNS) settings =========================
# For remote judge (vLLM OpenAI server), set:
#   export JUDGE_BASE_URL="http://127.0.0.1:8000/v1"
#   export JUDGE_MODEL="your-judge-model-name"
#   export JUDGE_API_KEY="EMPTY"   # if not required
#
# For local judge (CPU), set in YAML:
#   judge_backend: local
#   judge_model_name_or_path: /path/to/local/model
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-}"
export JUDGE_MODEL="${JUDGE_MODEL:-}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-}"

# If your judge is local on CPU, AgentLoop workers do not need GPU.
# You can force that here:
export VERL_AGENT_LOOP_WORKER_NUM_GPUS="${VERL_AGENT_LOOP_WORKER_NUM_GPUS:-0}"

# ========================= wandb =========================
export WANDB_PROJECT="${WANDB_PROJECT:-verl_test}"
# export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_API_KEY="e4ba3be8a795947e787d1344e6898362060309b9"  

project_name=${project_name:-step_causal_verifier}
experiment_name=${experiment_name:-qwen2_5_7b_step_causal_verifier_v1}
default_local_dir=${default_local_dir:-/home/projects/hku-medai/larrypl/code/mq/causal-reasoning/checkpoints/${experiment_name}}

# ========================= algorithm knobs =========================
adv_estimator=${adv_estimator:-grpo}
train_batch_size=${train_batch_size:-4}
ppo_mini_batch_size=${ppo_mini_batch_size:-4}
n_resp_per_prompt=${n_resp_per_prompt:-8}
n_resp_per_prompt_val=${n_resp_per_prompt_val:-1}

max_turns=${max_turns:-2}

max_prompt_length=${max_prompt_length:-4096}
max_response_length=${max_response_length:-256}

infer_tp=${infer_tp:-2}
train_sp=${train_sp:-2}
offload=${offload:-False}

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
  actor_rollout_ref.actor.optim.lr=$actor_lr \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.model.use_remove_padding=true \
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
  actor_rollout_ref.rollout.agent.default_agent_loop=step_causal_verifier_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=/home/projects/hku-medai/larrypl/code/mq/causal-reasoning/verl/verl/experimental/agent_loop/step_causal_verifier.yaml \
  trainer.logger='["console","file"]' \
  trainer.project_name=$project_name \
  trainer.experiment_name=$experiment_name \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.default_local_dir="$default_local_dir" \
  trainer.save_freq=500 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  "$@"


