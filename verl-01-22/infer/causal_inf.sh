#!/bin/bash
#SBATCH --job-name=causal_inf          # Job name, replace with your job name
#SBATCH --error=causal_inf-%j.log
#SBATCH --output=causal_inf-%j.log
#SBATCH --ntasks=1                      # Number of tasks (usually set to 1 for single node jobs)
#SBATCH --cpus-per-task=16              # Number of CPU cores per task
#SBATCH --mem=64G                      # Total memory per node (128GB)
#SBATCH --gres=gpu:1
# nvidia-smi

export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
export HOME=/local/scratch/zqin30
export XDG_CACHE_HOME=/local/scratch/zqin30/.cache
source /local/scratch/zqin30/miniconda/etc/profile.d/conda.sh
conda activate verl

export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"

export HF_HOME=/local/scratch/zqin30/.cache/huggingface 

# # GSM8K 评测
# python causal_inf.py \
#     --policy_model_path path/to/your/checkpoint \
#     --dataset_name openai/gsm8k \
#     --dataset_config main \
#     --split test \
#     --dataset_type gsm8k \
#     --out_jsonl outputs/gsm8k_preds.jsonl \
#     --max_examples 50

# # MATH-500 评测（保持原有用法）
# python causal_inf.py \
#     --policy_model_path path/to/your/checkpoint \
#     --dataset_name HuggingFaceH4/MATH-500 \
#     --out_jsonl outputs/math500_preds.jsonl \
#     --max_examples 50

cd /local/scratch/zqin30/projects/repo/verl-agent-loop/verl/inference
python /local/scratch/zqin30/projects/repo/verl-agent-loop/verl/inference/causal_inf.py \
  --policy_model_path /local/scratch/zqin30/checkpoints/math/agent_loop/qwen2_5_7b_step_causal_verifier_v1 \
  --out_jsonl causal_outputs/qwen2_5_7b_step_causal_verifier_v1.jsonl \
  --judge_device cuda \
  --policy_device cuda \
  --max_examples 500