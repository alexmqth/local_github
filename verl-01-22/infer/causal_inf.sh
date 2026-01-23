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
export XDG_CACHE_HOME="$CACHE_ROOT"

ENV_NAME="verlrl"
CONDA_SH="/home/projects/hku-medai/larrypl/anaconda3/etc/profile.d/conda.sh"

if [ -z "${CONDA_PREFIX:-}" ]; then
  source "$CONDA_SH"
  conda activate "$ENV_NAME"
fi

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LD_LIBRARY_PATH:-}"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"

export HF_HOME="$CACHE_ROOT/huggingface" 
export VLLM_USE_FLASHINFER=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
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
1
cd /home/projects/hku-medai/larrypl/code/mq/causal-reasoning/verl-01-22/infer
python /home/projects/hku-medai/larrypl/code/mq/causal-reasoning/verl-01-22/infer/causal_inf.py \
  --policy_model_path /home/projects/hku-medai/larrypl/code/mq/causal-reasoning/checkpoints/qwen2_5_7b_ins_step_causal_verifier_h800_optimized/global_step_29_merged \
  --out_jsonl /home/projects/hku-medai/larrypl/code/mq/causal-reasoning/causal_outputs/qwen2_5_7b_math_all.jsonl \
  --judge_device cuda \
  --policy_device cuda \
  --max_examples 500