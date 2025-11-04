#!/bin/bash
#SBATCH --job-name=llama_gpu
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/llama_gpu_%j.out
#SBATCH --error=logs/llama_gpu_%j.err

module purge
module load gcc/11.4.0
module load cuda/12.2.1
export LD_LIBRARY_PATH=/software/spackages_v0_21_prod/apps/linux-ubuntu22.04-zen2/gcc-13.2.0/cuda-12.2.1-rqt4vr3vbmq5edgzd5glgq73htbneaw6/lib64:/software/spackages_v0_21_prod/apps/linux-ubuntu22.04-zen2/gcc-13.2.0/cuda-12.2.1-rqt4vr3vbmq5edgzd5glgq73htbneaw6/targets/x86_64-linux/lib:/usr/lib/gcc/x86_64-linux-gnu/11:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=/usr/lib/gcc/x86_64-linux-gnu/11:${LIBRARY_PATH:-}
export TMPDIR=/scratch/users/k24016446/tmp
mkdir -p "$TMPDIR"

cd ~/fusionguard-analytics
mkdir -p logs

source .venv_hpc/bin/activate

LLAMA_CPP_MODEL="${LLAMA_CPP_MODEL:-/scratch/users/k24016446/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf}"
if [[ ! -f "$LLAMA_CPP_MODEL" ]]; then
  echo "[ERROR] $(date) Model file $LLAMA_CPP_MODEL not found" >&2
  exit 1
fi
export LLAMA_CPP_MODEL

LLAMA_SERVICE_PORT="${LLAMA_SERVICE_PORT:-8005}"
export LLAMA_SERVICE_PORT

echo "[INFO] $(date) Starting FastAPI service on $(hostname) with model ${LLAMA_CPP_MODEL} (port ${LLAMA_SERVICE_PORT})"
uvicorn src.agent.service:app --host 0.0.0.0 --port "${LLAMA_SERVICE_PORT}"
