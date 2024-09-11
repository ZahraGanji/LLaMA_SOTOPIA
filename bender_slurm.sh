#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=DIT_train
#SBATCH --output=./out/train_net-%j.out
#SBATCH --error=./out/train_net-%j.err
#SBATCH --partition=A100short
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task=6
#SBATCH --time=8:00:00

source /home/s33zganj/.bashrc
module load Python
module load CUDA

srun torchrun --standalone --nproc_per_node 4 /home/s33zganj/LLaMA_SOTOPIA/pipeline.py