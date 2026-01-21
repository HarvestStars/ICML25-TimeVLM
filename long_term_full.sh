#!/usr/bin/env bash
#SBATCH -p gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem-per-gpu=256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 24:00:00
#SBATCH -J TimeVLM_Traffic
#SBATCH -D /projects/prjs1859/ICML25-TimeVLM
#SBATCH -o /projects/prjs1859/users/%u/slurm/logs/%x_%j.out
#SBATCH -e /projects/prjs1859/users/%u/slurm/logs/%x_%j.err

mkdir -p /projects/prjs1859/users/$USER/slurm/logs

source /projects/prjs1859/load_exp_env.sh

which python
python -c "import torch; print('torch ok', torch.__version__)"

sh TimeVLM_long_1.0p_chronos2.sh
