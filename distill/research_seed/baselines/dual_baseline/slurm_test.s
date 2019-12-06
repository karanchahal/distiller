#!/bin/bash

#SBATCH --job-name=ssl_test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:p40:1
#SBATCH --output=slurm_%j.out

. ~/.bashrc
cd /scratch/ksc487/distill/nyu-cv-project/distill/research_seed/baselines/dual_baseline/
conda activate light
python dual_model.py --version 1