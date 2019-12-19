#!/bin/bash

#SBATCH --job-name=kd_test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:p100:1
#SBATCH --output=slurm_%j.out

. ~/.bashrc
cd /scratch/ksc487/distill/nyu-cv-project/distill/research_seed/baselines/kd_baseline
conda activate light
