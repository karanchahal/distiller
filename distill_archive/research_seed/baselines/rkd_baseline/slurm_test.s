#!/bin/bash

#SBATCH --job-name=seg_test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:p40:1
#SBATCH --output=slurm_%j.out

. ~/.bashrc
cd /scratch/ksc487/distill/nyu-cv-project/distill/research_seed/baselines/rkd_baseline
conda activate light

python rkd_distill.py --version 3 --path-to-teacher lightning_logs/default/version_1/checkpoints/_ckpt_epoch_40.ckpt
#python rkd_baseline_trainer.py --version 1 --path-to-teacher ../no_kd_baseline/lightning_logs/default/version_1/checkpoints/_ckpt_epoch_25.ckpt --epochs 40
