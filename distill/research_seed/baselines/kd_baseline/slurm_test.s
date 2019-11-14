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

python kd_baseline_trainer.py --cuda True --student-model resnet8 --teacher-model resnet110 --path-to-teacher ../no_kd_baseline/lightning_logs/default/version_1/checkpoints/_ckpt_epoch_25.ckpt --version 16 --alpha 0 --temperature 20
