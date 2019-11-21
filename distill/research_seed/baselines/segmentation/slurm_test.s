#!/bin/bash

#SBATCH --job-name=seg_test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:p40:1
#SBATCH --output=slurm_%j.out

. ~/.bashrc
cd /scratch/ksc487/distill/nyu-cv-project/distill/research_seed/baselines/segmentation
conda activate light
python segmentation_trainer.py --version 2 --path-to-teacher resnet101_chkps/model_29.pth --num-classes 21

