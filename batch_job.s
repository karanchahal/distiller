#!/bin/bash
#
#SBATCH --job-name=distillation
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=3GB
#SBATCH --gres=gpu:1

module purge
module load python3/intel/3.7.3
cd /home/$USER/nyu-cv-project
pip3 install --user torch torchvision numpy pandas tqdm seaborn
python3 evaluate_kd.py --epochs 350 --teacher resnet18 --student resnet8  --dataset cifar10 --teacher-checkpoint pretrained/resnet18_cifar10_95500.pth --mode nokd kd fd oh rkd takd