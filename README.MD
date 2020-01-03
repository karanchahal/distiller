# Distiller


<img src="distiller.jpg" alt="alt text" width="200" height="200">

A clean Pytorch implementation to run quick distillation experiments. Our findings are available for reading in our paper "The State of Knowledge Distillation for Classification" linked [here](https://arxiv.org/abs/1912.10850).

## Python Dependencies
This codebase only supports Python 3.6+.

Required Python packages:
- `torch torchvision tqdm numpy pandas seaborn`

All packages can be installed using `pip3 install --user -r requirements.txt`.

This project is also integerated with Pytorch Lightning. Use the lightning branch to see Pytorch Lightning compatible code.

## Run
The benchmarks can be run via `python3 evaluate_kd.py` and providing the
respective command line parameters. For example:

`python3 evaluate_kd.py --epochs 200 --teacher resnet18 --student resnet8  --dataset cifar10 --teacher-checkpoint pretrained/resnet18_cifar10_95260_parallel.pth --mode nokd kd`

Runs basic student training and knowledge distillation for 200 epochs using a
pretrained teacher. There are checkpoints of multiple models in the pretrained folder.


## Supported distillation modes

| Distillation Technique        | Mode           | Description  |
| ------------- |:-------------:| -----:|
| Baseline Accuracy | nokd | Plain training with no knowledge distillation. |
| Baseline Knowledge Distillation Accuracy | kd | Hinton loss to distill a student network. |
| Ensemble of Teachers | allkd | Distill from a list of teacher models and pick the best performing one. |
| Hyper parameter tuning of Hinton Loss | kdparam | Distill using varying combinations of temperature and alpha and pick the best performing combination. |
| Triplet Loss | triplet | Knowledge Distillation with a triplet loss using the student as negative example. |
| Ensemble of Students | multikd | Train a student under an ensemble of students that are picked from a list. |
| Unsupervised Data Augmentation Loss | uda | Run knowledge distillation in combination with [unsupervised data augmentation](https://github.com/google-research/uda). |
| Teacher Assistant Knowledge Distillation | takd | Run distillation using [Teacher-Assistant](https://github.com/imirzadeh/Teacher-Assistant-Knowledge-Distillation) distillation. |
| Activation Boundary Distillation | ab | Run feature distillation using [Activation-Boundary](https://github.com/bhheo/AB) distillation. |
| Overhaul Distillation | oh | Run feature distillation using the [Feature Overhaul](https://github.com/clovaai/overhaul-distillation) distillation. |
| Relational Knowledge Distillation | rkd | Run distillation using the [Relational Knowledge](https://github.com/lenscloth/RKD) distillation. |
| Patient Knowledge Distillation | pkd | Run feature distillation using the [Patient Knowledge](https://github.com/intersun/PKD-for-BERT-Model-Compression) distillation. |
| Simple Knowledge Distillation | sfd | Runs a custom feature distillation distillation (Simple Feature Distillation) that just pools and flattens feature layers. |

## Results

