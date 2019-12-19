"""
This file runs the main training/val loop, etc... using Lightning Trainer    
"""
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from research_seed.baselines.segmentation.segmentation import KD_Segmentation
from pytorch_lightning.logging import TestTubeLogger
import torchvision
import torch

def main(hparams):
    # init module

    # Load student and teacher
    
    student = torchvision.models.segmentation.__dict__[hparams.student_model](num_classes=int(hparams.num_classes),
                                                                 aux_loss=hparams.aux_loss,
                                                                 pretrained=hparams.pretrained)
    teacher = torchvision.models.segmentation.__dict__[hparams.teacher_model](num_classes=int(hparams.num_classes),
                                                                aux_loss=hparams.aux_loss,
                                                                pretrained=hparams.pretrained)
    checkpoint = torch.load(hparams.path_to_teacher, map_location='cpu')
    teacher.load_state_dict(checkpoint['model'])

    model = KD_Segmentation(student, teacher, hparams)

    logger = TestTubeLogger(
       save_dir=hparams.save_dir,
       version=hparams.version # An existing version with a saved checkpoint
    )

    # most basic trainer, uses good defaults
    if hparams.gpus > 1:
    
      dist = 'ddp'
    else:
      dist = None

    # most basic trainer, uses good defaults
    trainer = Trainer(
        max_nb_epochs=hparams.epochs,
        gpus=hparams.gpus,
        nb_gpu_nodes=hparams.nodes,
        early_stop_callback=None,
        logger=logger,
        default_save_path=hparams.save_dir,
        distributed_backend=dist,
    )

    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--epochs', default=30, type=int,  help='number of total epochs to run')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--save-dir', type=str, default='./lightning_logs')
    parser.add_argument('--version', type=int, required=True, help= "version number for experiment")
    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = KD_Segmentation.add_model_specific_args(parser)

    # parse params
    hparams = parser.parse_args()

    main(hparams)
