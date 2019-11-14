"""
This file runs the main training/val loop, etc... using Lightning Trainer    
"""
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from research_seed.baselines.kd_baseline.kd_baseline import KD_Cifar
from pytorch_lightning.logging import TestTubeLogger

def main(hparams):
    # init module
    model = KD_Cifar(hparams)
    logger = TestTubeLogger(
       save_dir='./lightning_logs/',
       version=hparams.version # An existing version with a saved checkpoint
    )
    # most basic trainer, uses good defaults
    trainer = Trainer(
        max_nb_epochs=hparams.epochs,
        gpus=1,
        nb_gpu_nodes=hparams.nodes,
        early_stop_callback=None,
        logger=logger,
        # default_save_path='./lightning_logs/',
        #distributed_backend='ddp',
    )

    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--epochs', default=100, type=int,  help='number of total epochs to run')
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--version', type=int, default=1)
    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = KD_Cifar.add_model_specific_args(parser)

    # parse params
    hparams = parser.parse_args()

    main(hparams)
