"""
This file runs the main training/val loop, etc... using Lightning Trainer    
"""
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from research_seed.baselines.no_kd_baseline.no_kd_baseline import NO_KD_Cifar


def main(hparams):
    # init module
    model = NO_KD_Cifar(hparams)

    # most basic trainer, uses good defaults
    trainer = Trainer(
        max_nb_epochs=hparams.epochs,
        gpus=hparams.gpus,
        nb_gpu_nodes=hparams.nodes
    )

    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--epochs', default=50, type=int,  help='number of total epochs to run')
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--nodes', type=int, default=1)

    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = NO_KD_Cifar.add_model_specific_args(parser)

    # parse params
    hparams = parser.parse_args()

    main(hparams)
