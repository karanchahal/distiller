import argparse
import copy
import torch

from data_loader import get_cifar
from models.model_factory import create_cnn_model
from optimizer import get_optimizer, get_scheduler

from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from pl_trainer import load_checkpoint, BaseTrainer, KDTrainer


BATCH_SIZE = 128


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation Params")
    parser.add_argument("--epochs", default=200, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--dataset", default="cifar10", type=str,
                        help="dataset. can be either cifar10 or cifar100")
    parser.add_argument("--batch-size", default=BATCH_SIZE,
                        type=int, help="batch_size")
    parser.add_argument("--learning-rate", default=0.1,
                        type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9,
                        type=float, help="SGD momentum")
    parser.add_argument("--weight-decay", default=5e-4,
                        type=float, help="SGD weight decay (default: 5e-4)")
    parser.add_argument("--teacher", default="WRN22_4", type=str,
                        dest="t_name", help="teacher student name")
    parser.add_argument("--student", "--model", default="resnet18",
                        dest="s_name", type=str, help="teacher student name")
    parser.add_argument("--teacher-checkpoint", default="",
                        dest="t_checkpoint", type=str,
                        help="optional pretrained checkpoint for teacher")
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--save-dir', dest="save_dir",
                        type=str, default='./lightning_logs')
    parser.add_argument('--version', type=int, default=1,
                        help="version number for experiment")
    args = parser.parse_args()
    return args


def train_pl(model, params):
    logger = TestTubeLogger(
        save_dir=params["save_dir"],
        # An existing version with a saved checkpoint
        version=params["version"]
    )
    # most basic trainer, uses good defaults
    if params["gpus"] > 1:
        dist = 'ddp'
    else:
        dist = None

    # most basic trainer, uses good defaults
    trainer = Trainer(
        max_nb_epochs=params["epochs"],
        gpus=params["gpus"],
        nb_gpu_nodes=params["nodes"],
        early_stop_callback=None,
        logger=logger,
        default_save_path=params["save_dir"],
        distributed_backend=dist,
    )

    trainer.fit(model)
    return model


def init_teacher(t_name, params):
    num_classes = params["num_classes"]
    # Teacher training
    t_net = create_cnn_model(t_name, num_classes, params["device"])
    train_config = copy.deepcopy(params)
    train_config["name"] = params["t_name"]

    if params["t_checkpoint"]:
        print("---------- Loading Teacher -------")
        t_net = load_checkpoint(t_net, params["t_checkpoint"])

    model = BaseTrainer(t_net, train_config)
    if params["t_checkpoint"]:
        best_t_acc = model.validate_full()
    else:
        teacher_name = params["t_name"]
        version = params["version"]
        best_teacher = f"{teacher_name}_{version}_best.pth.tar"
        print("---------- Training Teacher -------")
        model = train_pl(model, params)
        t_net = load_checkpoint(t_net, best_teacher)
    return t_net, best_t_acc


def test_kd(t_net, params):
    num_classes = params["num_classes"]
    # Student Model
    s_name = params["s_name"]
    s_net = create_cnn_model(s_name, num_classes, params["device"])
    # Arguments specifically for the teacher assistant approach
    params["ta_name"] = "resnet8"
    train_config = copy.deepcopy(params)
    train_config["name"] = params["s_name"]

    s_net = create_cnn_model(
        params["ta_name"], num_classes, params["device"])
    model = KDTrainer(s_net, t_net, train_config)
    model = train_pl(model, params)


def start_evaluation(args):
    # Setup cuda
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True
    # Maximum determinism
    torch.manual_seed(1)
    print(f"Using {device} to train.")

    num_classes = 100 if args.dataset == "cifar100" else 10
    train_loader, test_loader = get_cifar(num_classes,
                                          batch_size=args.batch_size)

    # Parsing arguments and prepare settings for training
    params = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "t_checkpoint": args.t_checkpoint,
        "device": device,
        "s_name": args.s_name,
        "t_name": args.t_name,
        "num_classes": num_classes,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "gpus": args.gpus,
        "nodes": args.nodes,
        "save_dir": args.save_dir,
        "version": args.version,
        # {"_type": "quniform", "_value": [0.05, 1.0, 0.05]},
        "lambda_student": 0.4,
        # {"_type": "choice", "_value": [1, 2, 5, 10, 15, 20]},
        "T_student": 20,
    }
    # Retrieve preconfigured optimizers and schedulers for all runs
    params["optim"] = get_optimizer("SGD", params)
    params["sched"] = get_scheduler("multisteplr", params)
    t_name = params["t_name"]
    t_net, best_t_acc = init_teacher(t_name, params)
    test_kd(t_net, params)


if __name__ == "__main__":
    args = parse_arguments()
    start_evaluation(args)
