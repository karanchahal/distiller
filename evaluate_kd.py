import argparse
from pathlib import Path

import torch
from data_loader import get_cifar
from models.model_factory import create_cnn_model
from distillers import *
from trainer import BaseTrainer, KDTrainer
from plot import plot_results
import util

BATCH_SIZE = 128
TESTFOLDER = "results"
USE_ID = False


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation Params")
    parser.add_argument("--epochs", default=200, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--dataset", default="cifar100", type=str,
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
    parser.add_argument("--mode", default=["KD"], dest="modes",
                        type=str, nargs='+',
                        help="What type of distillation to use")
    parser.add_argument("--results-dir", default=TESTFOLDER,
                        dest="results_dir", type=str,
                        help="Where all results are collected")
    args = parser.parse_args()
    return args


def setup_teacher(t_name, params):
    # Teacher Model
    num_classes = params["num_classes"]
    t_net = create_cnn_model(t_name, num_classes, params["device"])
    teacher_config = params.copy()
    teacher_config["test_name"] = params["teacher_name"]

    if params["t_checkpoint"]:
        # Just validate the performance
        print("---------- Loading Teacher -------")
        best_teacher = params["t_checkpoint"]
    else:
        # Teacher training
        print("---------- Training Teacher -------")
        teacher_trainer = BaseTrainer(t_net, config=teacher_config)
        teacher_trainer.train()
        best_teacher = teacher_trainer.best_model_file

    # reload and get the best model
    t_net = util.load_checkpoint(t_net, best_teacher)
    teacher_trainer = BaseTrainer(t_net, config=teacher_config)
    best_t_acc = teacher_trainer.validate()

    # also save this information in a csv file
    name = params["teacher_name"] + "_val"
    acc_file_name = params["results_dir"].joinpath(f"{name}.csv")
    with acc_file_name.open("w+") as acc_file:
        acc_file.write("Training Loss,Validation Loss\n")
        for _ in range(params["epochs"]):
            acc_file.write(f"0.0,{best_t_acc}\n")
    return t_net, best_teacher, best_t_acc


def init_student(s_name, params):
    # Student Model
    num_classes = params["num_classes"]
    s_net = create_cnn_model(s_name, num_classes, params["device"])
    return s_net


def freeze_teacher(t_net):
    # freeze the layers of the teacher
    for param in t_net.parameters():
        param.requires_grad = False
    # set the teacher net into evaluation mode
    t_net.eval()
    return t_net


def test_nokd(s_net, params):
    print("---------- Training NOKD -------")
    nokd_config = params.copy()
    nokd_trainer = BaseTrainer(s_net, config=nokd_config)
    best_nokd_acc = nokd_trainer.train()
    return best_nokd_acc


def test_kd(s_net, t_net, params):
    t_net = freeze_teacher(t_net)
    print("---------- Training KD -------")
    kd_config = params.copy()
    kd_trainer = KDTrainer(s_net, t_net=t_net, config=kd_config)
    best_kd_acc = kd_trainer.train()
    return best_kd_acc


def test_ta(s_net, t_net, params):
    t_net = freeze_teacher(t_net)
    num_classes = params["num_classes"]
    # Arguments specifically for the teacher assistant approach
    params["ta_name"] = "resnet8"
    ta_model = create_cnn_model(
        params["ta_name"], num_classes, params["device"])
    best_ta_acc = run_takd_distillation(s_net, ta_model, t_net, **params)
    return best_ta_acc


def test_ab(s_net, t_net, params):
    t_net = freeze_teacher(t_net)
    # Arguments specifically for the ab approach
    best_ab_acc = run_ab_distillation(s_net, t_net, **params)
    return best_ab_acc


def test_rkd(s_net, t_net, params):
    t_net = freeze_teacher(t_net)
    # Arguments specifically for the ab approach
    best_rkd_acc = run_rkd_distillation(s_net, t_net, **params)
    return best_rkd_acc


def test_pkd(s_net, t_net, params):
    t_net = freeze_teacher(t_net)
    # Arguments specifically for the ab approach
    best_pkd_acc = run_pkd_distillation(s_net, t_net, **params)
    return best_pkd_acc


def test_oh(s_net, t_net, params):
    t_net = freeze_teacher(t_net)
    # Arguments specifically for the ab approach
    best_oh_acc = run_oh_distillation(s_net, t_net, **params)
    return best_oh_acc


def test_fd(s_net, t_net, params):
    t_net = freeze_teacher(t_net)
    # Arguments specifically for the ab approach
    best_fd_acc = run_fd_distillation(s_net, t_net, **params)
    return best_fd_acc


def run_benchmarks(modes, params, s_name, t_name):
    results = {}

    t_net, best_teacher, best_t_acc = setup_teacher(t_name, params)

    for mode in modes:
        mode = mode.lower()
        params_t = params.copy()

        # reset the teacher
        t_net = util.load_checkpoint(t_net, best_teacher, params["device"])

        s_net = init_student(s_name, params)
        params_t["test_name"] = s_name
        params_t["results_dir"] = params_t["results_dir"].joinpath(mode)
        util.check_dir(params_t["results_dir"])
        if mode == "nokd":
            results[mode] = test_nokd(s_net, params_t)
        elif mode == "kd":
            results[mode] = test_kd(s_net, t_net, params_t)
        elif mode == "takd":
            results[mode] = test_ta(s_net, t_net, params_t)
        elif mode == "ab":
            results[mode] = test_ab(s_net, t_net, params_t)
        elif mode == "rkd":
            results[mode] = test_rkd(s_net, t_net, params_t)
        elif mode == "pkd":
            results[mode] = test_pkd(s_net, t_net, params_t)
        elif mode == "oh":
            results[mode] = test_oh(s_net, t_net, params_t)
        elif mode == "fd":
            results[mode] = test_fd(s_net, t_net, params_t)
        else:
            raise RuntimeError(f"Training mode {mode} not supported!")

    print(f"Best results teacher {t_name}: {best_t_acc}")
    for name, acc in results.items():
        print(f"Best results for {s_name} with {name} method: {acc}")


def setup_torch():
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    if use_cuda:
        torch.backends.cudnn.benchmark = True
    # Maximum determinism
    torch.manual_seed(1)
    print(f"Using {device} to train.")
    return device


def start_evaluation(args):
    device = setup_torch()
    num_classes = 100 if args.dataset == "cifar100" else 10
    train_loader, test_loader = get_cifar(num_classes,
                                          batch_size=args.batch_size)

    if USE_ID:
        test_id = util.generate_id()
    else:
        test_id = ""
    results_dir = Path(args.results_dir).joinpath(test_id)
    results_dir = Path(results_dir).joinpath(args.dataset)
    util.check_dir(results_dir)
    teacher_name = args.t_name + "_teacher"

    # Parsing arguments and prepare settings for training
    params = {
        "epochs": args.epochs,
        "modes": args.modes,
        "learning_rate": args.learning_rate,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "t_checkpoint": args.t_checkpoint,
        "results_dir": results_dir,
        "device": device,
        "num_classes": num_classes,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "optim": "SGD",
        "sched": "multisteplr",
        "teacher_name": teacher_name,
        "student_name": args.s_name,
        "lambda_student": 0.4,
        "T_student": 10,
    }
    test_conf_name = results_dir.joinpath("test_config.json")
    util.dump_json_config(test_conf_name, params)
    run_benchmarks(args.modes, params, args.s_name, args.t_name)
    plot_results(results_dir, test_id=test_id)


if __name__ == "__main__":
    args = parse_arguments()
    start_evaluation(args)
