import argparse
from pathlib import Path

from distillers import *
from data_loader import get_cifar
from models.model_factory import create_model
from trainer import BaseTrainer, KDTrainer, MultiTrainer, TripletTrainer
from plot import plot_results
import util

BATCH_SIZE = 128
TESTFOLDER = "results"
USE_ID = True


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Test parameters")
    parser.add_argument("--epochs", default=100, type=int,
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
    parser.add_argument("--optimizer", default="sgd",
                        dest="optimizer", type=str,
                        help="Which optimizer to use")
    parser.add_argument("--scheduler", default="multisteplr",
                        dest="scheduler", type=str,
                        help="Which scheduler to use")
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
    t_net = create_model(t_name, num_classes, params["device"])
    teacher_config = params.copy()
    teacher_config["test_name"] = t_name + "_teacher"

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

    # also save this information in a csv file for plotting
    name = teacher_config["test_name"] + "_val"
    acc_file_name = params["results_dir"].joinpath(f"{name}.csv")
    with acc_file_name.open("w+") as acc_file:
        acc_file.write("Training Loss,Validation Loss\n")
        for _ in range(params["epochs"]):
            acc_file.write(f"0.0,{best_t_acc}\n")
    return t_net, best_teacher, best_t_acc


def setup_student(s_name, params):
    # Student Model
    num_classes = params["num_classes"]
    s_net = create_model(s_name, num_classes, params["device"])
    return s_net


def freeze_teacher(t_net):
    # freeze the layers of the teacher
    for param in t_net.parameters():
        param.requires_grad = False
    # set the teacher net into evaluation mode
    t_net.eval()
    return t_net


def test_nokd(s_net, t_net, params):
    print("---------- Training NOKD -------")
    nokd_config = params.copy()
    nokd_trainer = BaseTrainer(s_net, config=nokd_config)
    best_acc = nokd_trainer.train()
    return best_acc


def test_kd(s_net, t_net, params):
    t_net = freeze_teacher(t_net)
    print("---------- Training KD -------")
    kd_config = params.copy()
    kd_trainer = KDTrainer(s_net, t_net=t_net, config=kd_config)
    best_acc = kd_trainer.train()
    return best_acc


def test_triplet(s_net, t_net, params):
    t_net = freeze_teacher(t_net)
    print("---------- Training TRIPLET -------")
    kd_config = params.copy()
    kd_trainer = TripletTrainer(s_net, t_net=t_net, config=kd_config)
    best_acc = kd_trainer.train()
    return best_acc


def test_multikd(s_net, t_net1, params):
    t_net1 = freeze_teacher(t_net1)
    print("---------- Training MULTIKD -------")
    kd_config = params.copy()
    params["t2_name"] = "WRN22_4"
    t_net2 = create_model(
        params["t2_name"], params["num_classes"], params["device"])
    t_net2 = util.load_checkpoint(
        t_net2, "pretrained/WRN22_4_cifar10.pth")
    t_net2 = freeze_teacher(t_net2)

    params["t3_name"] = "resnet18"
    t_net3 = create_model(
        params["t3_name"], params["num_classes"], params["device"])
    t_net3 = util.load_checkpoint(
        t_net3, "pretrained/resnet18_cifar10.pth")
    t_net3 = freeze_teacher(t_net3)

    t_nets = [t_net1, t_net2]
    kd_trainer = MultiTrainer(s_net, t_nets=t_nets, config=kd_config)
    best_acc = kd_trainer.train()
    return best_acc


def test_takd(s_net, t_net, params):
    t_net = freeze_teacher(t_net)
    num_classes = params["num_classes"]
    # Arguments specifically for the teacher assistant approach
    params["ta_name"] = "resnet20"
    ta_model = create_model(
        params["ta_name"], num_classes, params["device"])
    best_acc = run_takd_distillation(s_net, ta_model, t_net, **params)
    return best_acc


def test_uda(s_net, t_net, params):
    t_net = freeze_teacher(t_net)
    best_acc = run_uda_distillation(s_net, t_net, **params)
    return best_acc


def test_ab(s_net, t_net, params):
    t_net = freeze_teacher(t_net)
    best_acc = run_ab_distillation(s_net, t_net, **params)
    return best_acc


def test_rkd(s_net, t_net, params):
    t_net = freeze_teacher(t_net)
    best_acc = run_rkd_distillation(s_net, t_net, **params)
    return best_acc


def test_pkd(s_net, t_net, params):
    t_net = freeze_teacher(t_net)
    best_acc = run_pkd_distillation(s_net, t_net, **params)
    return best_acc


def test_oh(s_net, t_net, params):
    # do not freeze the teacher in oh distillation
    best_acc = run_oh_distillation(s_net, t_net, **params)
    return best_acc


def test_fd(s_net, t_net, params):
    t_net = freeze_teacher(t_net)
    best_acc = run_fd_distillation(s_net, t_net, **params)
    return best_acc


def test_allkd(s_name, params):
    teachers = ["resnet8", "resnet14", "resnet20", "resnet26",
                "resnet32", "resnet44", "resnet56",
                # "resnet34", "resnet50", "resnet101", "resnet152",
                ]
    accs = {}
    for t_name in teachers:
        params_t = params.copy()
        params_t["teacher_name"] = t_name
        t_net, best_teacher, best_t_acc = setup_teacher(t_name, params_t)
        t_net = util.load_checkpoint(t_net, best_teacher, params_t["device"])
        t_net = freeze_teacher(t_net)
        s_net = setup_student(s_name, params_t)
        params_t["test_name"] = f"{s_name}_{t_name}"
        params_t["results_dir"] = params_t["results_dir"].joinpath("allkd")
        util.check_dir(params_t["results_dir"])
        best_acc = test_kd(s_net, t_net, params_t)
        accs[t_name] = (best_t_acc, best_acc)

    best_acc = 0
    best_t_acc = 0
    for t_name, acc in accs.items():
        if acc[0] > best_t_acc:
            best_t_acc = acc[0]
        if acc[1] > best_acc:
            best_acc = acc[1]
        print(f"Best results teacher {t_name}: {acc[0]}")
        print(f"Best results for {s_name}: {acc[1]}")

    return best_t_acc, best_acc


def test_kdparam(s_net, t_net, params):
    temps = [1, 5, 10, 15, 20]
    alphas = [0.1, 0.4, 0.5, 0.7, 1.0]
    param_pairs = [(a, T) for T in temps for a in alphas]
    accs = {}

    for alpha, T, in param_pairs:
        params_s = params.copy()
        params_s["lambda_student"] = alpha
        params_s["T_student"]: T
        s_name = params_s["student_name"]
        s_net = setup_student(s_name, params_s)
        params_s["test_name"] = f"{s_name}_{T}_{alpha}"
        print(f"Testing {s_name} with alpha {alpha} and T {T}.")
        best_acc = test_kd(s_net, t_net, params_s)
        accs[params_s["test_name"]] = (alpha, T, best_acc)

    best_kdparam_acc = 0
    for test_name, acc in accs.items():
        alpha = acc[0]
        T = acc[1]
        kd_acc = acc[2]
        if acc[2] > best_kdparam_acc:
            best_kdparam_acc = acc[2]
        print(f"Best results for {s_name} with a {alpha} and T {T}: {kd_acc}")

    return best_kdparam_acc


def run_benchmarks(modes, params, s_name, t_name):
    results = {}

    # if we test allkd we do not need to train an individual teacher
    if "allkd" in modes:
        best_t_acc, results["allkd"] = test_allkd(s_name, params)
        modes.remove("allkd")
    else:
        t_net, best_teacher, best_t_acc = setup_teacher(t_name, params)

    for mode in modes:
        mode = mode.lower()
        params_s = params.copy()
        # reset the teacher
        t_net = util.load_checkpoint(t_net, best_teacher, params["device"])

        # load the student and create a results directory for the mode
        s_net = setup_student(s_name, params)
        params_s["test_name"] = s_name
        params_s["results_dir"] = params_s["results_dir"].joinpath(mode)
        util.check_dir(params_s["results_dir"])
        # start the test
        try:
            run_test = globals()[f"test_{mode}"]
            results[mode] = run_test(s_net, t_net, params_s)
        except KeyError:
            raise RuntimeError(f"Training mode {mode} not supported!")

    # Dump the overall results
    print(f"Best results teacher {t_name}: {best_t_acc}")
    for name, acc in results.items():
        print(f"Best results for {s_name} with {name} method: {acc}")


def start_evaluation(args):
    device = util.setup_torch()
    num_classes = 100 if args.dataset == "cifar100" else 10
    train_loader, test_loader = get_cifar(num_classes,
                                          batch_size=args.batch_size)

    # for benchmarking, decided whether we want to use unique test folders
    if USE_ID:
        test_id = util.generate_id()
    else:
        test_id = ""
    results_dir = Path(args.results_dir).joinpath(test_id)
    results_dir = Path(results_dir).joinpath(args.dataset)
    util.check_dir(results_dir)

    # Parsing arguments and prepare settings for training
    params = {
        "epochs": args.epochs,
        "modes": args.modes,
        "t_checkpoint": args.t_checkpoint,
        "results_dir": results_dir,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "batch_size": args.batch_size,
        # model configuration
        "device": device,
        "teacher_name": args.t_name,
        "student_name": args.s_name,
        "num_classes": num_classes,
        # hyperparameters
        "weight_decay": args.weight_decay,
        "learning_rate": args.learning_rate,
        "momentum": args.momentum,
        "sched": args.scheduler,
        "optim": args.optimizer,
        # fixed knowledge distillation parameters
        "lambda_student": 0.5,
        "T_student": 5,
    }
    test_conf_name = results_dir.joinpath("test_config.json")
    util.dump_json_config(test_conf_name, params)
    run_benchmarks(args.modes, params, args.s_name, args.t_name)
    plot_results(results_dir, test_id=test_id)


if __name__ == "__main__":
    ARGS = parse_arguments()
    start_evaluation(ARGS)
