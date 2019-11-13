import argparse
import torch

from data_loader import get_cifar
from models.model_factory import create_cnn_model
from teacher_assistant import run_teacher_assistant
from ab_distiller import run_ab_distillation

BATCH_SIZE = 128


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="TA Knowledge Distillation Code")
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
    parser.add_argument("--weight-decay", default=1e-4,
                        type=float, help="SGD weight decay (default: 1e-4)")
    parser.add_argument("--teacher", default="WRN22_4", type=str,
                        help="teacher student name")
    parser.add_argument("--student", "--model", default="resnet18",
                        type=str, help="teacher student name")
    parser.add_argument("--teacher-checkpoint", default="",
                        type=str, help="optinal pretrained checkpoint for teacher")
    parser.add_argument("--cuda", default=False, type=str2bool,
                        help="whether or not use cuda(train on GPU)")
    parser.add_argument("--dataset-dir", default="./data",
                        type=str, help="dataset directory")
    args = parser.parse_args()
    return args


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "y", "1")


def test_ta(dataset, params):
    # Arguments specifically for the teacher assistant approach
    params["ta"] = "resnet14"
    use_cuda = params["cuda"]
    ta_model = create_cnn_model(params["ta"], dataset, use_cuda=use_cuda)
    run_teacher_assistant(student_model, ta_model, teacher_model, **params)


def test_ab(dataset, params):
    # Model

    # Teacher network
    t_net = create_cnn_model("WRN22_4", dataset, use_cuda=args.cuda)
    model_ckp = torch.load("best_teacher_ab_95430.pth")
    t_net.load_state_dict(model_ckp["model_state_dict"])

    # Student network
    s_net = create_cnn_model("WRN16_2", dataset, use_cuda=args.cuda)
    # Arguments specifically for the ab approach
    run_ab_distillation(s_net, t_net, **params)


if __name__ == "__main__":
    args = parse_arguments()
    dataset = args.dataset
    num_classes = 100 if dataset == "cifar100" else 10
    train_loader, test_loader = get_cifar(num_classes, batch_size=BATCH_SIZE)

    # Parsing arguments and prepare settings for training
    torch.manual_seed(1)

    teacher_model = create_cnn_model(args.teacher, dataset, use_cuda=args.cuda)
    student_model = create_cnn_model(args.student, dataset, use_cuda=args.cuda)

    params = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "teacher_checkpoint": args.teacher_checkpoint,
        "cuda": args.cuda,
        "student": args.student,
        "teacher": args.teacher,
        "train_loader": train_loader,
        "test_loader": test_loader,
    }

    # test_ta(dataset, params)
    test_ab(dataset, params)
