import os
import copy
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from data_loader import get_cifar
from model_factory import create_cnn_model, is_resnet


BATCH_SIZE = 64


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', 'y', '1')


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='TA Knowledge Distillation Code')
    parser.add_argument('--epochs', default=200, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--dataset', default='cifar100', type=str,
                        help='dataset. can be either cifar10 or cifar100')
    parser.add_argument('--batch-size', default=BATCH_SIZE,
                        type=int, help='batch_size')
    parser.add_argument('--learning-rate', default=0.1,
                        type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9,
                        type=float, help='SGD momentum')
    parser.add_argument('--weight-decay', default=1e-4,
                        type=float, help='SGD weight decay (default: 1e-4)')
    parser.add_argument('--teacher', default='', type=str,
                        help='teacher student name')
    parser.add_argument('--ta', default='resnet14',
                        type=str, help='teacher student name')
    parser.add_argument('--student', '--model', default='resnet8',
                        type=str, help='teacher student name')
    parser.add_argument('--resume-state-ckp', default='', type=str,
                        help='optinal pretrained checkpoint for teacher training')
    parser.add_argument('--teacher-checkpoint', default='',
                        type=str, help='optinal pretrained checkpoint for teacher')
    parser.add_argument('--cuda', default=False, type=str2bool,
                        help='whether or not use cuda(train on GPU)')
    parser.add_argument('--dataset-dir', default='./data',
                        type=str, help='dataset directory')
    args = parser.parse_args()
    return args


def load_checkpoint(model, checkpoint_path):
    """
    Loads weights from checkpoint
    :param model: a pytorch nn student
    :param str checkpoint_path: address/path of a file
    :return: pytorch nn student with weights loaded from checkpoint
    """
    model_ckp = torch.load(checkpoint_path)
    model.load_state_dict(model_ckp['model_state_dict'])
    return model


def load_train_state(model, optimizer, checkpoint_path):
    """
    Loads weights from checkpoint
    :param model: a pytorch nn student
    :param str checkpoint_path: address/path of a file
    :return: pytorch nn student with weights loaded from checkpoint
    """
    model_ckp = torch.load(checkpoint_path)
    model.load_state_dict(model_ckp['model_state_dict'])
    optimizer.load_state_dict(model_ckp['optimizer_state_dict'])
    epoch = model_ckp['epoch']
    return model, optimizer, epoch


class TrainManager(object):
    def __init__(self, student, teacher=None, train_loader=None,
                 test_loader=None, train_config={}):
        self.student = student
        self.teacher = teacher
        self.have_teacher = bool(self.teacher)
        self.device = train_config['device']
        self.name = train_config['name']
        self.optimizer = optim.SGD(self.student.parameters(),
                                   lr=train_config['learning_rate'],
                                   momentum=train_config['momentum'],
                                   weight_decay=train_config['weight_decay'])
        if self.have_teacher:
            self.teacher.eval()
            self.teacher.train(mode=False)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = train_config
        self.start_epoch = 0
        self.criterion = nn.CrossEntropyLoss()

    def train_single_epoch(self, lambda_, T, epoch):
        total_loss = 0
        bar_format = "{desc} {percentage:3.0f}%"
        bar_format += "|{bar}|"
        bar_format += " {n_fmt}/{total_fmt} [{elapsed} < {remaining}]"
        t = tqdm(total=len(train_loader) * BATCH_SIZE, bar_format=bar_format)
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            output = self.student(data)
            # Standard Learning Loss ( Classification Loss)
            loss_SL = self.criterion(output, target)
            loss = loss_SL

            if self.have_teacher:
                teacher_outputs = self.teacher(data)
                # Knowledge Distillation Loss
                loss_KD = nn.KLDivLoss()(F.log_softmax(output / T, dim=1),
                                         F.softmax(teacher_outputs / T, dim=1))
                loss = (1 - lambda_) * loss_SL + lambda_ * T * T * loss_KD
            total_loss += loss
            loss.backward()
            self.optimizer.step()
            t.update(len(data))
            if batch_idx % 5 == 0:
                loss_avg = total_loss / batch_idx
                t.set_description(f"Epoch {epoch} Loss {loss_avg:.6f}")
                t.refresh()
        t.close()
        tqdm.clear(t)

    def train(self):
        lambda_ = self.config['lambda_student']
        T = self.config['T_student']
        epochs = self.config['epochs']
        trial_id = self.config['trial_id']

        max_val_acc = 0
        best_acc = 0
        for epoch in range(self.start_epoch, epochs):
            self.student.train()
            self.adjust_learning_rate(self.optimizer, epoch)

            self.train_single_epoch(lambda_, T, epoch)

            val_acc = self.validate(step=epoch)
            if val_acc > best_acc:
                best_acc = val_acc
                self.save(epoch, name=f"{self.name}_{trial_id}_best.pth.tar")

        return best_acc

    def validate(self, step=0):
        self.student.eval()
        with torch.no_grad():
            correct = 0
            acc = 0
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                output = self.student(images)
                # Standard Learning Loss ( Classification Loss)
                loss = self.criterion(output, labels)
                # get the index of the max log-probability
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(labels.data.view_as(pred)).cpu().sum()

            acc = 100.0 * correct / len(test_loader.dataset)
            print(f"Validation set: Average loss: {loss:.4f},"
                  f"Accuracy: {correct}/{len(test_loader.dataset)} "
                  f"({acc:.3f}%)\n")
            return acc

    def save(self, epoch, name=None):
        trial_id = self.config['trial_id']
        if name is None:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.student.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, '{}_{}_epoch{}.pth.tar'.format(self.name, trial_id, epoch))
        else:
            torch.save({
                'model_state_dict': self.student.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch,
            }, name)

    def adjust_learning_rate(self, optimizer, epoch):
        epochs = self.config['epochs']
        models_are_plane = self.config['is_plane']

        # depending on dataset
        if models_are_plane:
            lr = 0.01
        else:
            if epoch < int(epoch / 2.0):
                lr = 0.1
            elif epoch < int(epochs * 3 / 4.0):
                lr = 0.1 * 0.1
            else:
                lr = 0.1 * 0.01

        # update optimizer's learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


config = {
    # {"_type": "quniform", "_value": [0.05, 1.0, 0.05]},
    "lambda_student": 0.4,
    # {"_type": "choice", "_value": [1, 2, 5, 10, 15, 20]},
    "T_student": 10,
    #{"_type": "choice", "_value": [20, 31, 55]}
    "seed": 1,
}



def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1, bn=True, activation=True):
    op = [
        torch.nn.Conv2d(channels_in, channels_out,
                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
    ]
    if bn:
        op.append(torch.nn.BatchNorm2d(channels_out))
    if activation:
        op.append(torch.nn.ReLU(inplace=True))
    return torch.nn.Sequential(*op)


class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class Mul(torch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight


def build_network(num_class=10):
    return torch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        # torch.nn.MaxPool2d(2),

        Residual(torch.nn.Sequential(
            conv_bn(128, 128),
            conv_bn(128, 128),
        )),

        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),

        Residual(torch.nn.Sequential(
            conv_bn(256, 256),
            conv_bn(256, 256),
        )),

        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),

        torch.nn.AdaptiveMaxPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(128, num_class, bias=False),
        Mul(0.2)
    )


if __name__ == "__main__":
    # Parsing arguments and prepare settings for training
    args = parse_arguments()
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    trial_id = 1
    dataset = args.dataset
    num_classes = 100 if dataset == 'cifar100' else 10
    teacher_model = None
    train_config = {
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'device': 'cuda' if args.cuda else 'cpu',
        'is_plane': not is_resnet(args.student),
        'trial_id': trial_id,
        'T_student': config["T_student"],
        'lambda_student': config["lambda_student"],
    }

    # Train Teacher if provided a teacher, otherwise it's a normal training using only cross entropy loss
    # This is for training single models(NOKD in paper) for baselines models (or training the first teacher)
    train_loader, test_loader = get_cifar(num_classes, batch_size=BATCH_SIZE)
    if args.teacher:
        args.teacher = 'resnet101'
        teacher_model = build_network().cuda()
        # teacher_model = create_cnn_model(
        # args.teacher, dataset, use_cuda=args.cuda)
        if args.teacher_checkpoint:
            print("---------- Loading Teacher -------")
            teacher_model = load_checkpoint(
                teacher_model, args.teacher_checkpoint)
        else:
            print("---------- Training Teacher -------")
            teacher_train_config = copy.deepcopy(train_config)
            teacher_name = f"{args.teacher}_{trial_id}_best.pth.tar"
            teacher_train_config['name'] = args.teacher
            teacher_trainer = TrainManager(teacher_model,
                                           teacher=None,
                                           train_loader=train_loader,
                                           test_loader=test_loader,
                                           train_config=teacher_train_config)
            if args.resume_state_ckp:
                teacher_trainer.student, teacher_trainer.optimizer, teacher_trainer.start_epoch = load_train_state(
                    teacher_trainer.student, teacher_trainer.optimizer, args.resume_state_ckp)
            teacher_trainer.train()
            teacher_model = load_checkpoint(
                teacher_model, os.path.join('./', teacher_name))

    # Teaching Assistant training
    print("---------- Training TA -------")
    ta_model = create_cnn_model(args.ta, dataset, use_cuda=args.cuda)

    ta_train_config = copy.deepcopy(train_config)
    ta_train_config['name'] = args.ta
    ta_trainer = TrainManager(ta_model,
                              teacher=teacher_model,
                              train_loader=train_loader,
                              test_loader=test_loader,
                              train_config=ta_train_config)
    if args.resume_state_ckp:
        ta_trainer.student, ta_trainer.optimizer, ta_trainer.start_epoch = load_train_state(
            ta_trainer.student, ta_trainer.optimizer, args.resume_state_ckp)
    best_ta_acc = ta_trainer.train()

    # Student training
    print("---------- Training Student -------")
    student_model = create_cnn_model(args.student, dataset, use_cuda=args.cuda)
    student_train_config = copy.deepcopy(train_config)
    student_train_config['name'] = args.student
    student_trainer = TrainManager(student_model,
                                   teacher=ta_model,
                                   train_loader=train_loader,
                                   test_loader=test_loader,
                                   train_config=student_train_config)
    if args.resume_state_ckp:
        student_trainer.student, student_trainer.optimizer, student_trainer.start_epoch = load_train_state(
            student_trainer.student, student_trainer.optimizer, args.resume_state_ckp)
    best_student_acc = student_trainer.train()
