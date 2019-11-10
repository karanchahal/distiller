import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm


def load_checkpoint(model, checkpoint_path):
    """
    Loads weights from checkpoint
    :param model: a pytorch nn student
    :param str checkpoint_path: address/path of a file
    :return: pytorch nn student with weights loaded from checkpoint
    """
    model_ckp = torch.load(checkpoint_path)
    model.load_state_dict(model_ckp["model_state_dict"])
    return model


def load_train_state(model, optimizer, checkpoint_path):
    """
    Loads weights from checkpoint
    :param model: a pytorch nn student
    :param str checkpoint_path: address/path of a file
    :return: pytorch nn student with weights loaded from checkpoint
    """
    model_ckp = torch.load(checkpoint_path)
    model.load_state_dict(model_ckp["model_state_dict"])
    optimizer.load_state_dict(model_ckp["optimizer_state_dict"])
    epoch = model_ckp["epoch"]
    return model, optimizer, epoch


class TrainManager(object):
    def __init__(self, student, teacher=None, train_loader=None,
                 test_loader=None, train_config={}):
        self.student = student
        self.teacher = teacher
        self.have_teacher = bool(self.teacher)
        self.device = train_config["device"]
        self.name = train_config["name"]
        self.optimizer = optim.SGD(self.student.parameters(),
                                   lr=train_config["learning_rate"],
                                   momentum=train_config["momentum"],
                                   weight_decay=train_config["weight_decay"])
        if self.have_teacher:
            self.teacher.eval()
            self.teacher.train(mode=False)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.batch_size = train_loader.batch_size
        self.config = train_config
        self.criterion = nn.CrossEntropyLoss()

    def train_single_epoch(self, lambda_, T, epoch):
        total_loss = 0
        bar_format = "{desc} {percentage:3.0f}%"
        bar_format += "|{bar}|"
        bar_format += " {n_fmt}/{total_fmt} [{elapsed} < {remaining}]"
        t = tqdm(total=len(self.train_loader) *
                 self.batch_size, bar_format=bar_format)
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
        lambda_ = self.config["lambda_student"]
        T = self.config["T_student"]
        epochs = self.config["epochs"]
        trial_id = self.config["trial_id"]

        best_acc = 0
        for epoch in range(epochs):
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

            acc = 100.0 * correct / len(self.test_loader.dataset)
            print(f"Validation set: Average loss: {loss:.4f},"
                  f"Accuracy: {correct}/{len(self.test_loader.dataset)} "
                  f"({acc:.3f}%)\n")
            return acc

    def save(self, epoch, name=None):
        trial_id = self.config["trial_id"]
        if name is None:
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.student.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }, "{}_{}_epoch{}.pth.tar".format(self.name, trial_id, epoch))
        else:
            torch.save({
                "model_state_dict": self.student.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
            }, name)

    def adjust_learning_rate(self, optimizer, epoch):
        epochs = self.config["epochs"]

        if epoch < int(epoch / 2.0):
            lr = 0.1
        elif epoch < int(epochs * 3 / 4.0):
            lr = 0.1 * 0.1
        else:
            lr = 0.1 * 0.01

        # update optimizer"s learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


config = {
    # {"_type": "quniform", "_value": [0.05, 1.0, 0.05]},
    "lambda_student": 0.4,
    # {"_type": "choice", "_value": [1, 2, 5, 10, 15, 20]},
    "T_student": 10,
}


def run_teacher_assistant(student_model, ta_model, teacher_model, **params):
    trial_id = 1
    train_config = {
        "epochs": params["epochs"],
        "learning_rate": params["learning_rate"],
        "momentum": params["momentum"],
        "weight_decay": params["weight_decay"],
        "device": "cuda" if params["cuda"] else "cpu",
        "trial_id": trial_id,
        "T_student": config["T_student"],
        "lambda_student": config["lambda_student"],
    }
    train_loader = params["train_loader"]
    test_loader = params["test_loader"]
    # Train Teacher if provided a teacher, otherwise it"s a normal training using only cross entropy loss
    # This is for training single models(NOKD in paper) for baselines models (or training the first teacher)
    if params["teacher"]:
        if params["teacher_checkpoint"]:
            print("---------- Loading Teacher -------")
            teacher_model = load_checkpoint(
                teacher_model, params["teacher_checkpoint"])
            best_teacher_acc = 0
        else:
            print("---------- Training Teacher -------")
            teacher_train_config = copy.deepcopy(train_config)
            teacher_name = params["teacher"]
            best_teacher = f"{teacher_name}_{trial_id}_best.pth.tar"
            teacher_train_config["name"] = params["teacher"]
            teacher_trainer = TrainManager(teacher_model,
                                           teacher=None,
                                           train_loader=train_loader,
                                           test_loader=test_loader,
                                           train_config=teacher_train_config)
            best_teacher_acc = teacher_trainer.train()
            teacher_model = load_checkpoint(
                teacher_model, os.path.join("./", best_teacher))

    # Teaching Assistant training
    print("---------- Training TA -------")

    ta_train_config = copy.deepcopy(train_config)
    ta_name = params["teacher"]
    best_ta = f"{ta_name}_{trial_id}_best.pth.tar"
    ta_trainer = TrainManager(ta_model,
                              teacher=teacher_model,
                              train_loader=train_loader,
                              test_loader=test_loader,
                              train_config=ta_train_config)
    best_ta_acc = ta_trainer.train()
    ta_model = load_checkpoint(
        ta_model, os.path.join("./", best_ta))

    # Student training
    print("---------- Training Student -------")
    student_train_config = copy.deepcopy(train_config)
    student_trainer = TrainManager(student_model,
                                   teacher=ta_model,
                                   train_loader=train_loader,
                                   test_loader=test_loader,
                                   train_config=student_train_config)
    best_student_acc = student_trainer.train()

    print(f"Final results teacher: {best_teacher_acc}")
    print(f"Final results ta: {best_ta_acc}")
    print(f"Final results student: {best_student_acc}")
