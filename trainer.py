import torch
from torch import nn
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


def init_progress_bar(train_loader):
    batch_size = train_loader.batch_size
    bar_format = "{desc}{percentage:3.0f}%"
    bar_format += "|{bar}|"
    bar_format += " {n_fmt}/{total_fmt} [{elapsed} < {remaining}]"
    bar_format += "{postfix}"
    t = tqdm(total=len(train_loader) * batch_size, bar_format=bar_format)
    return t


class Trainer(object):
    def __init__(self, net, train_config):
        self.net = net
        self.device = train_config["device"]
        self.name = train_config["name"]

        optim_cls, optim_args = train_config["optim"]
        sched_cls, sched_args = train_config["sched"]
        self.optimizer = optim_cls(net.parameters(), **optim_args)
        self.scheduler = sched_cls(self.optimizer, **sched_args)
        self.loss_fun = nn.CrossEntropyLoss()
        self.train_loader = train_config["train_loader"]
        self.test_loader = train_config["test_loader"]
        self.config = train_config
        # tqdm bar
        self.t_bar = None

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def set_loss_fun(self, loss_fun):
        self.loss_fun = loss_fun

    def calculate_loss(self, data, target):
        raise NotImplementedError("calculate_loss not implemented!")

    def train_single_epoch(self, t_bar):
        self.net.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            loss = self.calculate_loss(data, target)
            total_loss += loss
            t_bar.update(len(data))
            if batch_idx % 5 == 0:
                loss_avg = total_loss / batch_idx
                t_bar.set_postfix_str(f"Loss {loss_avg:.6f}")

    def train(self):
        epochs = self.config["epochs"]
        trial_id = self.config["trial_id"]

        best_acc = 0
        t_bar = init_progress_bar(self.train_loader)
        for epoch in range(epochs):
            # update progress bar
            t_bar.reset()
            t_bar.set_description(f"Epoch {epoch}")
            # perform training
            self.train_single_epoch(t_bar)
            # validate the output and save if it is the best so far
            val_acc = self.validate()
            if val_acc > best_acc:
                best_acc = val_acc
                self.save(epoch, name=f"{self.name}_{trial_id}_best.pth.tar")
            # update the scheduler
            if self.scheduler:
                self.scheduler.step()
        tqdm.clear(t_bar)
        t_bar.close()

        return best_acc

    def validate(self):
        self.net.eval()
        with torch.no_grad():
            correct = 0
            acc = 0
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                output = self.net(images)
                # Standard Learning Loss ( Classification Loss)
                loss = self.loss_fun(output, labels)
                # get the index of the max log-probability
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(labels.data.view_as(pred)).cpu().sum()

            acc = float(correct) / len(self.test_loader.dataset)
            print(f"\nValidation set: Average loss: {loss:.4f}, "
                  f"Accuracy: {correct}/{len(self.test_loader.dataset)} "
                  f"({acc * 100.0:.3f}%)\n")
            return acc

    def save(self, epoch, name=None):
        trial_id = self.config["trial_id"]
        if name is None:
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }, "{}_{}_epoch{}.pth.tar".format(self.name, trial_id, epoch))
        else:
            torch.save({
                "model_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
            }, name)


class BaseTrainer(Trainer):

    def __init__(self, net, train_config):
        super(BaseTrainer, self).__init__(net, train_config)

    def calculate_loss(self, data, target):
        # Standard Learning Loss ( Classification Loss)
        output = self.net(data)
        loss = self.loss_fun(output, target)
        loss.backward()
        self.optimizer.step()
        return loss
