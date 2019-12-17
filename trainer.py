import sys
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from optimizer import get_optimizer, get_scheduler


def init_progress_bar(train_loader):
    batch_size = train_loader.batch_size
    bar_format = "{desc}{percentage:3.0f}%"
    # bar_format += "|{bar}|"
    bar_format += " {n_fmt}/{total_fmt} [{elapsed} < {remaining}]"
    bar_format += "{postfix}"
    # if stderr has no tty disable the progress bar
    disable = not sys.stderr.isatty()
    t = tqdm(total=len(train_loader) * batch_size,
             bar_format=bar_format, disable=disable)
    if disable:
        # a trick to allow execution in environments where stderr is redirected
        t._time = lambda: 0.0
    return t


class Trainer():
    def __init__(self, net, config):
        self.net = net
        self.device = config["device"]
        self.name = config["test_name"]
        # Retrieve preconfigured optimizers and schedulers for all runs
        optim = config["optim"]
        sched = config["sched"]
        self.optim_cls, self.optim_args = get_optimizer(optim, config)
        self.sched_cls, self.sched_args = get_scheduler(sched, config)
        self.optimizer = self.optim_cls(net.parameters(), **self.optim_args)
        self.scheduler = self.sched_cls(self.optimizer, **self.sched_args)

        self.loss_fun = nn.CrossEntropyLoss()
        self.train_loader = config["train_loader"]
        self.test_loader = config["test_loader"]
        self.batch_size = self.train_loader.batch_size
        self.config = config
        # tqdm bar
        self.t_bar = None
        folder = config["results_dir"]
        self.best_model_file = folder.joinpath(f"{self.name}_best.pth.tar")
        acc_file_name = folder.joinpath(f"{self.name}_train.csv")
        self.acc_file = acc_file_name.open("w+")
        self.acc_file.write("Training Loss,Validation Loss\n")

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def set_loss_fun(self, loss_fun):
        self.loss_fun = loss_fun

    def calculate_loss(self, data, target):
        raise NotImplementedError(
            "calculate_loss should be implemented by subclass!")

    def train_single_epoch(self, t_bar):
        self.net.train()
        total_correct = 0.0
        total_loss = 0.0
        len_train_set = len(self.train_loader.dataset)
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()

            # this function is implemented by the subclass
            y_hat, loss = self.calculate_loss(x, y)

            # Metric tracking boilerplate
            pred = y_hat.data.max(1, keepdim=True)[1]
            total_correct += pred.eq(y.data.view_as(pred)).sum()
            total_loss += loss
            curr_acc = 100.0 * (total_correct / float(len_train_set))
            curr_loss = (total_loss / float(batch_idx))
            t_bar.update(self.batch_size)
            t_bar.set_postfix_str(f"Acc {curr_acc:.3f}% Loss {curr_loss:.3f}")
        total_acc = float(total_correct / len_train_set)
        return total_acc

    def train(self):
        epochs = self.config["epochs"]

        best_acc = 0
        t_bar = init_progress_bar(self.train_loader)
        for epoch in range(epochs):
            # update progress bar
            t_bar.reset()
            t_bar.set_description(f"Epoch {epoch}")
            # perform training
            train_acc = self.train_single_epoch(t_bar)
            # validate the output and save if it is the best so far
            val_acc = self.validate(epoch)
            if val_acc > best_acc:
                best_acc = val_acc
                self.save(epoch, name=self.best_model_file)
            # update the scheduler
            if self.scheduler:
                self.scheduler.step()
            self.acc_file.write(f"{train_acc},{val_acc}\n")
        tqdm.clear(t_bar)
        t_bar.close()
        self.acc_file.close()
        return best_acc

    def validate(self, epoch=0):
        self.net.eval()
        acc = 0.0
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
            print(f"\nEpoch {epoch}: Validation set: Average loss: {loss:.4f},"
                  f" Accuracy: {correct}/{len(self.test_loader.dataset)} "
                  f"({acc * 100.0:.3f}%)")
        return acc

    def save(self, epoch, name):
        torch.save({"model_state_dict": self.net.state_dict(), }, name)


class BaseTrainer(Trainer):

    def calculate_loss(self, data, target):
        # Standard Learning Loss ( Classification Loss)
        output = self.net(data)
        loss = self.loss_fun(output, target)
        loss.backward()
        self.optimizer.step()
        return output, loss


class KDTrainer(Trainer):
    def __init__(self, s_net, t_net, config):
        super(KDTrainer, self).__init__(s_net, config)
        # the student net is the base net
        self.s_net = self.net
        self.t_net = t_net
        self.kd_fun = nn.KLDivLoss(size_average=False)

    def kd_loss(self, out_s, out_t, target):
        lambda_ = self.config["lambda_student"]
        T = self.config["T_student"]
        # Standard Learning Loss ( Classification Loss)
        loss = self.loss_fun(out_s, target)
        # Knowledge Distillation Loss
        batch_size = target.shape[0]
        s_max = F.log_softmax(out_s / T, dim=1)
        t_max = F.softmax(out_t / T, dim=1)
        loss_kd = self.kd_fun(s_max, t_max) / batch_size
        loss = (1 - lambda_) * loss + lambda_ * T * T * loss_kd
        return loss

    def calculate_loss(self, data, target):
        out_s = self.s_net(data)
        out_t = self.t_net(data)
        loss = self.kd_loss(out_s, out_t, target)
        loss.backward()
        self.optimizer.step()
        return out_s, loss


class TripletTrainer(KDTrainer):
    def __init__(self, s_net, t_net, config):
        super(TripletTrainer, self).__init__(s_net, t_net, config)
        # the student net is the base net
        self.s_net = self.net
        self.t_net = t_net
        self.triplet = F.cosine_embedding_loss

    def kd_loss(self, out_s, out_t, target):
        lambda_ = self.config["lambda_student"]
        T = self.config["T_student"]
        # Standard Learning Loss ( Classification Loss)
        # loss = self.loss_fun(out_s, target)
        # Knowledge Distillation Loss
        batch_size = target.shape[0]
        s_max = F.log_softmax(out_s / T, dim=1)
        t_max = F.softmax(out_t / T, dim=1)
        # pred_s = out_s.data.max(1, keepdim=True)[1]
        # pred_t = out_t.data.max(1, keepdim=True)[1]
        y = torch.ones(target.shape[0]).cuda()
        loss = self.triplet(out_s, out_t, y)
        loss_kd = self.kd_fun(s_max, t_max) / batch_size
        loss = (1 - lambda_) * loss + lambda_ * T * T * loss_kd
        return loss

    def calculate_loss(self, data, target):
        out_s = self.s_net(data)
        out_t = self.t_net(data)
        loss = self.kd_loss(out_s, out_t, target)
        loss.backward()
        self.optimizer.step()
        return out_s, loss


class MultiTrainer(KDTrainer):
    def __init__(self, s_net, t_nets, config):
        super(MultiTrainer, self).__init__(s_net, s_net, config)
        # the student net is the base net
        self.s_net = self.net
        self.t_nets = t_nets

    def kd_loss(self, out_s, out_t, target):
        T = self.config["T_student"]
        # Knowledge Distillation Loss
        batch_size = target.shape[0]
        s_max = F.log_softmax(out_s / T, dim=1)
        t_max = F.softmax(out_t / T, dim=1)
        loss_kd = self.kd_fun(s_max, t_max) / batch_size
        return loss_kd

    def calculate_loss(self, data, target):
        lambda_ = self.config["lambda_student"]
        T = self.config["T_student"]
        out_s = self.s_net(data)
        # Standard Learning Loss ( Classification Loss)
        loss = self.loss_fun(out_s, target)
        # Knowledge Distillation Loss
        loss_kd = 0.0
        for t_net in self.t_nets:
            out_t = t_net(data)
            loss_kd += self.kd_loss(out_s, out_t, target)
        loss_kd /= len(self.t_nets)
        loss = (1 - lambda_) * loss + lambda_ * T * T * loss_kd
        loss.backward()
        self.optimizer.step()
        return out_s, loss


class BlindTrainer(KDTrainer):
    def __init__(self, s_net, t_net, config):
        super(BlindTrainer, self).__init__(s_net, config)
        # the student net is the base net
        self.s_net = self.net
        self.t_net = t_net

    def calculate_loss(self, data):
        lambda_ = self.config["lambda_student"]
        T = self.config["T_student"]
        out_s = self.s_net(data)

        # Knowledge Distillation Loss
        out_t = self.t_net(data)
        s_max = F.log_softmax(out_s / T, dim=1)
        t_max = F.softmax(out_t / T, dim=1)
        batch_size = s_max.shape[0]
        loss_kd = F.kl_div(s_max, t_max, size_average=False) / batch_size
        loss = lambda_ * T * T * loss_kd
        loss.backward()
        self.optimizer.step()
        return out_s, loss

    def train_single_epoch(self, t_bar):
        self.net.train()
        total_loss = 0
        iters = int(len(self.train_loader.dataset) / self.batch_size)
        for batch_idx in range(iters):
            data = torch.randn((self.batch_size, 3, 32, 32)).to(self.device)
            self.optimizer.zero_grad()
            loss = self.calculate_loss(data)
            total_loss += loss
            t_bar.update(self.batch_size)
            loss_avg = total_loss / batch_idx
            t_bar.set_postfix_str(f"Loss {loss_avg:.6f}")
        return total_loss / len(self.train_loader.dataset)
