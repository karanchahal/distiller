'''Train CIFAR10 with PyTorch.'''

import time
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import math

from .distillation_module import Active_Soft_WRN_norelu


def criterion_alternative_L2(source, target, margin):
    # Proposed alternative loss function
    loss = ((source + margin)**2 * ((source > -margin) & (target <= 0)).float() +
            (source - margin)**2 * ((source <= margin) & (target > 0)).float())
    return torch.abs(loss).sum()


criterion_CE = torch.nn.CrossEntropyLoss()

# Settings
gpu_num = 0

distill_epoch = 10
max_epoch = 10

temperature = 3
base_lr = 0.1
KD = True


def init_distillation(s_net, ta_net, epoch, train_loader):
    epoch_start_time = time.time()
    print('\nDistillation epoch: %d' % epoch)
    ta_net.train()
    ta_net.s_net.train()
    ta_net.t_net.train()
    train_loss1 = 0
    train_loss2 = 0
    train_loss3 = 0
    optimizer = optim.SGD([{'params': s_net.parameters()},
                           {'params': ta_net.Connectors.parameters()}], lr=base_lr, nesterov=True, momentum=0.9, weight_decay=5e-4)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)

        batch_size = inputs.shape[0]
        ta_net(inputs)

        # Activation transfer loss
        loss_AT1 = ((ta_net.Connect1(ta_net.res1) > 0) ^ (
            ta_net.res1_t.detach() > 0)).sum().float() / ta_net.res1_t.nelement()
        loss_AT2 = ((ta_net.Connect2(ta_net.res2) > 0) ^ (
            ta_net.res2_t.detach() > 0)).sum().float() / ta_net.res2_t.nelement()
        loss_AT3 = ((ta_net.Connect3(ta_net.res3) > 0) ^ (
            ta_net.res3_t.detach() > 0)).sum().float() / ta_net.res3_t.nelement()

        # Alternative loss
        margin = 1.0
        loss_alter = criterion_alternative_L2(ta_net.Connect3(
            ta_net.res3), ta_net.res3_t.detach(), margin) / batch_size
        loss_alter += criterion_alternative_L2(ta_net.Connect2(
            ta_net.res2), ta_net.res2_t.detach(), margin) / batch_size / 2
        loss_alter += criterion_alternative_L2(ta_net.Connect1(
            ta_net.res1), ta_net.res1_t.detach(), margin) / batch_size / 4

        loss = loss_alter / 1000 * 3

        loss.backward()
        optimizer.step()

        train_loss1 += loss_AT1.item()
        train_loss2 += loss_AT2.item()
        train_loss3 += loss_AT3.item()

        b_idx = batch_idx

    print('Train \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('layer1_activation similarity %.1f%%' %
          (100 * (1 - train_loss1 / (b_idx + 1))))
    print('layer2_activation similarity %.1f%%' %
          (100 * (1 - train_loss2 / (b_idx + 1))))
    print('layer3_activation similarity %.1f%%' %
          (100 * (1 - train_loss3 / (b_idx + 1))))


# Training with KD loss


def train_KD(t_net, s_net, train_loader, optimizer, epoch):
    epoch_start_time = time.time()
    print('\nClassification training Epoch: %d' % epoch)
    s_net.train()
    t_net.eval()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        batch_size = inputs.shape[0]

        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        out_t = t_net(inputs)
        out_s = s_net(inputs)

        loss_CE = criterion_CE(out_s, targets)
        loss_KD = - (F.softmax(out_t / temperature, 1).detach() *
                     (F.log_softmax(out_s / temperature, 1) - F.log_softmax(out_t / temperature, 1).detach())).sum() / batch_size

        loss = loss_KD  # + loss_CE

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(out_s.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float()
        b_idx = batch_idx

    print('Train \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' %
          (train_loss / (b_idx + 1), 100. * correct / total, correct, total))
    return train_loss / (b_idx + 1)


def test(net, test_loader, save=False):
    epoch_start_time = time.time()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion_CE(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float()
        b_idx = batch_idx

    print('Test \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' %
          (test_loss / (b_idx + 1), 100. * correct / total, correct, total))
    return test_loss / (b_idx + 1), correct / total


def adjust_learning_rate(optimizer, epoch):
    lr = base_lr
    if epoch == math.ceil(max_epoch * 0.3) + 1:
        lr = base_lr / 5
    elif epoch == math.ceil(max_epoch * 0.6) + 1:
        lr = base_lr / (5 * 5)
    elif epoch == math.ceil(max_epoch * 0.8) + 1:
        lr = base_lr / (5 * 5 * 5)

    # update optimizer"s learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def run_ab_distillation(s_net, t_net, **params):
    ta_net = Active_Soft_WRN_norelu(t_net, s_net)
    train_loader = params["train_loader"]
    test_loader = params["test_loader"]

    torch.cuda.set_device(gpu_num)
    ta_net.cuda()
    s_net.cuda()
    t_net.cuda()
    cudnn.benchmark = True

    # Distillation (Initialization)
    for epoch in range(1, int(distill_epoch) + 1):
        init_distillation(s_net, ta_net, epoch, train_loader)
    # Classification training
    optimizer = optim.SGD(s_net.parameters(), lr=base_lr,
                          nesterov=True, momentum=0.9, weight_decay=5e-4)
    for epoch in range(1, max_epoch + 1):
        adjust_learning_rate(optimizer, epoch)
        train_loss = train_KD(t_net, s_net, train_loader, optimizer, epoch)
        test_loss, accuracy = test(s_net, test_loader, save=True)

    print('\nFinal accuracy: %.3f%%' % (100 * accuracy))
