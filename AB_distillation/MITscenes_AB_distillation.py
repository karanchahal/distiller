'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import argparse
import time
import os

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from models import *


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res

dataset_name = 'MITscenes'

parser = argparse.ArgumentParser(description='PyTorch MITscenes Training')

gpu_nums = [0, 1]

parser.add_argument('--loss_multiplier', default=1, type=float, help='multiplier to loss')
parser.add_argument('--pretrained', default=False, type=int, help='Use pretrained network')
parser.add_argument('--DTL', default=True, type=int, help='DTL (Distillation in Transfer Learning) method')
parser.add_argument('--distill_epoch', default=60, type=int, help='epoch for distillation')
parser.add_argument('--max_epoch', default=100, type=int, help='epoch for all')
parser.add_argument('--lr', default=0.01, type=float, metavar='LR', help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--batch_size', default=32, type=int, help='batchsize')
parser.add_argument('--network', default='mobilenet', type=str, help='network architecture')
parser.add_argument('--teacher', default='resnet50', type=str, help='network architecture')
parser.add_argument('--data_root', default='/dataset/MIT_scenes', type=str, help='Path to ImageNet')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(e) for e in gpu_nums)


max_epoch = args.max_epoch - args.distill_epoch

batch_size = args.batch_size
base_lr = args.lr
distill_epoch = args.distill_epoch

use_cuda = torch.cuda.is_available()

# Load dataset
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

trainset = torchvision.datasets.ImageFolder(root=os.path.join(args.data_root, 'train'), transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

testset = torchvision.datasets.ImageFolder(root=os.path.join(args.data_root, 'test'), transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)


# Teacher model
if args.teacher is 'resnet50':
    t_net = resnet50_imagenet(pretrained=True)
else:
    raise AssertionError("Undefined teacher network architecture")

# Student model
if args.network.startswith('mobilenetV2') and args.network.endswith('mobilenetV2'):
    s_net = mobilenetv2(pretrained=args.pretrained)
    s_net.classifier[-1] = nn.Linear(s_net.last_channel, 67)
    s_net.classifier[-1].weight.data.normal_(0, 0.01)
    s_net.classifier[-1].bias.data.zero_()

    distill_net = AB_distill_Resnet2mobilenetV2(t_net, s_net, args.batch_size, len(gpu_nums), args.DTL, args.loss_multiplier)
elif args.network.startswith('mobilenet') and args.network.endswith('mobilenet'):
    s_net = mobilenet(pretrained=args.pretrained)
    s_net.fc = nn.Linear(1024, 67)
    s_net.fc.weight.data.normal_(0, 0.01)
    s_net.fc.bias.data.zero_()

    distill_net = AB_distill_Resnet2mobilenet(t_net, s_net, args.batch_size, len(gpu_nums), args.DTL, args.loss_multiplier)
else:
    raise AssertionError("Undefined student network architecture")



if use_cuda:
    s_net = torch.nn.DataParallel(s_net).cuda()
    distill_net = torch.nn.DataParallel(distill_net).cuda()
    cudnn.benchmark = True

criterion_CE = nn.CrossEntropyLoss()

# Distillation
def Distillation(distill_net, epoch, withCE=False):
    epoch_start_time = time.time()
    print('\nDistillation Epoch: %d' % epoch)

    distill_net.train()
    distill_net.module.s_net.train()
    distill_net.module.t_net.eval()

    train_loss = 0
    train_loss1 = 0
    train_loss2 = 0
    train_loss3 = 0
    train_loss4 = 0
    correct = 0
    total = 0

    global optimizer
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)

        distill_net.module.batch_size = inputs.shape[0]
        outputs = distill_net(inputs, targets)

        loss = outputs[:, 0].sum()

        if args.DTL is True:
            loss += outputs[:, 2].sum()

        if withCE is True:
            loss += outputs[:, 1].sum()
            correct += outputs[:, 7].sum().item()
            total += targets.size(0)

        loss_AT1 = outputs[:, 3].mean()
        loss_AT2 = outputs[:, 4].mean()
        loss_AT3 = outputs[:, 5].mean()
        loss_AT4 = outputs[:, 6].mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_loss1 += loss_AT1.item()
        train_loss2 += loss_AT2.item()
        train_loss3 += loss_AT3.item()
        train_loss4 += loss_AT4.item()

        b_idx = batch_idx

    print('Train \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('layer1_activation similarity %.1f%%' % (100 * (1 - train_loss1 / (b_idx+1))))
    print('layer2_activation similarity %.1f%%' % (100 * (1 - train_loss2 / (b_idx+1))))
    print('layer3_activation similarity %.1f%%' % (100 * (1 - train_loss3 / (b_idx+1))))
    print('layer4_activation similarity %.1f%%' % (100 * (1 - train_loss4 / (b_idx+1))))

    return train_loss1 / (b_idx+1), train_loss2 / (b_idx+1), train_loss3 / (b_idx+1)


# Training with DTL(Distillation in Transfer Learning) loss
def train_DTL(distill_net, epoch):
    epoch_start_time = time.time()
    print('\nClassification training Epoch: %d' % epoch)
    distill_net.train()
    distill_net.module.s_net.train()
    distill_net.module.t_net.eval()
    train_loss = 0
    correct = 0
    total = 0
    global optimizer
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)

        distill_net.module.batch_size = inputs.shape[0]
        outputs = distill_net(inputs, targets)

        # CE loss
        loss = outputs[:, 1].sum()

        if args.DTL:
            # DTL loss
            loss += outputs[:, 2].sum()

        correct += outputs[:, 7].sum().item()
        total += targets.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        b_idx = batch_idx

    print('Train \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (b_idx + 1), 100. * correct / total, correct, total))

    return train_loss / (b_idx + 1)

# Training
def train(net, epoch):
    epoch_start_time = time.time()
    print('\nClassification training Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    global optimizer
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion_CE(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float()
        b_idx = batch_idx


    print('Train \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (b_idx + 1), 100. * correct / total, correct, total))
    return train_loss / (b_idx + 1)

# Test
def test(net, epoch, save=False):
    epoch_start_time = time.time()
    net.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    test_loss = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion_CE(outputs, targets)

        test_loss += loss.item()
        err1, err5 = accuracy(outputs.data, targets, topk=(1, 5))
        top1.update(err1[0], inputs.size(0))
        top5.update(err5[0], inputs.size(0))
        b_idx = batch_idx

    print('Test \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Top 1-err: %.3f%%, Top 5-err: %.3f%%' % (test_loss / (b_idx + 1), top1.avg, top5.avg))

    return test_loss / (b_idx + 1), top1.avg, top5.avg

# Learning rate scheduling
def adjust_learning_rate(optimizer, epoch):
    if epoch < max_epoch / 2:
        lr = base_lr
    else:
        lr = base_lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Distillation (Initialization)
optimizer = optim.SGD([{'params': s_net.parameters()},
                       {'params': distill_net.module.Connectors.parameters()}], lr=0.1, nesterov=True, momentum=args.momentum, weight_decay=args.weight_decay)

for epoch in range(1, int(distill_epoch) + 1):
    Distillation(distill_net, epoch)

# Cross-entropy training
distill_net.module.stage1 = False
optimizer = optim.SGD([{'params': s_net.parameters()},
                       {'params': distill_net.module.Connectfc.parameters()}], lr=base_lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
train_loss = 0
for epoch in range(1, max_epoch+1):
    adjust_learning_rate(optimizer, epoch - 1)

    if args.DTL:
        train_loss = train_DTL(distill_net, epoch)
    else:
        train_loss = train(s_net, epoch)

    if epoch % 5 is 0:
        test_loss, top1, top5 = test(s_net, epoch, save=True)

print('\nFinal Top1 : %.3f%%, Top5 : %.3f%%' % (top1, top5))
