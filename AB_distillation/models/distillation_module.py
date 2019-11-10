import torch
import torch.nn as nn
import torch.nn.functional as F

import math


##########################
# For CIFAR
###########################

class Active_Soft_WRN_norelu(nn.Module):
    def __init__(self, t_net, s_net):
        super(Active_Soft_WRN_norelu, self).__init__()

        # Connection layers
        if t_net.nChannels == s_net.nChannels:
            C1 = []
            C2 = []
            C3 = []
        else:
            C1 = [nn.Conv2d(int(s_net.nChannels / 4), int(t_net.nChannels / 4), kernel_size=1, stride=1, padding=0, bias=False),
                  nn.BatchNorm2d(int(t_net.nChannels / 4))]
            C2 = [nn.Conv2d(int(s_net.nChannels / 2), int(t_net.nChannels / 2), kernel_size=1, stride=1, padding=0, bias=False),
                  nn.BatchNorm2d(int(t_net.nChannels / 2))]
            C3 = [nn.Conv2d(s_net.nChannels, t_net.nChannels, kernel_size=1, stride=1, padding=0, bias=False),
                  nn.BatchNorm2d(t_net.nChannels)]

        # Weight initialize
        for m in C1 + C2 + C3:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.Connect1 = nn.Sequential(*C1)
        self.Connect2 = nn.Sequential(*C2)
        self.Connect3 = nn.Sequential(*C3)
        self.Connectors = nn.ModuleList([self.Connect1, self.Connect2, self.Connect3])

        self.t_net = t_net
        self.s_net = s_net

    def forward(self, x):

        # For teacher network
        self.res0_t = self.t_net.conv1(x)

        self.res1_t = self.t_net.block1(self.res0_t)
        self.res2_t = self.t_net.block2(self.res1_t)
        self.res3_t = self.t_net.bn1(self.t_net.block3(self.res2_t))

        out = self.t_net.relu(self.res3_t)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.t_net.nChannels)
        self.out_t = self.t_net.fc(out)

        # For student network
        self.res0 = self.s_net.conv1(x)

        self.res1 = self.s_net.block1(self.res0)
        self.res2 = self.s_net.block2(self.res1)
        self.res3 = self.s_net.block3(self.res2)

        out = self.s_net.relu(self.s_net.bn1(self.res3))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.s_net.nChannels)
        self.out_s = self.s_net.fc(out)

        # Features before ReLU
        self.res0_t = self.t_net.block1.layer[0].bn1(self.res0_t)
        self.res1_t = self.t_net.block2.layer[0].bn1(self.res1_t)
        self.res2_t = self.t_net.block3.layer[0].bn1(self.res2_t)

        self.res0 = self.s_net.block1.layer[0].bn1(self.res0)
        self.res1 = self.s_net.block2.layer[0].bn1(self.res1)
        self.res2 = self.s_net.block3.layer[0].bn1(self.res2)
        self.res3 = self.s_net.bn1(self.res3)

        return self.out_s


##########################
# ResNet 50 to MobileNet
# For transfer learning
###########################

# Designed for data parallel
class AB_distill_Resnet2mobilenetV2(nn.Module):

    # Proposed alternative loss function
    def criterion_active_L2(self, source, target, margin):
        loss = ((source + margin) ** 2 * ((source > -margin) & (target <= 0)).float() +
                (source - margin) ** 2 * ((source <= margin) & (target > 0)).float())
        return torch.abs(loss).sum()

    def __init__(self, t_net, s_net, batch_size, gpu_num, DTL, loss_multiplier):
        super(AB_distill_Resnet2mobilenetV2, self).__init__()

        self.batch_size = batch_size
        self.gpu_num = gpu_num
        self.loss_multiplier = loss_multiplier
        self.DTL = DTL
        self.expansion = 6

        # Connector layers
        C1 = [nn.Conv2d(24 * self.expansion, 256, kernel_size=1, stride=1, padding=0, bias=False),
              nn.BatchNorm2d(256)]
        C2 = [nn.Conv2d(32 * self.expansion, 512, kernel_size=1, stride=1, padding=0, bias=False),
              nn.BatchNorm2d(512)]
        C3 = [nn.Conv2d(96 * self.expansion, 1024, kernel_size=1, stride=1, padding=0, bias=False),
              nn.BatchNorm2d(1024)]
        C4 = [nn.Conv2d(1280, 2048, kernel_size=1, stride=1, padding=0, bias=False),
              nn.BatchNorm2d(2048)]

        for m in C1 + C2 + C3 + C4:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.Connect1 = nn.Sequential(*C1)
        self.Connect2 = nn.Sequential(*C2)
        self.Connect3 = nn.Sequential(*C3)
        self.Connect4 = nn.Sequential(*C4)

        self.Connectfc = nn.Linear(1280, 1000)
        self.Connectors = nn.ModuleList([self.Connect1, self.Connect2, self.Connect3, self.Connect4, self.Connectfc])

        self.t_net = t_net
        self.s_net = s_net

        self.stage1 = True
        self.criterion_CE = nn.CrossEntropyLoss(size_average=False)

    def forward(self, inputs, targets):

        # Teacher network
        res0_t = self.t_net.maxpool(self.t_net.relu(self.t_net.bn1(self.t_net.conv1(inputs))))

        res1_t = self.t_net.layer1(res0_t)
        res2_t = self.t_net.layer2(res1_t)
        res3_t = self.t_net.layer3(res2_t)
        res4_t = self.t_net.layer4(res3_t)

        out = self.t_net.relu(res4_t)
        out = self.t_net.avgpool(out)
        out = out.view(out.size(0), -1)
        out_t = self.t_net.fc(out)


        # Student network
        res1_s = self.s_net.features[0:4](inputs)
        res2_s = self.s_net.features[4:7](res1_s)
        res3_s = self.s_net.features[7:14](res2_s)
        res4_s = self.s_net.features[14:18](res3_s)

        out = self.s_net.features[18:](res4_s)
        out = out.view(-1, 1280)
        out_imagenet = self.Connectfc(out)
        out_s = self.s_net.classifier(out)

        # Features before ReLU
        res1_s = self.s_net.features[4].conv[0:2](res1_s)
        res2_s = self.s_net.features[7].conv[0:2](res2_s)
        res3_s = self.s_net.features[14].conv[0:2](res3_s)
        res4_s = self.s_net.features[18][0:2](res4_s)


        # Activation transfer loss
        loss_AT4 = ((self.Connect4(res4_s) > 0) ^ (res4_t > 0)).sum().float() / res4_t.nelement()
        loss_AT3 = ((self.Connect3(res3_s) > 0) ^ (res3_t > 0)).sum().float() / res3_t.nelement()
        loss_AT2 = ((self.Connect2(res2_s) > 0) ^ (res2_t > 0)).sum().float() / res2_t.nelement()
        loss_AT1 = ((self.Connect1(res1_s) > 0) ^ (res1_t > 0)).sum().float() / res1_t.nelement()

        loss_AT4 = loss_AT4.unsqueeze(0).unsqueeze(1)
        loss_AT3 = loss_AT3.unsqueeze(0).unsqueeze(1)
        loss_AT2 = loss_AT2.unsqueeze(0).unsqueeze(1)
        loss_AT1 = loss_AT1.unsqueeze(0).unsqueeze(1)

        # Alternative loss
        if self.stage1 is True:
            margin = 1.0
            loss = self.criterion_active_L2(self.Connect4(res4_s), res4_t.detach(), margin) / self.batch_size
            loss += self.criterion_active_L2(self.Connect3(res3_s), res3_t.detach(), margin) / self.batch_size / 2
            loss += self.criterion_active_L2(self.Connect2(res2_s), res2_t.detach(), margin) / self.batch_size / 4
            loss += self.criterion_active_L2(self.Connect1(res1_s), res1_t.detach(), margin) / self.batch_size / 8

            loss /= 1000

            loss = loss.unsqueeze(0).unsqueeze(1)
        else:
            loss = torch.zeros(1, 1).cuda()

        loss *= self.loss_multiplier

        # Cross-entropy loss
        loss_CE = self.criterion_CE(out_s, targets) / self.batch_size
        loss_CE = loss_CE.unsqueeze(0).unsqueeze(1)

        # DTL (Distillation in Transfer Learning) loss
        if self.DTL is True:
            loss_DTL = torch.mean(torch.pow((out_t - torch.mean(out_t, 1, keepdim=True)).detach()
                                           - (out_imagenet - torch.mean(out_imagenet, 1, keepdim=True)), 2)) * 10

            loss_DTL = loss_DTL.unsqueeze(0).unsqueeze(1)
        else:
            loss_DTL = torch.zeros(1,1).cuda()

        # Training accuracy
        _, predicted = torch.max(out_s.data, 1)
        correct = predicted.eq(targets.data).sum().float().unsqueeze(0).unsqueeze(1)

        # Return all losses
        return torch.cat([loss, loss_CE, loss_DTL, loss_AT1, loss_AT2, loss_AT3, loss_AT4, correct], dim=1)


# Designed for data parallel
class AB_distill_Resnet2mobilenet(nn.Module):

    # Proposed alternative loss function
    def criterion_active_L2(self, source, target, margin):
        loss = ((source + margin) ** 2 * ((source > -margin) & (target <= 0)).float() +
                (source - margin) ** 2 * ((source <= margin) & (target > 0)).float())
        return torch.abs(loss).sum()

    def __init__(self, t_net, s_net, batch_size, gpu_num, DTL, loss_multiplier):
        super(AB_distill_Resnet2mobilenet, self).__init__()

        self.batch_size = batch_size
        self.gpu_num = gpu_num
        self.loss_multiplier = loss_multiplier
        self.DTL = DTL
        self.expansion = 2

        # Connector layers
        C1 = [nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False),
              nn.BatchNorm2d(256)]
        C2 = [nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),
              nn.BatchNorm2d(512)]
        C3 = [nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False),
              nn.BatchNorm2d(1024)]
        C4 = [nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False),
              nn.BatchNorm2d(2048)]

        for m in C1 + C2 + C3 + C4:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.Connect1 = nn.Sequential(*C1)
        self.Connect2 = nn.Sequential(*C2)
        self.Connect3 = nn.Sequential(*C3)
        self.Connect4 = nn.Sequential(*C4)

        self.Connectfc = nn.Linear(1024, 1000)
        self.Connectors = nn.ModuleList([self.Connect1, self.Connect2, self.Connect3, self.Connect4, self.Connectfc])

        self.t_net = t_net
        self.s_net = s_net

        self.stage1 = True
        self.criterion_CE = nn.CrossEntropyLoss(size_average=False)

    def forward(self, x):

        inputs = x[0]
        targets = x[1]

        # Teacher network
        res0_t = self.t_net.maxpool(self.t_net.relu(self.t_net.bn1(self.t_net.conv1(inputs))))

        res1_t = self.t_net.layer1(res0_t)
        res2_t = self.t_net.layer2(res1_t)
        res3_t = self.t_net.layer3(res2_t)
        res4_t = self.t_net.layer4(res3_t)

        out = self.t_net.relu(res4_t)
        out = self.t_net.avgpool(out)
        out = out.view(out.size(0), -1)
        out_t = self.t_net.fc(out)

        # Student network
        res1_s = self.s_net.model[3][:-1](self.s_net.model[0:3](inputs))
        res2_s = self.s_net.model[5][:-1](self.s_net.model[4:5](F.relu(res1_s)))
        res3_s = self.s_net.model[11][:-1](self.s_net.model[6:11](F.relu(res2_s)))
        res4_s = self.s_net.model[13][:-1](self.s_net.model[12:13](F.relu(res3_s)))

        out = self.s_net.model[14](F.relu(res4_s))
        out = out.view(-1, 1024)
        out_imagenet = self.Connectfc(out)
        out_s = self.s_net.fc(out)

        # Activation transfer loss
        loss_AT4 = ((self.Connect4(res4_s) > 0) ^ (res4_t > 0)).sum().float() / res4_t.nelement()
        loss_AT3 = ((self.Connect3(res3_s) > 0) ^ (res3_t > 0)).sum().float() / res3_t.nelement()
        loss_AT2 = ((self.Connect2(res2_s) > 0) ^ (res2_t > 0)).sum().float() / res2_t.nelement()
        loss_AT1 = ((self.Connect1(res1_s) > 0) ^ (res1_t > 0)).sum().float() / res1_t.nelement()

        loss_AT4 = loss_AT4.unsqueeze(0).unsqueeze(1)
        loss_AT3 = loss_AT3.unsqueeze(0).unsqueeze(1)
        loss_AT2 = loss_AT2.unsqueeze(0).unsqueeze(1)
        loss_AT1 = loss_AT1.unsqueeze(0).unsqueeze(1)

        # Alternative loss
        if self.stage1 is True:
            margin = 1.0
            loss = self.criterion_active_L2(self.Connect4(res4_s), res4_t.detach(), margin) / self.batch_size
            loss += self.criterion_active_L2(self.Connect3(res3_s), res3_t.detach(), margin) / self.batch_size / 2
            loss += self.criterion_active_L2(self.Connect2(res2_s), res2_t.detach(), margin) / self.batch_size / 4
            loss += self.criterion_active_L2(self.Connect1(res1_s), res1_t.detach(), margin) / self.batch_size / 8

            loss /= 1000

            loss = loss.unsqueeze(0).unsqueeze(1)
        else:
            loss = torch.zeros(1, 1).cuda()
            
        loss *= self.loss_multiplier

        # Cross-entropy loss
        loss_CE = self.criterion_CE(out_s, targets) / self.batch_size
        loss_CE = loss_CE.unsqueeze(0).unsqueeze(1)

        # DTL (Distillation in Transfer Learning) loss
        if self.DTL is True:
            loss_DTL = torch.mean(torch.pow((out_t - torch.mean(out_t, 1, keepdim=True)).detach()
                                           - (out_imagenet - torch.mean(out_imagenet, 1, keepdim=True)), 2)) * 10
            loss_DTL = loss_DTL.unsqueeze(0).unsqueeze(1)
        else:
            loss_DTL = torch.zeros(1,1).cuda()

        # Training accuracy
        _, predicted = torch.max(out_s.data, 1)
        correct = predicted.eq(targets.data).sum().float().unsqueeze(0).unsqueeze(1)

        # Return all losses
        return torch.cat([loss, loss_CE, loss_DTL, loss_AT4, loss_AT3, loss_AT2, loss_AT1, correct], dim=1)

