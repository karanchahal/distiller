import torch
import torch.nn as nn
from trainer import BaseTrainer

DEPTH = 2


def get_layer_types(feat_layers):
    conv_layers = []
    for layer in feat_layers:
        if not isinstance(layer, nn.Linear):
            conv_layers.append(layer)
    return conv_layers


def get_net_info(net, feats_as_module=False):
    device = next(net.parameters()).device
    if isinstance(net, nn.DataParallel):
        net = net.module
    layers = list(net.children())
    feat_layers = get_layer_types(layers)
    linear = layers[-1]
    channels = []
    input_size = [[3, 32, 32]]
    x = [torch.rand(2, *in_size) for in_size in input_size]
    x = torch.Tensor(*x).to(device)
    for layer in feat_layers:
        x = layer(x)
        channels.append(x.shape[1:])
    if feats_as_module:
        return nn.ModuleList(feat_layers), linear, channels
    return feat_layers, linear, channels


def get_layers(layers, x, lasts=[]):
    layer_feats = []
    outs = []
    out = x
    for layer in layers:
        out = nn.functional.relu(layer(out))
        layer_feats.append(out)
    for last in lasts:
        out = last(out)
        outs.append(out)
    return layer_feats, outs


def compute_last_layer(linear, last_channel):
    # assume that h_in and w_in are equal...
    c_in = last_channel[0]
    h_in = last_channel[1]
    w_in = last_channel[2]
    flat_size = c_in * h_in * w_in
    pooling = int((flat_size / linear.in_features)**(0.5))
    modules = [nn.AvgPool2d(pooling), nn.Flatten(), linear]
    return nn.ModuleList(modules)


class Distiller(nn.Module):
    def __init__(self, s_net, t_net, batch_size=128, device="cuda"):
        super(Distiller, self).__init__()

        self.s_feat_layers, self.s_linear, s_channels = get_net_info(
            s_net, True)
        self.t_feat_layers, self.t_linear, t_channels = get_net_info(t_net)
        self.s_last = compute_last_layer(self.s_linear, s_channels[-1])
        self.t_last = compute_last_layer(self.t_linear, t_channels[-1])
        # freeze the teacher completely
        for t_layer in self.t_feat_layers:
            t_layer.requires_grad = False
        for t_layer in self.t_last:
            t_layer.requires_grad = False

    def compute_feature_loss(self, s_feats, t_feats):
        feature_loss = 0.0
        depth = len(t_feats)
        for idx in range(depth):
            s_idx = idx
            if idx > (len(s_feats) - 1):
                s_idx = len(s_feats) - 1
            s_feat = s_feats[s_idx]
            t_feat = t_feats[idx]
            s_c = s_feat.shape[1]
            s_h = s_feat.shape[2]
            s_w = s_feat.shape[3]
            t_c = t_feat.shape[1]
            t_h = t_feat.shape[2]
            t_w = t_feat.shape[3]
            if t_c > s_c:
                c_ratio = t_c / s_c
                h = int(c_ratio * t_h / 2)
                t_feat = t_feat.view(t_feat.shape[0], s_c, h, -1)
                out_shape = (s_h, s_w)
                t_feat = nn.functional.adaptive_avg_pool2d(t_feat, out_shape)

            feature_loss += nn.functional.mse_loss(s_feat, t_feat)
        return feature_loss

    def forward(self, x, targets=None, is_loss=False):
        s_feats, s_outs = get_layers(self.s_feat_layers, x, self.s_last)
        t_feats, t_outs = get_layers(self.t_feat_layers, x, self.t_last)
        if is_loss:
            feature_loss = 0.0
            feature_loss += self.compute_feature_loss(s_feats, t_feats)
            feature_loss += nn.functional.mse_loss(s_outs[-1], t_outs[-1])
            return s_outs[-1], feature_loss
        return s_outs[-1]


class FDTrainer(BaseTrainer):
    def __init__(self, s_net, t_net, config):
        super(FDTrainer, self).__init__(s_net, config)
        self.t_net = t_net

    def distill_loss(self, out_s, out_t, targets):
        lambda_ = self.config["lambda_student"]
        T = self.config["T_student"]
        # Standard Learning Loss ( Classification Loss)
        loss = self.loss_fun(out_s, targets)

        # Knowledge Distillation Loss
        student_max = nn.functional.log_softmax(out_s / T, dim=1)
        teacher_max = nn.functional.softmax(out_t / T, dim=1)
        loss_KD = nn.KLDivLoss(reduction="batchmean")(student_max, teacher_max)
        loss = (1 - lambda_) * loss + lambda_ * T * T * loss_KD
        return loss

    def calculate_loss(self, data, targets):
        s_out, feature_loss = self.net(data, targets, is_loss=True)
        loss_CE = self.loss_fun(s_out, targets)
        loss = feature_loss + loss_CE
        loss.backward()
        self.optimizer.step()
        return s_out, loss


def run_fd_distillation(s_net, t_net, **params):

    # Student training
    print("---------- Training FD Student -------")
    s_net = Distiller(s_net, t_net).to(params["device"])
    total_params = sum(p.numel() for p in s_net.parameters())
    print(f"FD distiller total parameters: {total_params}")
    s_trainer = FDTrainer(s_net, t_net, config=params)
    best_s_acc = s_trainer.train()

    return best_s_acc
