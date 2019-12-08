import torch
import torch.nn as nn
from trainer import BaseTrainer


def get_layer_types(feat_layers):
    conv_layers = []
    for layer in feat_layers:
        if not isinstance(layer, nn.Linear):
            conv_layers.append(layer)
    return conv_layers


def get_net_info(net, as_module=False):
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
    if as_module:
        return nn.ModuleList(feat_layers), linear, channels
    return feat_layers, linear, channels


def set_last_layers(linear, last_channel, as_module=False):
    # assume that h_in and w_in are equal...
    c_in = last_channel[0]
    h_in = last_channel[1]
    w_in = last_channel[2]
    flat_size = c_in * h_in * w_in
    pooling = int((flat_size / linear.in_features)**(0.5))
    modules = [nn.ReLU(), nn.AvgPool2d(pooling), nn.Flatten(), linear]
    if as_module:
        return nn.ModuleList(modules)
    return modules


def get_layers(layers, x, lasts=[], use_relu=True):
    layer_feats = []
    outs = []
    out = x
    for layer in layers:
        out = layer(out)
        if use_relu:
            out = nn.functional.relu(out)
        layer_feats.append(out)
    for last in lasts:
        out = last(out)
        outs.append(out)
    return layer_feats, outs


class Distiller(nn.Module):
    def __init__(self, s_net, t_net):
        super(Distiller, self).__init__()

        self.s_feat_layers, self.s_linear, s_channels = get_net_info(
            s_net, as_module=True)
        self.t_feat_layers, self.t_linear, t_channels = get_net_info(
            t_net, as_module=False)
        self.s_last = set_last_layers(
            self.s_linear, s_channels[-1], as_module=True)
        self.t_last = set_last_layers(
            self.t_linear, t_channels[-1], as_module=False)
        self.y_tensors = []
        for s_channel in s_channels:
            shape = [s_channel[0] * s_channel[1] * s_channel[2]]
            y_tensor = torch.zeros(shape).to("cuda")
            self.y_tensors.append(y_tensor)

    def compute_feature_loss(self, s_feats, t_feats):
        feature_loss = 0.0
        t_depth = len(t_feats)
        s_depth = len(s_feats)
        diff = t_depth - s_depth
        if diff < 0:
            offset = 0
            depth = t_depth
        else:
            offset = diff
            depth = s_depth
        for idx in range(depth):
            s_feat = s_feats[idx]
            t_feat = t_feats[idx + offset]
            s_c = s_feat.shape[1]
            t_feat = t_feat.view(t_feat.shape[0], s_c, -1)
            s_feat = s_feat.view(s_feat.shape[0], s_c, -1)
            if t_feat.shape[2] > s_feat.shape[2]:
                out_shape = s_feat.shape[2]
                t_feat = nn.functional.adaptive_avg_pool1d(t_feat, out_shape)
            else:
                out_shape = t_feat.shape[2]
                s_feat = nn.functional.adaptive_avg_pool1d(s_feat, out_shape)
            s_feat = s_feat.view(s_feat.shape[0], -1)
            t_feat = t_feat.view(t_feat.shape[0], -1)

            feature_loss += nn.functional.mse_loss(
                s_feat, t_feat)
        return feature_loss / 10000

    def forward(self, x, targets=None, is_loss=False):
        s_feats, s_outs = get_layers(
            self.s_feat_layers, x, self.s_last, False)
        if is_loss:
            t_feats, t_outs = get_layers(
                self.t_feat_layers, x, self.t_last, False)
            feature_loss = 0.0
            feature_loss += self.compute_feature_loss(s_feats, t_feats)
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
        loss_ce = self.loss_fun(s_out, targets)
        loss = loss_ce + feature_loss
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
