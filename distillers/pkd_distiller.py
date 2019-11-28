# Original Repo:
# https://github.com/intersun/PKD-for-BERT-Model-Compression
# @article{sun2019patient,
# title={Patient Knowledge Distillation for BERT Model Compression},
# author={Sun, Siqi and Cheng, Yu and Gan, Zhe and Liu, Jingjing},
# journal={arXiv preprint arXiv:1908.09355},
# year={2019}
# }

import torch.nn.functional as F
import torch.nn as nn
import torch
from trainer import Trainer


class PKDTrainer(Trainer):
    def __init__(self, s_net, t_net, train_config):
        super(PKDTrainer, self).__init__(s_net, train_config)
        # the student net is the base net
        self.s_net = self.net
        self.t_net = t_net

    def patience_loss(self, teacher_patience, student_patience,
                      normalized_patience=False):
        # n_batch = teacher_patience.shape[0]
        if normalized_patience:
            teacher_patience = F.normalize(teacher_patience, p=2, dim=2)
            student_patience = F.normalize(student_patience, p=2, dim=2)
        return F.mse_loss(teacher_patience.float(), student_patience.float())

    def distill_loss(self, pool_s, pool_t, target):
        lambda_ = self.config["lambda_student"]
        T = self.config["T_student"]
        # Standard Learning Loss ( Classification Loss)
        loss = self.loss_fun(pool_s, target)

        # Knowledge Distillation Loss
        student_max = F.log_softmax(pool_s / T, dim=1)
        teacher_max = F.softmax(pool_t / T, dim=1)
        loss_KD = nn.KLDivLoss(reduction="batchmean")(student_max, teacher_max)
        loss = (1 - lambda_) * loss + lambda_ * T * T * loss_KD
        return loss

    def calculate_loss(self, data, target):
        beta = 10
        s_feats, s_pool, s_out = self.s_net(data, is_feat=True)
        t_feats, t_pool, t_out = self.t_net(data, is_feat=True)
        s_feats = [feat.view(feat.size(0), -1) for feat in s_feats]
        t_feats = [feat.view(feat.size(0), -1) for feat in t_feats]
        student_patience = torch.cat(s_feats, dim=1)
        teacher_patience = torch.cat(t_feats, dim=1)

        loss_KD = self.distill_loss(s_out, t_out, target)

        pt_loss = beta * \
            self.patience_loss(teacher_patience, student_patience)
        loss = loss_KD + pt_loss

        loss.backward()
        self.optimizer.step()
        return s_out, loss


def run_pkd_distillation(s_net, t_net, **params):

    # Student training
    # Define loss and the optimizer
    print("---------- Training PKD Student -------")
    s_trainer = PKDTrainer(s_net, t_net=t_net, train_config=params)
    best_s_acc = s_trainer.train()

    return best_s_acc
