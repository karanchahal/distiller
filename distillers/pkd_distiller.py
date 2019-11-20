# Original Repo:
# https://github.com/intersun/PKD-for-BERT-Model-Compression
# @article{sun2019patient,
# title={Patient Knowledge Distillation for BERT Model Compression},
# author={Sun, Siqi and Cheng, Yu and Gan, Zhe and Liu, Jingjing},
# journal={arXiv preprint arXiv:1908.09355},
# year={2019}
# }

import copy
from trainer import load_checkpoint, Trainer
import torch.nn.functional as F
import torch.nn as nn
import torch


class PKDTrainer(Trainer):
    def __init__(self, s_net, t_net, train_config):
        super(PKDTrainer, self).__init__(s_net, train_config)
        # the student net is the base net
        self.s_net = self.net
        self.t_net = t_net
        # set the teacher net into evaluation mode
        self.t_net.eval()
        self.t_net.train(mode=False)

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
        feature_output, pool_s = self.s_net(data, is_feat=True)
        feature_output_t, pool_t = self.t_net(data, is_feat=True)
        student_patience = torch.cat(feature_output, dim=1)
        teacher_patience = torch.cat(feature_output_t, dim=1)

        loss_KD = self.distill_loss(pool_s, pool_t, target)

        pt_loss = beta * \
            self.patience_loss(teacher_patience, student_patience)
        loss = loss_KD + pt_loss

        loss.backward()
        self.optimizer.step()
        return loss


def run_pkd_distillation(s_net, t_net, **params):

    # Student training
    # Define loss and the optimizer
    print("---------- Training PKD Student -------")
    student_name = params["s_name"]
    s_train_config = copy.deepcopy(params)
    s_train_config["name"] = student_name
    s_trainer = PKDTrainer(s_net, t_net=t_net,
                           train_config=s_train_config)
    best_s_acc = s_trainer.train()

    return best_s_acc
