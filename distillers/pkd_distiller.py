# Original Repo:
# https://github.com/intersun/PKD-for-BERT-Model-Compression
# @article{sun2019patient,
# title={Patient Knowledge Distillation for BERT Model Compression},
# author={Sun, Siqi and Cheng, Yu and Gan, Zhe and Liu, Jingjing},
# journal={arXiv preprint arXiv:1908.09355},
# year={2019}
# }

import torch.nn.functional as F
import torch
from trainer import KDTrainer
import util

SUPPORTED = ["resnet8", "resnet14", "resnet20", "resnet26",
             "resnet32", "resnet44", "resnet56", "resnet10",
             "resnet18", "resnet34", "resnet50", "resnet101",
             "resnet152", ]


class PKDTrainer(KDTrainer):

    def patience_loss(self, teacher_patience, student_patience,
                      normalized_patience=False):
        # n_batch = teacher_patience.shape[0]
        if normalized_patience:
            teacher_patience = F.normalize(teacher_patience, p=2, dim=2)
            student_patience = F.normalize(student_patience, p=2, dim=2)
        return F.mse_loss(teacher_patience.float(), student_patience.float())

    def calculate_loss(self, data, target):
        beta = 10
        s_feats, s_pool, s_out = self.s_net(data, is_feat=True)
        t_feats, t_pool, t_out = self.t_net(data, is_feat=True)
        s_feats = [feat.view(feat.size(0), -1) for feat in s_feats]
        t_feats = [feat.view(feat.size(0), -1) for feat in t_feats]
        student_patience = torch.cat(s_feats, dim=1)
        teacher_patience = torch.cat(t_feats, dim=1)

        loss_KD = self.kd_loss(s_out, t_out, target)

        pt_loss = beta * \
            self.patience_loss(teacher_patience, student_patience)
        loss = loss_KD + pt_loss

        loss.backward()
        self.optimizer.step()
        return s_out, loss


def run_pkd_distillation(s_net, t_net, **params):

    # check if this technique supports these kinds of models
    models = [params["student_name"], params["teacher_name"]]
    if not util.check_support(models, SUPPORTED):
        return 0.0

    # Student training
    # Define loss and the optimizer
    print("---------- Training PKD Student -------")
    s_trainer = PKDTrainer(s_net, t_net=t_net, config=params)
    best_s_acc = s_trainer.train()

    return best_s_acc
