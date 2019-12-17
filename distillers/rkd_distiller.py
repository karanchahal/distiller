# Original Repo:
# https://github.com/lenscloth/RKD
# @inproceedings{park2019relational,
#  title={Relational Knowledge Distillation},
#  author={Park, Wonpyo and Kim, Dongju and Lu, Yan and Cho, Minsu},
#  booktitle={Proceedings of the IEEE Conference on Computer Vision
#  and Pattern Recognition},
#  pages={3967--3976},
#  year={2019}
# }

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer import KDTrainer
import util


BIG_NUMBER = 1e12


SUPPORTED = ["resnet8", "resnet14", "resnet20", "resnet26",
             "resnet32", "resnet44", "resnet56", "resnet10",
             "resnet18", "resnet34", "resnet50", "resnet101",
             "resnet152", ]


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class _Triplet(nn.Module):
    def __init__(self, p=2, margin=0.2, sampler=None, reduce=True, size_average=True):
        super().__init__()
        self.p = p
        self.margin = margin

        # update distance function accordingly
        self.sampler = sampler
        self.sampler.dist_func = lambda e: pdist(e, squared=(p == 2))

        self.reduce = reduce
        self.size_average = size_average

    def forward(self, embeddings, labels):
        anchor_idx, pos_idx, neg_idx = self.sampler(embeddings, labels)
        anchor_embed = embeddings[anchor_idx]
        positive_embed = embeddings[pos_idx]
        negative_embed = embeddings[neg_idx]

        loss = F.triplet_margin_loss(anchor_embed,
                                     positive_embed, negative_embed,
                                     margin=self.margin, p=self.p, reduction='none')

        if not self.reduce:
            return loss

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class L2Triplet(_Triplet):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__(p=2, margin=margin, sampler=sampler)


class L1Triplet(_Triplet):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__(p=1, margin=margin, sampler=sampler)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__()
        self.margin = margin
        self.sampler = sampler

    def forward(self, embeddings, labels):
        anchor_idx, pos_idx, neg_idx = self.sampler(embeddings, labels)

        anchor_embed = embeddings[anchor_idx]
        positive_embed = embeddings[pos_idx]
        negative_embed = embeddings[neg_idx]

        pos_loss = (F.pairwise_distance(
            anchor_embed, positive_embed, p=2)).pow(2)
        neg_loss = (self.margin - F.pairwise_distance(anchor_embed,
                                                      negative_embed, p=2)).clamp(min=0).pow(2)

        loss = torch.cat((pos_loss, neg_loss))
        return loss.mean()


class HardDarkRank(nn.Module):
    def __init__(self, alpha=3, beta=3, permute_len=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.permute_len = permute_len

    def forward(self, student, teacher):
        score_teacher = -1 * self.alpha * \
            pdist(teacher, squared=False).pow(self.beta)
        score_student = -1 * self.alpha * \
            pdist(student, squared=False).pow(self.beta)

        permute_idx = score_teacher.sort(dim=1, descending=True)[
            1][:, 1:(self.permute_len + 1)]
        ordered_student = torch.gather(score_student, 1, permute_idx)

        log_prob = (ordered_student - torch.stack([torch.logsumexp(
            ordered_student[:, i:], dim=1) for i in range(permute_idx.size(1))], dim=1)).sum(dim=1)
        loss = (-1 * log_prob).mean()

        return loss


class AttentionTransfer(nn.Module):
    def forward(self, student, teacher):
        s_attention = F.normalize(student.pow(
            2).mean(1).view(student.size(0), -1))

        with torch.no_grad():
            t_attention = F.normalize(teacher.pow(
                2).mean(1).view(teacher.size(0), -1))

        return (s_attention - t_attention).pow(2).mean()


class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        return loss


class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss


def pos_neg_mask(labels):
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) * \
               (1 - torch.eye(labels.size(0), dtype=torch.uint8, device=labels.device))
    neg_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)) * \
               (1 - torch.eye(labels.size(0), dtype=torch.uint8, device=labels.device))

    return pos_mask, neg_mask


class _Sampler(nn.Module):
    def __init__(self, dist_func=pdist):
        self.dist_func = dist_func
        super().__init__()

    def forward(self, embeddings, labels):
        raise NotImplementedError


class AllPairs(_Sampler):
    def forward(self, embeddings, labels):
        with torch.no_grad():
            pos_mask, neg_mask = pos_neg_mask(labels)
            pos_pair_idx = pos_mask.nonzero()

            apns = []
            for pair_idx in pos_pair_idx:
                anchor_idx = pair_idx[0]
                neg_indices = neg_mask[anchor_idx].nonzero()

                apn = torch.cat((pair_idx.unsqueeze(0).repeat(
                    len(neg_indices), 1), neg_indices), dim=1)
                apns.append(apn)
            apns = torch.cat(apns, dim=0)
            anchor_idx = apns[:, 0]
            pos_idx = apns[:, 1]
            neg_idx = apns[:, 2]

        return anchor_idx, pos_idx, neg_idx


class RandomNegative(_Sampler):
    def forward(self, embeddings, labels):
        with torch.no_grad():
            pos_mask, neg_mask = pos_neg_mask(labels)

            pos_pair_index = pos_mask.nonzero()
            anchor_idx = pos_pair_index[:, 0]
            pos_idx = pos_pair_index[:, 1]
            neg_index = torch.multinomial(
                neg_mask.float()[anchor_idx], 1).squeeze(1)

        return anchor_idx, pos_idx, neg_index


class HardNegative(_Sampler):
    def forward(self, embeddings, labels):
        with torch.no_grad():
            pos_mask, neg_mask = pos_neg_mask(labels)
            dist = self.dist_func(embeddings)

            pos_pair_index = pos_mask.nonzero()
            anchor_idx = pos_pair_index[:, 0]
            pos_idx = pos_pair_index[:, 1]

            neg_dist = (neg_mask.float() * dist)
            neg_dist[neg_dist <= 0] = BIG_NUMBER
            neg_idx = neg_dist.argmin(dim=1)[anchor_idx]

        return anchor_idx, pos_idx, neg_idx


class SemiHardNegative(_Sampler):
    def forward(self, embeddings, labels):
        with torch.no_grad():
            dist = self.dist_func(embeddings)
            pos_mask, neg_mask = pos_neg_mask(labels)
            neg_dist = dist * neg_mask.float()

            pos_pair_idx = pos_mask.nonzero()
            anchor_idx = pos_pair_idx[:, 0]
            pos_idx = pos_pair_idx[:, 1]

            tiled_negative = neg_dist[anchor_idx]
            satisfied_neg = (tiled_negative > dist[pos_mask].unsqueeze(
                1)) * neg_mask[anchor_idx]
            """
            When there is no negative pair that its distance bigger than positive pair,
            then select negative pair with largest distance.
            """
            unsatisfied_neg = (satisfied_neg.sum(dim=1) ==
                               0).unsqueeze(1) * neg_mask[anchor_idx]

            tiled_negative = (satisfied_neg.float() * tiled_negative) - \
                (unsatisfied_neg.float() * tiled_negative)
            tiled_negative[tiled_negative == 0] = BIG_NUMBER
            neg_idx = tiled_negative.argmin(dim=1)

        return anchor_idx, pos_idx, neg_idx


class DistanceWeighted(_Sampler):
    cut_off = 0.5
    nonzero_loss_cutoff = 1.4
    """
    Distance Weighted loss assume that embeddings are normalized py 2-norm.
    """

    def forward(self, embeddings, labels):
        with torch.no_grad():
            embeddings = F.normalize(embeddings, dim=1, p=2)
            pos_mask, neg_mask = pos_neg_mask(labels)
            pos_pair_idx = pos_mask.nonzero()
            anchor_idx = pos_pair_idx[:, 0]
            pos_idx = pos_pair_idx[:, 1]

            d = embeddings.size(1)
            dist = (pdist(embeddings, squared=True) +
                    torch.eye(embeddings.size(0),
                              device=embeddings.device,
                              dtype=torch.float32)).sqrt()
            dist = dist.clamp(min=self.cut_off)

            log_weight = ((2.0 - d) * dist.log() - ((d - 3.0) / 2.0)
                          * (1.0 - 0.25 * (dist * dist)).log())
            weight = (log_weight - log_weight.max(dim=1,
                                                  keepdim=True)[0]).exp()
            weight = weight * \
                (neg_mask * (dist < self.nonzero_loss_cutoff)).float()

            weight = weight + \
                ((weight.sum(dim=1, keepdim=True) == 0) * neg_mask).float()
            weight = weight / (weight.sum(dim=1, keepdim=True))
            weight = weight[anchor_idx]
            neg_idx = torch.multinomial(weight, 1).squeeze(1)

        return anchor_idx, pos_idx, neg_idx


class RKDTrainer(KDTrainer):
    def __init__(self, s_net, t_net, config):
        super(RKDTrainer, self).__init__(s_net, t_net, config)
        # the student net is the base net
        self.s_net = self.net
        self.t_net = t_net

        self.triplet_ratio = 0.0
        self.triplet_margin = 0.2
        self.dist_ratio = 25.0
        self.angle_ratio = 50.0
        self.at_ratio = 50

        self.dark_ratio = 0.0
        self.dark_alpha = 2.0
        self.dark_beta = 3.0
        self.triplet_sample = DistanceWeighted
        self.dist_criterion = RkdDistance()
        self.angle_criterion = RKdAngle()
        self.dark_criterion = HardDarkRank(alpha=self.dark_alpha,
                                           beta=self.dark_beta)
        self.triplet_criterion = L2Triplet(
            sampler=self.triplet_sample(),
            margin=self.triplet_margin)
        self.at_criterion = AttentionTransfer()

    def calculate_loss(self, data, target):
        t_feats, t_pool, t_out = self.t_net(data, is_feat=True)
        s_feats, s_pool, s_out = self.s_net(data, is_feat=True)

        loss = self.kd_loss(s_out, t_out, target)

        at_loss = 0
        # technically we should use every layer expect the first here
        # not sure if this will change the outcome significantly
        for idx, s_feat in enumerate(s_feats, start=1):
            at_loss += self.at_ratio * \
                self.at_criterion(s_feat, t_feats[idx])
        dist_loss = self.dist_ratio * self.dist_criterion(s_pool, t_pool)
        angle_loss = self.angle_ratio * self.angle_criterion(s_pool, t_pool)
        dark_loss = self.dark_ratio * self.dark_criterion(s_out, t_out)
        loss += dist_loss + angle_loss + dark_loss
        loss.backward()
        self.optimizer.step()
        return s_out, loss


def run_rkd_distillation(s_net, t_net, **params):

    # check if this technique supports these kinds of models
    models = [params["student_name"], params["teacher_name"]]
    if not util.check_support(models, SUPPORTED):
        return 0.0

    # Student training
    print("---------- Training RKD Student -------")
    params = params.copy()
    s_trainer = RKDTrainer(s_net, t_net=t_net,
                           config=params)
    best_s_acc = s_trainer.train()

    return best_s_acc
