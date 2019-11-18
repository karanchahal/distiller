import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer import load_checkpoint, Trainer


BIG_NUMBER = 1e12


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

        loss = F.triplet_margin_loss(anchor_embed, positive_embed, negative_embed,
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


class FitNet(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.transform = nn.Conv2d(in_feature, out_feature, 1, bias=False)
        self.transform.weight.data.uniform_(-0.005, 0.005)

    def forward(self, student, teacher):
        if student.dim() == 2:
            student = student.unsqueeze(2).unsqueeze(3)
            teacher = teacher.unsqueeze(2).unsqueeze(3)

        return (self.transform(student) - teacher).pow(2).mean()


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


triplet_ratio = 0.0
triplet_margin = 0.2
dist_ratio = 1.0
angle_ratio = 2.0
at_ratio = 0.0

dark_ratio = 0.0
dark_alpha = 2.0
dark_beta = 3.0

l2normalize = True
teacher_embedding_size = 128
teacher_l2normalize = True
embedding_size = 128

triplet_sample = DistanceWeighted
dist_criterion = RkdDistance()
angle_criterion = RKdAngle()
dark_criterion = HardDarkRank(alpha=dark_alpha,
                              beta=dark_beta)
triplet_criterion = L2Triplet(
    sampler=triplet_sample(),
    margin=triplet_margin)
at_criterion = AttentionTransfer()


def othernorm(x):
    other_mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).cuda()
    other_std = torch.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).cuda()
    x = (x - other_mean) / other_std
    return x


class TrainManager(Trainer):
    def __init__(self, s_net, t_net, train_config):
        super(TrainManager, self).__init__(s_net, train_config)
        # the student net is the base net
        self.s_net = self.net
        self.t_net = t_net
        # set the teacher net into evaluation mode
        self.t_net.eval()
        self.t_net.train(mode=False)

    def calculate_loss(self, data, target):
        t_b1, t_b2, t_b3, t_b4, t_pool, t_e = self.t_net(othernorm(data), True)

        b1, b2, b3, b4, pool, e = self.s_net(othernorm(data), True)
        at_loss = at_ratio * \
            (at_criterion(b2, t_b2) +
             at_criterion(b3, t_b3) + at_criterion(b4, t_b4))

        triplet_loss = triplet_ratio * triplet_criterion(e, target)
        dist_loss = dist_ratio * dist_criterion(e, t_e)
        angle_loss = angle_ratio * angle_criterion(e, t_e)
        dark_loss = dark_ratio * dark_criterion(e, t_e)

        loss = triplet_loss + dist_loss + angle_loss + dark_loss + at_loss
        loss.backward()
        self.optimizer.step()
        return loss


class LinearEmbedding(nn.Module):
    def __init__(self, base, output_size=512, embedding_size=128, normalize=True):
        super(LinearEmbedding, self).__init__()
        self.base = base
        self.linear = nn.Linear(output_size, embedding_size)
        self.normalize = normalize

    def forward(self, x, get_ha=False):
        if get_ha:
            b1, b2, b3, b4, pool = self.base(x, True)
        else:
            pool = self.base(x)

        pool = pool.view(x.size(0), -1)
        embedding = self.linear(pool)

        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=1)

        if get_ha:
            return b1, b2, b3, b4, pool, embedding

        return embedding


def run_relational_kd(s_net, t_net, **params):

    s_net = LinearEmbedding(s_net,
                            output_size=10,
                            embedding_size=embedding_size,
                            normalize=l2normalize == 'true')
    s_net = s_net.to(params["device"])

    t_net = LinearEmbedding(t_net,
                            output_size=10,
                            embedding_size=teacher_embedding_size,
                            normalize=teacher_l2normalize)
    t_net = t_net.to(params["device"])
    # Student training
    # Define loss and the optimizer
    print("---------- Training TA Student -------")
    student_name = params["s_name"]
    s_train_config = copy.deepcopy(params)
    s_train_config["name"] = student_name
    s_trainer = TrainManager(s_net, t_net=t_net,
                             train_config=s_train_config)
    best_s_acc = s_trainer.train()

    return best_s_acc
