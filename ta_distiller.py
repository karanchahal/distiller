import copy
import torch.nn as nn
import torch.nn.functional as F

from trainer import load_checkpoint, Trainer


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
        lambda_ = self.config["lambda_student"]
        T = self.config["T_student"]
        output = self.s_net(data)

        # Standard Learning Loss ( Classification Loss)
        loss = self.loss_fun(output, target)

        # Knowledge Distillation Loss
        teacher_outputs = self.t_net(data)
        student_max = F.log_softmax(output / T, dim=1)
        teacher_max = F.softmax(teacher_outputs / T, dim=1)
        loss_KD = nn.KLDivLoss()(student_max, teacher_max)
        loss = (1 - lambda_) * loss + lambda_ * T * T * loss_KD
        loss.backward()
        self.optimizer.step()
        return loss


def run_teacher_assistant(s_net, ta_net, t_net, **params):

    # Teaching Assistant training
    # Define loss and the optimizer

    print("---------- Training TA -------")

    ta_train_config = copy.deepcopy(params)
    ta_name = params["ta_name"]
    trial_id = params["trial_id"]
    best_ta = f"{ta_name}_{trial_id}_best.pth.tar"
    ta_train_config["name"] = ta_name
    ta_trainer = TrainManager(ta_net, t_net=t_net,
                              train_config=ta_train_config)
    best_ta_acc = ta_trainer.train()
    ta_net = load_checkpoint(ta_net, best_ta)

    # Student training
    # Define loss and the optimizer
    print("---------- Training TA Student -------")
    student_name = params["s_name"]
    s_train_config = copy.deepcopy(params)
    s_train_config["name"] = student_name
    s_trainer = TrainManager(s_net, t_net=ta_net,
                             train_config=s_train_config)
    best_s_acc = s_trainer.train()

    print(f"Final results ta {ta_name}: {best_ta_acc}")
    return best_s_acc
