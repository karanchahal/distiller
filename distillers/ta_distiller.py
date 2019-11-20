# Original Repo:
# https://github.com/imirzadeh/Teacher-Assistant-Knowledge-Distillation
# @article{mirzadeh2019improved,
#  title={Improved knowledge distillation via teacher assistant:
#  Bridging the gap between student and teacher},
#  author={Mirzadeh, Seyed-Iman and Farajtabar, Mehrdad and Li, Ang and
#  Ghasemzadeh, Hassan},
#  journal={arXiv preprint arXiv:1902.03393},
#  year={2019}
# }


import copy

from trainer import load_checkpoint, KDTrainer


def run_teacher_assistant(s_net, ta_net, t_net, **params):

    # Teaching Assistant training
    # Define loss and the optimizer

    print("---------- Training TA -------")

    ta_train_config = copy.deepcopy(params)
    ta_name = params["ta_name"]
    trial_id = params["trial_id"]
    best_ta = f"{ta_name}_{trial_id}_best.pth.tar"
    ta_train_config["name"] = ta_name
    ta_trainer = KDTrainer(ta_net, t_net=t_net,
                           train_config=ta_train_config)
    best_ta_acc = ta_trainer.train()
    ta_net = load_checkpoint(ta_net, best_ta)

    # Student training
    # Define loss and the optimizer
    print("---------- Training TA Student -------")
    student_name = params["s_name"]
    s_train_config = copy.deepcopy(params)
    s_train_config["name"] = student_name
    s_trainer = KDTrainer(s_net, t_net=ta_net,
                          train_config=s_train_config)
    best_s_acc = s_trainer.train()

    print(f"Final results ta {ta_name}: {best_ta_acc}")
    return best_s_acc
