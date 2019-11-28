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


from trainer import KDTrainer
import util


def run_takd_distillation(s_net, ta_net, t_net, **params):

    # Teaching Assistant training
    print("---------- Training TA -------")
    ta_config = params.copy()
    ta_name = ta_config["ta_name"]
    ta_config["test_name"] = f"{ta_name}_ta_trainer"
    ta_trainer = KDTrainer(ta_net, t_net=t_net, config=ta_config)
    best_ta_acc = ta_trainer.train()
    best_ta = ta_trainer.best_model_file
    ta_net = util.load_checkpoint(ta_net, best_ta)

    # Student training
    print("---------- Training TA Student -------")
    s_trainer = KDTrainer(s_net, t_net=ta_net, config=params)
    best_s_acc = s_trainer.train()

    print(f"Best results teacher {ta_name}: {best_ta_acc}")
    return best_s_acc
