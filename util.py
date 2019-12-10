from collections import OrderedDict
import random
import string
import json
import os
import torch


def generate_id():
    sw_id = "".join(random.choice("".join([random.choice(
        string.ascii_letters + string.digits)
        for ch in range(4)])) for _ in range(4))
    return sw_id


def dump_json_config(conf_file_name, config):
    with conf_file_name.open("w+") as conf:
        json.dump(config, conf, indent=4, sort_keys=True,
                  default=lambda o: "<object>")


def check_support(models, supported):
    supported = [sup.lower() for sup in supported]
    for model in models:
        print(f"Checking support of {model}")
        if model.lower() not in supported:
            print(f"Model {model} not supported!")
            return False
    return True


def setup_torch():
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    if use_cuda:
        torch.backends.cudnn.benchmark = True
    # Maximum determinism
    torch.manual_seed(1)
    print(f"Using {device} to train.")
    return device


def check_dir(directory):
    # create the folder if it does not exit
    if not directory == "" and not os.path.exists(directory):
        print(f"Folder {directory} does not exist! Creating...")
        os.makedirs(directory)


def load_checkpoint(model, checkpoint_path, device="cpu"):
    device = torch.device(device)
    model_ckp = torch.load(checkpoint_path, map_location=device)

    # handle both dataparallel and normal models
    model_tmp_dict = OrderedDict()
    for name, value in model_ckp["model_state_dict"].items():
        if name.startswith("module."):
            name = name[7:]
        model_tmp_dict[name] = value

    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(model_tmp_dict)
    else:
        model.load_state_dict(model_tmp_dict)
    return model
