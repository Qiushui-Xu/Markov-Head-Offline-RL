from torch import optim
from itertools import chain
import timm.optim.optim_factory as optim_factory


def get_optimizer(args, model):
    arch_param, ctl_param = [], []
    
    for name, param in model.named_parameters():
        if "control" in name:
            arch_param.append(param)
        else:
            ctl_param.append(param)

    arch_opt = optim.AdamW(
        arch_param,
        lr=args["learning_rate"],
        weight_decay=args["weight_decay"],
    )  
    ctl_opt = optim.AdamW(
        ctl_param,
        lr=args["learning_rate"],
        weight_decay=args["weight_decay"],
    )  

    return arch_opt, ctl_opt
