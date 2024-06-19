import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from typing import Optional

from decision_transformer.envs.d4rl_infos import (
    D4RL_DATASET_STATS,
    REF_MAX_SCORE,
    REF_MIN_SCORE,
)

class ResidualBlock(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1, dim2)
        self.fc2 = torch.nn.Linear(dim2, dim2)
        self.activation = nn.GELU()
    def forward(self, x):
        hidden = self.fc1(x)
        residual = hidden
        hidden = self.activation(hidden)
        out = self.fc2(hidden)
        out += residual
        return out

class MLPBlock(nn.Module):
    def __init__(self, dim1, dim2, hidden):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1, hidden)
        self.fc2 = torch.nn.Linear(hidden, dim2)
        self.activation = nn.GELU()
    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        return out

def cross_entropy(logits, labels, num_bins):
    # labels = F.one_hot(labels.long(), num_classes=int(num_bin)).squeeze(2)
    labels = F.one_hot(
        labels.long(), num_classes=int(num_bins)
    ).squeeze()
    criterion = nn.CrossEntropyLoss()
    return criterion(logits, labels.float())

def encode_return(env_name, ret, scale=1.0, num_bin=120, rtg_scale=1000):
    # quantify the return, from a real number to an integer that indicates the bin index
    env_key = env_name.split("-")[0].lower()
    if env_key not in REF_MAX_SCORE:
        ret_max = 100
    else:
        ret_max = REF_MAX_SCORE[env_key]
    if env_key not in REF_MIN_SCORE:
        ret_min = -20
    else:
        ret_min = REF_MIN_SCORE[env_key]
    ret_max /= rtg_scale
    ret_min /= rtg_scale
    interval = (ret_max - ret_min) / (num_bin-1)
    ret = torch.clip(ret, ret_min, ret_max)
    return ((ret - ret_min) // interval).float()

def decode_return(env_name, ret, scale=1.0, num_bin=120, rtg_scale=1000):
    # from the bin index to a real number
    env_key = env_name.split("-")[0].lower()
    if env_key not in REF_MAX_SCORE:
        ret_max = 100
    else:
        ret_max = REF_MAX_SCORE[env_key]
    if env_key not in REF_MIN_SCORE:
        ret_min = -20
    else:
        ret_min = REF_MIN_SCORE[env_key]
    ret_max /= rtg_scale
    ret_min /= rtg_scale
    interval = (ret_max - ret_min) / num_bin
    return ret * interval + ret_min


def sample_from_logits(
    logits: torch.Tensor,
    temperature: Optional[float] = 1e0,
    top_percentile: Optional[float] = None,
) -> torch.Tensor:

    if top_percentile is not None:
        percentile = torch.quantile(
            logits, top_percentile, axis=-1, keepdim=True
        )
        logits = torch.where(logits >= percentile, logits, -np.inf)
    m = Categorical(logits=temperature * logits)
    return m.sample().unsqueeze(-1)

def expert_sampling(
    logits: torch.Tensor,
    temperature: Optional[float] = 1e0,
    top_percentile: Optional[float] = None,
    expert_weight: Optional[float] = 10, # this is the `k`
) -> torch.Tensor:
    B, T, num_bin = logits.shape
    expert_logits = (
        torch.linspace(0, 1, num_bin).repeat(B, T, 1).to(logits.device)
    )
    return sample_from_logits(
        logits + expert_weight * expert_logits, temperature, top_percentile
    )


