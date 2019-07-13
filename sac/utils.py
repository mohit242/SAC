import torch
import numpy as np


def soft_update(target, source, polyak=0.001):
    for target_param, local_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(polyak * local_param.data + (1.0 - polyak) * target_param.data)


def hard_update(target, source):
    for target_param, local_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(local_param.data)


#TODO Wrapper to squash actions in [-1, 1]
