import math

import torch


def round_by_factor(number: int, factor: int) -> int:
    return int(round(number / factor) * factor)


def ceil_by_factor(number: int, factor: int) -> int:
    return int(math.ceil(number / factor) * factor)


def floor_by_factor(number: int, factor: int) -> int:
    return int(math.floor(number / factor) * factor)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")
