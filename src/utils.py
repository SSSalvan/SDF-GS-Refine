"""
utils.py
Basic infra: seeding, device detection, config loading, simple logger.
Caveman-simple: no external logging framework, just print with timestamps.
"""
import os
import json
import random
import time
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def check_tensor(name: str, t: torch.Tensor, raise_on_bad=True):
    """Basic NaN/Inf validation used throughout the pipeline (Section 12/13)."""
    if t is None:
        return
    n_nan = torch.isnan(t).sum().item()
    n_inf = torch.isinf(t).sum().item()
    if n_nan or n_inf:
        msg = f"[VALIDATION FAILED] tensor '{name}' has {n_nan} NaN and {n_inf} Inf values (shape={tuple(t.shape)})"
        if raise_on_bad:
            raise ValueError(msg)
        else:
            print(msg)
    return n_nan == 0 and n_inf == 0
