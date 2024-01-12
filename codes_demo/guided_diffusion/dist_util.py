"""
Helpers for distributed training.
"""

import io

import blobfile as bf
import torch as th


def dev(device):
    """
    Get the device to use for torch.distributed.
    """
    if device is None:
        if th.cuda.is_available():
            return th.device(f"cuda")
        return th.device("cpu")
    return th.device(device)


def load_state_dict(path, backend=None, **kwargs):
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)


