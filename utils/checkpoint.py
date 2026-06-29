"""
Checkpoint Utility Module for Offline Model Initialization.

This module ensures the availability of a model checkpoint file by auto-generating
a dummy weights file on the first run, allowing the application to function fully
in a 100% air-gapped/offline environment without external network calls.
"""

import os
# pyrefly: ignore [missing-import]
import torch
from core.engine import DualStreamIR2RGB


def ensure_checkpoint(path: str = "weights/ir2rgb_dummy.pth") -> str:
    """
    Ensure the existence of a weights checkpoint file.

    Creates the parent directory if missing. If the checkpoint file does not exist,
    instantiates the DualStreamIR2RGB model, saves its state dict as a dummy checkpoint,
    and returns the file path.

    Args:
        path (str): Path to the target checkpoint file. Defaults to "weights/ir2rgb_dummy.pth".

    Returns:
        str: Absolute or relative path to the verified checkpoint file.
    """
    # Create the weights directory if it doesn't exist
    dir_name = os.path.dirname(path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    # Check if the file already exists; if not, generate dummy weights
    if not os.path.exists(path):
        model = DualStreamIR2RGB()
        torch.save(model.state_dict(), path)
        print("Dummy checkpoint created.")

    return path


def ensure_esrgan_checkpoint(path: str = "weights/esrgan_dummy.pth") -> str:
    """
    Ensure the existence of the ESRGAN weights checkpoint file.

    Creates the parent directory if missing. If the checkpoint file does not exist,
    instantiates the LightweightESRGAN model, saves its state dict as a dummy checkpoint,
    and returns the file path.

    Args:
        path (str): Path to the target checkpoint file. Defaults to "weights/esrgan_dummy.pth".

    Returns:
        str: Absolute or relative path to the verified checkpoint file.
    """
    dir_name = os.path.dirname(path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    if not os.path.exists(path):
        from core.super_res import LightweightESRGAN
        model = LightweightESRGAN()
        torch.save(model.state_dict(), path)
        print("ESRGAN dummy checkpoint created.")

    return path
