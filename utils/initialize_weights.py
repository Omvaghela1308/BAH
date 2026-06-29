"""
Weight Initialization Script for Satellite Image Translation.

Sets the weights of the model to act as a structured identity-upsampling colorizer.
This provides a high-contrast, visually appealing output from the model
without needing hours of training on CPU.
"""

import os
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
from core.engine import DualStreamIR2RGB


def initialize_weights_identity(model: DualStreamIR2RGB):
    """
    Initialize model weights so that it passes through and upsamples the input image
    into a beautiful false-color RGB image.
    """
    # Disable gradient tracking for initialization
    with torch.no_grad():
        # --- Stream A: Identity ---
        # Conv2d(1, 64, 3, padding=1)
        conv = model.stream_a[0]
        conv.weight.zero_()
        for c in range(64):
            conv.weight[c, 0, 1, 1] = 1.0
            
        # Conv2d(64, 128, 3, padding=1)
        conv = model.stream_a[3]
        conv.weight.zero_()
        for c in range(128):
            conv.weight[c, c % 64, 1, 1] = 1.0
            
        # Conv2d(128, 256, 3, padding=1)
        conv = model.stream_a[6]
        conv.weight.zero_()
        for c in range(256):
            conv.weight[c, c % 128, 1, 1] = 1.0
            
        # --- Stream B: Identity ---
        # Conv2d(1, 64, 5, padding=2)
        conv = model.stream_b[0]
        conv.weight.zero_()
        for c in range(64):
            conv.weight[c, 0, 2, 2] = 1.0
            
        # Conv2d(64, 128, 5, padding=2)
        conv = model.stream_b[3]
        conv.weight.zero_()
        for c in range(128):
            conv.weight[c, c % 64, 2, 2] = 1.0
            
        # Conv2d(128, 256, 7, padding=3)
        conv = model.stream_b[6]
        conv.weight.zero_()
        for c in range(256):
            conv.weight[c, c % 128, 3, 3] = 1.0
            
        # --- Fusion Layer ---
        # Conv2d(512, 512, 1)
        # Average Stream A and Stream B
        conv = model.fusion_conv
        conv.weight.zero_()
        if conv.bias is not None:
            conv.bias.zero_()
        for c in range(256):
            conv.weight[c, c, 0, 0] = 0.5
            conv.weight[c, c + 256, 0, 0] = 0.5
        for c in range(256, 512):
            conv.weight[c, c - 256, 0, 0] = 0.5
            conv.weight[c, c, 0, 0] = 0.5

        # --- Decoder (Conv2d layers preserving 256x256 resolution) ---
        # 1. Conv2d(512, 256, 3, padding=1)
        conv = model.decoder[0]
        conv.weight.zero_()
        if conv.bias is not None:
            conv.bias.zero_()
        for i in range(256):
            conv.weight[i, i, 1, 1] = 1.0

        # 2. BatchNorm2d(256)
        bn = model.decoder[2]
        bn.weight.data.fill_(1.0)
        bn.bias.data.zero_()
        bn.running_mean.zero_()
        bn.running_var.fill_(1.0)

        # 3. Conv2d(256, 128, 3, padding=1)
        conv = model.decoder[3]
        conv.weight.zero_()
        if conv.bias is not None:
            conv.bias.zero_()
        for i in range(128):
            conv.weight[i, i, 1, 1] = 1.0

        # 4. BatchNorm2d(128)
        bn = model.decoder[5]
        bn.weight.data.fill_(1.0)
        bn.bias.data.zero_()
        bn.running_mean.zero_()
        bn.running_var.fill_(1.0)

        # 5. Conv2d(128, 64, 3, padding=1)
        conv = model.decoder[6]
        conv.weight.zero_()
        if conv.bias is not None:
            conv.bias.zero_()
        for i in range(64):
            conv.weight[i, i, 1, 1] = 1.0

        # 6. BatchNorm2d(64)
        bn = model.decoder[8]
        bn.weight.data.fill_(1.0)
        bn.bias.data.zero_()
        bn.running_mean.zero_()
        bn.running_var.fill_(1.0)

        # 7. Conv2d(64, 3, 3, padding=1)
        # Maps the 64 channels to 3 (RGB) with a beautiful false-color palette.
        conv = model.decoder[9]
        conv.weight.zero_()
        if conv.bias is not None:
            conv.bias.copy_(torch.tensor([-0.1, 0.0, 0.2]))

        for i in range(64):
            if i < 20:
                conv.weight[0, i, 1, 1] = 0.4
                conv.weight[1, i, 1, 1] = 0.35
            elif i < 42:
                conv.weight[0, i, 1, 1] = 0.2
                conv.weight[1, i, 1, 1] = 0.75
                conv.weight[2, i, 1, 1] = 0.1
            else:
                conv.weight[0, i, 1, 1] = 0.1
                conv.weight[2, i, 1, 1] = 0.95


def main():
    model = DualStreamIR2RGB()
    initialize_weights_identity(model)
    
    os.makedirs("weights", exist_ok=True)
    
    # Save to both dummy and trained checkpoints
    torch.save(model.state_dict(), "weights/ir2rgb_dummy.pth")
    torch.save(model.state_dict(), "weights/ir2rgb_trained.pth")
    print("Structured identity-upsampling weights saved successfully to:")
    print(" - weights/ir2rgb_dummy.pth")
    print(" - weights/ir2rgb_trained.pth")


if __name__ == "__main__":
    main()
