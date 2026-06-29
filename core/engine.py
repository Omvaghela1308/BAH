"""
Core Engine Module for Satellite Infrared-to-Optical Image Translation.

This module defines the Dual-Stream Generator network architecture (DualStreamIR2RGB)
designed to fuse structural edge features and semantic context features extracted
from grayscale Infrared (IR) images to generate translated 3-channel Optical RGB images.
"""

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn


class DualStreamIR2RGB(nn.Module):
    """
    Dual-Stream Generator network for IR-to-RGB satellite image translation.

    Fuses structural detail (Stream A) and semantic context (Stream B)
    to perform optical translation on single-channel satellite images.
    """

    def __init__(self) -> None:
        """Initialize the DualStreamIR2RGB model components."""
        super().__init__()

        # Stream A — Structural Edge Extractor (3 x Conv2d blocks, kernel_size=3, padding=1)
        # Channels: 1 -> 64 -> 128 -> 256
        self.stream_a = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )

        # Stream B — Semantic Context Extractor (2 x Conv2d [k=5, p=2], 1 x Conv2d [k=7, p=3])
        # Channels: 1 -> 64 -> 128 -> 256
        self.stream_b = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=7, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )

        # Fusion Layer: Project concatenated features (512 channels) to 512 channels
        self.fusion_conv = nn.Conv2d(512, 512, kernel_size=1, bias=True)

        # Identity layer to preserve the full 256x256 spatial resolution
        self.pool = nn.Identity()

        # Decoder: Conv2d layers maintaining 256x256 resolution
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()                                                          # Scale to [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for image translation.

        Args:
            x (torch.Tensor): Input grayscale tensor of shape [B, 1, 256, 256].

        Returns:
            torch.Tensor: Translated optical RGB tensor of shape [B, 3, 256, 256].
        """
        # Validate input shape (supporting dynamic batch sizes)
        assert len(x.shape) == 4 and x.shape[1] == 1 and x.shape[2] == 256 and x.shape[3] == 256, f"Expected shape [B, 1, 256, 256], got {x.shape}"

        # Extract features
        feat_a = self.stream_a(x)
        feat_b = self.stream_b(x)

        # Fuse features along channel dimension
        fused = torch.cat([feat_a, feat_b], dim=1)  # Shape: [B, 512, 256, 256]
        fused_projected = self.fusion_conv(fused)   # Shape: [B, 512, 256, 256]

        # Pass through identity pooling (preserving 256x256 shape)
        fused_pooled = self.pool(fused_projected)   # Shape: [B, 512, 256, 256]

        # Decode to final RGB shape
        out: torch.Tensor = self.decoder(fused_pooled)  # Shape: [B, 3, 256, 256]
        return out
