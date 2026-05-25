"""
QuantumResNet – a 1D transposed-convolution network that upsamples a
64-point low-resolution wavefunction to a 1024-point high-resolution
prediction (×16 spatial upsampling).

Architecture
------------
Four upsample stages, each doubling the spatial dimension:
    64 → 128 → 256 → 512 → 1024

Each stage pairs a ConvTranspose1d (stride 2) with a lightweight
residual block (two Conv1d layers + BatchNorm + GELU) operating at
the new resolution, keeping gradients healthy without excessive depth.
"""

import torch
import torch.nn as nn


class ResBlock1d(nn.Module):
    """Pre-activation residual block at a fixed spatial resolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class UpsampleStage(nn.Module):
    """ConvTranspose1d (×2) followed by a residual refinement block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # stride=2, kernel=4, padding=1 → exact ×2 length
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4,
                                     stride=2, padding=1)
        self.res = ResBlock1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.up(x))


class QuantumResNet(nn.Module):
    """Upsample a (B, 1, 64) low-res wavefunction to (B, 1, 1024).

    Parameters
    ----------
    base_channels : int
        Channel width used in the hidden upsample stages.  The input and
        output both have 1 channel (a single real-valued wavefunction).
    """

    def __init__(self, base_channels: int = 64):
        super().__init__()
        c = base_channels

        # lift from 1 channel to feature space
        self.input_proj = nn.Conv1d(1, c, kernel_size=3, padding=1)

        # 4 × upsample stages: 64 → 128 → 256 → 512 → 1024
        self.stages = nn.Sequential(
            UpsampleStage(c, c),
            UpsampleStage(c, c),
            UpsampleStage(c, c),
            UpsampleStage(c, c),
        )

        # project back to a single channel
        self.output_proj = nn.Sequential(
            nn.BatchNorm1d(c),
            nn.GELU(),
            nn.Conv1d(c, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (B, 1, 64)

        Returns
        -------
        Tensor of shape (B, 1, 1024)
        """
        x = self.input_proj(x)   # (B, C, 64)
        x = self.stages(x)       # (B, C, 1024)
        x = self.output_proj(x)  # (B, 1, 1024)
        return x
