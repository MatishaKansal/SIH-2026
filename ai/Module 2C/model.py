"""Small temporal CNN operating on [batch, frequency_bins, time]."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SpectralCNN(nn.Module):
    """Lightweight 1D CNN for learned spectral/time-frequency evidence."""

    def __init__(
        self,
        in_channels: int = 64,
        embedding_dim: int = 64,
        channels: tuple[int, int, int] = (64, 96, 128),
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels must be positive")

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        if len(channels) != 3 or any(c <= 0 for c in channels):
            raise ValueError("channels must contain three positive integers")

        self.in_channels = in_channels
        self.frequency_bins = in_channels
        self.embedding_dim = embedding_dim

        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels,
                channels[0],
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(channels[0]),
            nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                channels[0],
                channels[1],
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(channels[1]),
            nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                channels[1],
                channels[2],
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(channels[2]),
            nn.GELU(),
        )

        self.embedding = nn.Linear(
            channels[2],
            embedding_dim,
        )

        # Raw spoof logit.
        # Sigmoid is applied later during evaluation/inference.
        self.classifier = nn.Linear(
            embedding_dim,
            1,
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Return learned embedding and raw spoof logit."""

        if features.ndim != 3:
            raise ValueError(
                "expected [batch, frequency_bins, time], "
                f"got {tuple(features.shape)}"
            )

        if features.shape[1] != self.in_channels:
            raise ValueError(
                f"expected [batch, {self.in_channels}, time], "
                f"got {tuple(features.shape)}"
            )

        if not torch.isfinite(features).all():
            raise ValueError("features contain NaN or Inf")

        temporal = self.features(features)

        pooled = temporal.mean(dim=-1)

        embedding = self.embedding(pooled)

        logits = self.classifier(embedding).squeeze(-1)

        return embedding, logits


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )