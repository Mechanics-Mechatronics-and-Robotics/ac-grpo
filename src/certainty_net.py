from __future__ import annotations

import torch
from torch import nn


class CertaintyNet(nn.Module):
    def __init__(self, obs_size: int = 8, hidden_size: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(obs).squeeze(-1)
        certainty = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
        return certainty, logits
