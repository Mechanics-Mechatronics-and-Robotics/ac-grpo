from __future__ import annotations

import torch
from torch import nn


class CertaintyNet(nn.Module):
    def __init__(self, obs_size: int = 8, hidden_size: int = 128, initial_certainty: float = 0.5) -> None:
        super().__init__()
        if not 0.0 < initial_certainty < 1.0:
            raise ValueError("initial_certainty must be in (0, 1)")
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        # With a competent pretrained anchor, AC should begin close to PPO and
        # let the certainty losses move the gate only when evidence appears.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, torch.logit(torch.tensor(float(initial_certainty))).item())

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(obs).squeeze(-1)
        certainty = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
        return certainty, logits
