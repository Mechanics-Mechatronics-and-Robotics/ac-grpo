from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class PolicyNet(nn.Module):
    def __init__(self, obs_size: int = 8, action_size: int = 4, hidden_size: int = 128) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def distribution(self, obs: torch.Tensor, temperature: float = 1.0) -> Categorical:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        return Categorical(logits=self.actor(obs) / temperature)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def act(self, obs: torch.Tensor, temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs, temperature=temperature)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        probs = dist.probs
        delta = probs.gather(1, action.view(-1, 1)).squeeze(1)
        return action, log_prob, entropy, delta, self.value(obs)
