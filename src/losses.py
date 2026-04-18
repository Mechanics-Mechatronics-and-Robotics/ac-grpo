from __future__ import annotations

import torch


def alignment_loss(
    delta_t: torch.Tensor,
    certainty: torch.Tensor,
    k: int = 4,
    temperature: float = 1.0,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    delta = delta_t.detach()
    c = certainty.clamp(1e-6, 1.0 - 1e-6)
    if temperature != 1.0:
        c = torch.sigmoid(torch.logit(c) / temperature).clamp(1e-6, 1.0 - 1e-6)
    return -delta * torch.log(c) - (1.0 - delta) * torch.log((1.0 - c) / k)


def outcome_loss(success: torch.Tensor, certainty: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    r = success.detach()
    c = certainty.clamp(1e-6, 1.0 - 1e-6)
    return -alpha * (r * torch.log(c) + (1.0 - r) * torch.log(1.0 - c))


def dispersion_proxy_loss(entropy: torch.Tensor, certainty_logits: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Discrete-action surrogate: policy entropy replaces continuous Gaussian orbit geometry."""
    return 0.5 * torch.exp(certainty_logits) * entropy.detach() - 0.5 * beta * certainty_logits
