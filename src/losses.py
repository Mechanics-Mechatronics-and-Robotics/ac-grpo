from __future__ import annotations

import torch
import torch.nn.functional as F


def runner_up_stats(
    probs: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Executed-action prob, runner-up prob, runner-up index.

    Args:
        probs:   (B, A) softmax probabilities.
        actions: (B,)  executed action indices.

    Returns:
        action_probs:     (B,)
        runner_up_probs:  (B,)
        runner_up_actions:(B,)
    """
    action_probs = probs.gather(1, actions.view(-1, 1)).squeeze(1)
    masked = probs.clone()
    masked.scatter_(1, actions.view(-1, 1), -1.0)
    runner_up_probs, runner_up_actions = masked.max(dim=1)
    return action_probs, runner_up_probs, runner_up_actions


def mixture_nll(
    certainty: torch.Tensor,
    action_probs: torch.Tensor,
    runner_up_probs: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean NLL of the runner-up mixture model.

        L = -log( c * p_a + (1-c) * p_runner_up )

    No detach. Gradient isolation is the training loop's responsibility:
    separate zero_grad / backward / step for each optimizer.

    Args:
        certainty:       (B,) sigmoid output of the certainty network.
        action_probs:    (B,) probability of the executed action.
        runner_up_probs: (B,) probability of the runner-up action.
        eps:             numerical floor before log.

    Returns:
        Scalar mean NLL.
    """
    c = certainty.clamp(eps, 1.0 - eps)
    mixture = c * action_probs + (1.0 - c) * runner_up_probs
    return -torch.log(mixture.clamp_min(eps)).mean()


def outcome_nll(
    certainty: torch.Tensor,
    episode_ids: torch.Tensor,
    success: torch.Tensor,
    outcome_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean Bernoulli NLL of episode outcomes against mean certainty.

        L = -R * log(c̄) - (1-R) * log(1 - c̄)

    where c̄ = mean certainty over the episode.

    No detach. Same gradient isolation rule as mixture_nll.

    Args:
        certainty:    (N,) certainty values for all steps in the batch.
        episode_ids:  (N,) integer episode id per step.
        success:      (N,) binary outcome (1=success), valid at terminal steps.
        outcome_mask: (N,) True at the single terminal step of each episode.
        eps:          numerical floor.

    Returns:
        Scalar mean NLL over completed episodes.
        Zero (no grad) if no completed episode is present.
    """
    R = success.float()
    mask = outcome_mask.bool()
    losses: list[torch.Tensor] = []

    for eid in torch.unique(episode_ids):
        ep = episode_ids == eid
        terminal = ep & mask
        n = int(terminal.sum().item())
        if n == 0:
            continue
        if n != 1:
            raise ValueError(f"Episode {int(eid.item())} has {n} terminal steps.")
        mean_c = certainty[ep].mean().clamp(eps, 1.0 - eps)
        r = R[terminal][0]
        losses.append(-(r * torch.log(mean_c) + (1.0 - r) * torch.log(1.0 - mean_c)))

    if not losses:
        return torch.zeros((), dtype=certainty.dtype, device=certainty.device)

    return torch.stack(losses).mean()