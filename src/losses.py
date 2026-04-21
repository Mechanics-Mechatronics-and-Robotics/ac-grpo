from __future__ import annotations

import torch


def runner_up_stats(
    probs: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return executed-action prob, runner-up prob, and runner-up action index.

    Args:
        probs:   (B, A) softmax probabilities.
        actions: (B,)  executed action indices.

    Returns:
        action_probs:    (B,) probability of the executed action.
        runner_up_probs: (B,) highest probability among non-executed actions.
        runner_up_actions: (B,) index of the runner-up action.
    """
    action_probs = probs.gather(1, actions.view(-1, 1)).squeeze(1)
    masked = probs.clone()
    masked.scatter_(1, actions.view(-1, 1), -1.0)  # safe: probs in [0,1]
    runner_up_probs, runner_up_actions = masked.max(dim=1)
    return action_probs, runner_up_probs, runner_up_actions


def runner_up_margin(
    action_probs: torch.Tensor,
    runner_up_probs: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalised margin between the executed action and its best competitor.

        delta = p_a / (p_a + p_runner_up)

    The eps term is added symmetrically to numerator (×0.5) and denominator (×1)
    so that delta = 0.5 exactly when p_a == p_runner_up, regardless of magnitude:

        (p + 0.5*eps) / (2p + eps)  =  (p + 0.5*eps) / 2*(p + 0.5*eps)  = 0.5

    This ensures the indifference point c† = 0.5 is reachable at uniform
    initialisation (p_a ≈ p_runner_up ≈ 1/A).

    Args:
        action_probs:    (B,) probability of the executed action.
        runner_up_probs: (B,) probability of the runner-up action.
        eps:             numerical stability constant (default 1e-8).

    Returns:
        delta: (B,) margin in (eps, 1-eps).
    """
    denom = action_probs + runner_up_probs + eps        # single eps, no clamp_min needed
    delta = (action_probs + 0.5 * eps) / denom
    return delta.clamp(eps, 1.0 - eps)


def mixture_probability(
    certainty: torch.Tensor,
    action_probs: torch.Tensor,
    runner_up_probs: torch.Tensor,
) -> torch.Tensor:
    """Mixture likelihood p(a | s, c) = c * p_a + (1-c) * p_runner_up.

    Used in two places:
      - Policy gradient path: action_probs and runner_up_probs are NOT detached.
        Gradients flow to theta through the mixture ratio.
      - Certainty training path (mixture_mle_loss): probs ARE detached.
        Gradients flow to psi only.

    The caller is responsible for detaching probs appropriately.

    Args:
        certainty:       (B,) certainty values in (0, 1), output of sigmoid.
        action_probs:    (B,) probability of the executed action.
        runner_up_probs: (B,) probability of the runner-up action.

    Returns:
        mixture: (B,) mixture probability, clamped away from zero for log safety.
    """
    c = certainty.clamp(1e-6, 1.0 - 1e-6)
    return (c * action_probs + (1.0 - c) * runner_up_probs).clamp_min(1e-8)


def mixture_mle_loss(
    certainty: torch.Tensor,
    action_probs: torch.Tensor,
    runner_up_probs: torch.Tensor,
) -> torch.Tensor:
    """Per-step runner-up mixture NLL for the certainty network only.

    Policy probabilities are detached: the certainty network is trained to track
    the runner-up margin, not to influence the policy's action probabilities.
    The policy is trained through mixture_probability in the AC PPO ratio.

    At the fixed point, c_t* = delta_t = p_a / (p_a + p_runner_up), so the
    certainty network learns to predict the policy's relative commitment to the
    executed action vs. its best alternative.

    Args:
        certainty:       (B,) certainty values in (0, 1).
        action_probs:    (B,) executed-action probabilities (will be detached).
        runner_up_probs: (B,) runner-up probabilities (will be detached).

    Returns:
        Scalar mean NLL.
    """
    mix = mixture_probability(certainty, action_probs.detach(), runner_up_probs.detach())
    return -torch.log(mix).mean()


def trajectory_outcome_mle_loss(
    certainty: torch.Tensor,
    episode_ids: torch.Tensor,
    success: torch.Tensor,
    outcome_mask: torch.Tensor,
) -> torch.Tensor:
    """Trajectory-level Bernoulli NLL using mean certainty per completed episode.

    Models the binary outcome as:
        p(R_i | c̄_i) = c̄_i^R_i * (1 - c̄_i)^(1-R_i)

    where c̄_i = mean(c_t) over the episode. Training this loss pushes the
    certainty network to predict episode success from within-episode observations
    before the outcome is known.

    Under reward noise, this loss conflicts with mixture_mle_loss on corrupted
    episodes: mixture_mle_loss pushes certainty up (policy is committed) while
    this loss pushes certainty down (flipped label = failure). The certainty
    settles at an intermediate value, attenuating the policy gradient on
    corrupted trajectories.

    Episodes without a terminal step in outcome_mask are silently skipped.
    Callers should log the skip rate to detect silent data issues.

    Args:
        certainty:    (N,) certainty values for all steps in the update batch.
        episode_ids:  (N,) integer episode identifier for each step.
        success:      (N,) binary outcome label (1=success, 0=failure).
                      Values are only meaningful at terminal steps.
        outcome_mask: (N,) boolean — True only at terminal steps.

    Returns:
        Scalar mean NLL over completed episodes; zero (no grad) if none found.
    """
    outcome_mask_bool = outcome_mask.bool()
    success_float = success.detach().float()   # labels are constants, not parameters
    losses: list[torch.Tensor] = []

    for episode_id in torch.unique(episode_ids.detach()):
        episode_mask = episode_ids == episode_id
        terminal_mask = episode_mask & outcome_mask_bool
        n_terminals = int(terminal_mask.sum().item())

        if n_terminals == 0:
            continue
        if n_terminals != 1:
            raise ValueError(
                f"Episode {int(episode_id.item())} has {n_terminals} terminal steps; "
                "expected exactly 1. Check episode boundary logic."
            )

        mean_c = certainty[episode_mask].mean().clamp(1e-6, 1.0 - 1e-6)
        R = success_float[terminal_mask][0]   # scalar label for this episode
        nll = -(R * torch.log(mean_c) + (1.0 - R) * torch.log(1.0 - mean_c))
        losses.append(nll)

    if not losses:
        return torch.zeros((), dtype=certainty.dtype, device=certainty.device)

    return torch.stack(losses).mean()