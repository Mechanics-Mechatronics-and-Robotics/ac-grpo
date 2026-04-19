from __future__ import annotations

import csv
from pathlib import Path

import torch

from src.config import TrainConfig
from src.env import LunarLanderDiagnosticEnv
from src.policy_net import PolicyNet


def evaluate_policy_checkpoint(
    checkpoint_path: Path,
    mode: str,
    config: TrainConfig,
    device: torch.device,
    output_path: Path,
) -> dict[str, float | str]:
    policy = PolicyNet(config.obs_size, config.action_size, config.hidden_size).to(device)
    policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
    policy.eval()

    rows: list[list[float | int | str]] = []
    returns: list[float] = []
    successes: list[float] = []
    lengths: list[int] = []

    for eval_seed in config.eval_seeds:
        env = LunarLanderDiagnosticEnv(
            mode=mode,
            seed=eval_seed,
            env_id=config.env_id,
            reward_noise_p=config.reward_noise_p,
            obs_noise_sigma=config.obs_noise_sigma,
        )
        for episode_idx in range(config.eval_episodes_per_seed):
            obs, _ = env.reset(seed=eval_seed + episode_idx)
            done = False
            episode_return = 0.0
            episode_length = 0
            info = {}
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action = int(policy.distribution(obs_t).probs.argmax(dim=1).item())
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_return += float(reward)
                episode_length += 1
            success = float(bool(info and info.get("is_success", False)) or episode_return >= 200.0)
            rows.append([str(checkpoint_path.name), eval_seed, episode_idx, episode_return, success, episode_length])
            returns.append(episode_return)
            successes.append(success)
            lengths.append(episode_length)
        env.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["checkpoint", "eval_seed", "episode", "return", "success", "episode_length"])
        writer.writerows(rows)

    return {
        "checkpoint": str(checkpoint_path),
        "eval_return_mean": sum(returns) / max(1, len(returns)),
        "eval_success_mean": sum(successes) / max(1, len(successes)),
        "eval_length_mean": sum(lengths) / max(1, len(lengths)),
    }
