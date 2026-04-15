from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.config import CHECKPOINT_DIR, LOG_DIR, TrainConfig, ensure_output_dirs
from src.env import LunarLanderDiagnosticEnv
from src.policy_net import PolicyNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class PPOBaselineTrainer:
    def __init__(self, mode: str, seed: int, config: TrainConfig | None = None, device: str | None = None) -> None:
        self.mode = mode
        self.seed = seed
        self.config = config or TrainConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        set_seed(seed)
        ensure_output_dirs()
        self.env = LunarLanderDiagnosticEnv(
            mode=mode,
            seed=seed,
            env_id=self.config.env_id,
            reward_noise_p=self.config.reward_noise_p,
            obs_noise_sigma=self.config.obs_noise_sigma,
        )
        self.policy = PolicyNet(self.config.obs_size, self.config.action_size, self.config.hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        self.run_id = f"BASELINE_{mode}_seed{seed}"
        self.episode_log = LOG_DIR / f"{self.run_id}_episodes.csv"
        self.step_log = LOG_DIR / f"{self.run_id}_steps.csv"
        self.summary_log = LOG_DIR / f"{self.run_id}_summary.json"

    def train(self) -> dict[str, float]:
        self._init_logs()
        obs, _ = self.env.reset(seed=self.seed)
        global_step = 0
        episode_return = 0.0
        episode_length = 0
        episode_id = 0
        while global_step < self.config.total_steps:
            rollout = self._collect_rollout(obs, global_step, episode_id, episode_return, episode_length)
            obs = rollout.pop("next_obs")
            global_step = int(rollout.pop("global_step"))
            episode_return = float(rollout.pop("episode_return"))
            episode_length = int(rollout.pop("episode_length"))
            episode_id = int(rollout.pop("episode_id"))
            self._update(rollout)
            if not math.isfinite(float(rollout["rewards"].mean())):
                raise FloatingPointError("NaN detected in rollout rewards")
        torch.save(self.policy.state_dict(), CHECKPOINT_DIR / f"{self.run_id}.pt")
        summary = {"method": "BASELINE", "mode": self.mode, "seed": self.seed, "total_steps": global_step}
        self.summary_log.write_text(json.dumps(summary | {"config": asdict(self.config)}, indent=2), encoding="utf-8")
        self.env.close()
        return summary

    def _collect_rollout(
        self, obs: np.ndarray, global_step: int, episode_id: int, episode_return: float, episode_length: int
    ) -> dict[str, torch.Tensor | np.ndarray | int | float]:
        cfg = self.config
        obs_buf, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
        entropies, deltas, episode_ids, timesteps = [], [], [], []
        step_rows = []
        episode_rows = []
        for _ in range(cfg.steps_per_update):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, entropy, delta, value = self.policy.act(obs_t)
            next_obs, reward, terminated, truncated, info = self.env.step(int(action.item()))
            done = terminated or truncated
            obs_buf.append(obs)
            actions.append(action.item())
            log_probs.append(log_prob.item())
            rewards.append(float(reward))
            dones.append(float(done))
            values.append(value.item())
            entropies.append(entropy.item())
            deltas.append(delta.item())
            episode_ids.append(episode_id)
            timesteps.append(episode_length)
            step_rows.append([global_step, episode_id, episode_length, entropy.item(), delta.item(), ""])
            episode_return += float(reward)
            episode_length += 1
            global_step += 1
            obs = next_obs
            if done:
                outcome = self.env.episode_outcome(episode_return, info)
                episode_rows.append([global_step, episode_id, episode_return, outcome.logged_success, outcome.raw_success, episode_length])
                obs, _ = self.env.reset()
                episode_return = 0.0
                episode_length = 0
                episode_id += 1
        self._append_rows(self.step_log, step_rows)
        self._append_rows(self.episode_log, episode_rows)
        with torch.no_grad():
            next_value = self.policy.value(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)).item()
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        values_t = torch.tensor(values, dtype=torch.float32, device=self.device)
        advantages = self._gae(rewards_t, dones_t, values_t, next_value)
        return {
            "obs": torch.tensor(np.asarray(obs_buf), dtype=torch.float32, device=self.device),
            "actions": torch.tensor(actions, dtype=torch.long, device=self.device),
            "old_log_probs": torch.tensor(log_probs, dtype=torch.float32, device=self.device),
            "returns": advantages + values_t,
            "advantages": advantages,
            "rewards": rewards_t,
            "next_obs": obs,
            "global_step": global_step,
            "episode_return": episode_return,
            "episode_length": episode_length,
            "episode_id": episode_id,
        }

    def _gae(self, rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor, next_value: float) -> torch.Tensor:
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0.0
        for t in reversed(range(len(rewards))):
            next_nonterminal = 1.0 - dones[t]
            next_values = torch.tensor(next_value, device=self.device) if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.config.gamma * next_values * next_nonterminal - values[t]
            lastgaelam = delta + self.config.gamma * self.config.gae_lambda * next_nonterminal * lastgaelam
            advantages[t] = lastgaelam
        return advantages

    def _update(self, rollout: dict[str, torch.Tensor]) -> None:
        advantages = rollout["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        n = len(advantages)
        idxs = torch.arange(n, device=self.device)
        for _ in range(self.config.update_epochs):
            perm = idxs[torch.randperm(n, device=self.device)]
            for start in range(0, n, self.config.batch_size):
                mb = perm[start : start + self.config.batch_size]
                dist = self.policy.distribution(rollout["obs"][mb])
                new_log_prob = dist.log_prob(rollout["actions"][mb])
                ratio = (new_log_prob - rollout["old_log_probs"][mb]).exp()
                pg_loss = -torch.min(
                    advantages[mb] * ratio,
                    advantages[mb] * torch.clamp(ratio, 1.0 - self.config.clip_coef, 1.0 + self.config.clip_coef),
                ).mean()
                value_loss = nn.functional.mse_loss(self.policy.value(rollout["obs"][mb]), rollout["returns"][mb])
                entropy_loss = dist.entropy().mean()
                loss = pg_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("NaN loss detected")
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

    def _init_logs(self) -> None:
        with self.episode_log.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["step", "episode_id", "return", "success", "raw_success", "episode_length"])
        with self.step_log.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["step", "episode_id", "timestep", "entropy", "delta", "certainty"])

    @staticmethod
    def _append_rows(path: Path, rows: list[list[object]]) -> None:
        if rows:
            with path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(rows)
