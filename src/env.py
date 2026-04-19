from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class EpisodeOutcome:
    raw_success: int
    logged_success: int
    policy_success: int
    reward_was_corrupted: int


class LunarLanderDiagnosticEnv:
    """LunarLander wrapper for clean, false-negative reward noise, and obs noise."""

    def __init__(
        self,
        mode: str = "CLEAN",
        seed: int = 42,
        env_id: str = "LunarLander-v2",
        reward_noise_p: float = 0.2,
        obs_noise_sigma: float = 0.1,
    ) -> None:
        if mode not in {"CLEAN", "REWARD_NOISE", "OBS_NOISE"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode
        self.reward_noise_p = reward_noise_p
        self.obs_noise_sigma = obs_noise_sigma
        self.np_rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)
        self.env = gym.make(env_id, max_episode_steps=1000)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed)
        return self._maybe_corrupt_obs(obs), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._maybe_corrupt_obs(obs), reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()

    def episode_outcome(self, episode_return: float, info: dict[str, Any] | None = None) -> EpisodeOutcome:
        raw_success = int(bool(info and info.get("is_success", False)) or episode_return >= 200.0)
        logged_success = raw_success
        reward_was_corrupted = 0
        if self.mode == "REWARD_NOISE" and raw_success == 1 and self.py_rng.random() < self.reward_noise_p:
            logged_success = 0
            reward_was_corrupted = 1
        return EpisodeOutcome(
            raw_success=raw_success,
            logged_success=logged_success,
            policy_success=logged_success,
            reward_was_corrupted=reward_was_corrupted,
        )

    def _maybe_corrupt_obs(self, obs: np.ndarray) -> np.ndarray:
        obs_array = np.asarray(obs, dtype=np.float32)
        if self.mode == "OBS_NOISE":
            noise = self.np_rng.normal(0.0, self.obs_noise_sigma, size=obs_array.shape).astype(np.float32)
            obs_array = obs_array + noise
        return obs_array
