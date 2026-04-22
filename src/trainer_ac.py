from __future__ import annotations

import csv
import json
import math
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.certainty_net import CertaintyNet
from src.config import CHECKPOINT_DIR, LOG_DIR, TrainConfig, ensure_output_dirs
from src.env import LunarLanderDiagnosticEnv
from src.evaluation import evaluate_policy_checkpoint
from src.losses import mixture_nll, outcome_nll, per_episode_mixture_nll, runner_up_stats
from src.policy_net import PolicyNet
from src.trainer_baseline import SPARSE_REWARD_SEMANTICS, set_seed


class ACPPOTrainer:
    @staticmethod
    def _nan_corr(xs: list[float], ys: list[float]) -> float:
        if len(xs) < 2 or len(xs) != len(ys):
            return math.nan
        x = np.asarray(xs, dtype=np.float64)
        y = np.asarray(ys, dtype=np.float64)
        if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
            return math.nan
        return float(np.corrcoef(x, y)[0, 1])

    @classmethod
    def _episode_certainty_summary(
        cls,
        certainties: list[float],
        deltas: list[float],
        action_probs: list[float],
        runner_up_probs: list[float],
    ) -> tuple[float, float, float, float]:
        if not certainties:
            return math.nan, math.nan, math.nan, math.nan
        return (
            float(np.mean(certainties)),
            cls._nan_corr(certainties, deltas),
            cls._nan_corr(certainties, action_probs),
            cls._nan_corr(certainties, runner_up_probs),
        )

    @staticmethod
    def _runner_up_margin(action_probs: torch.Tensor, runner_up_probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        denom = (action_probs + runner_up_probs).clamp_min(eps)
        return ((action_probs + 0.5 * eps) / (denom + eps)).clamp(eps, 1.0 - eps)

    @staticmethod
    def _mixture_probability(
        certainty: torch.Tensor,
        action_probs: torch.Tensor,
        runner_up_probs: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        c = certainty.clamp(eps, 1.0 - eps)
        return (c * action_probs + (1.0 - c) * runner_up_probs).clamp_min(eps)

    def _train_reward(self, env_reward: float, done: bool, outcome_policy_success: float) -> float:
        if self.config.reward_mode == "DENSE":
            return float(env_reward)
        return float(outcome_policy_success) if done else 0.0

    def __init__(
        self,
        method: str,
        mode: str,
        seed: int,
        config: TrainConfig | None = None,
        device: str | None = None,
        output_dir: Path | None = None,
        run_name: str | None = None,
    ) -> None:
        if method not in {"AC_LITE", "AC_FULL"}:
            raise ValueError("AC trainer supports only AC_LITE or AC_FULL")
        self.method = method
        self.mode = mode
        self.seed = seed
        self.config = config or TrainConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        set_seed(seed)
        ensure_output_dirs()
        self.env = LunarLanderDiagnosticEnv(mode, seed, self.config.env_id, self.config.reward_noise_p, self.config.obs_noise_sigma)
        self.policy = PolicyNet(self.config.obs_size, self.config.action_size, self.config.hidden_size).to(self.device)
        self._load_pretrained_policy()
        self.certainty = CertaintyNet(self.config.obs_size, self.config.hidden_size).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.policy_lr)
        self.certainty_optimizer = torch.optim.Adam(self.certainty.parameters(), lr=self.config.certainty_lr)
        self.run_id = f"{run_name or method}_{mode}_seed{seed}"
        if output_dir is None:
            self.log_dir = LOG_DIR
            self.checkpoint_dir = CHECKPOINT_DIR
        else:
            self.log_dir = output_dir / f"seed_{seed}" / "logs"
            self.checkpoint_dir = output_dir / f"seed_{seed}" / "checkpoints"
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.episode_log = self.log_dir / f"{self.run_id}_episodes.csv"
        self.step_log = self.log_dir / f"{self.run_id}_steps.csv"
        self.update_log = self.log_dir / f"{self.run_id}_updates.csv"
        self.summary_log = self.log_dir / f"{self.run_id}_summary.json"

    def train(self) -> dict[str, float]:
        self._init_logs()
        obs, _ = self.env.reset(seed=self.seed)
        global_step = 0
        episode_return = 0.0
        episode_length = 0
        episode_id = 0
        next_checkpoint_step = self.config.checkpoint_interval
        saved_policy_checkpoints: list[Path] = []
        while global_step < self.config.total_steps:
            dynamic_sampling_active = self.config.dynamic_sampling and global_step >= self.config.dynamic_sampling_warmup_steps
            rollout = self._collect_grouped_rollout(obs, global_step, episode_id, episode_return, episode_length, dynamic_sampling_active)
            obs = rollout.pop("next_obs")
            global_step = int(rollout.pop("global_step"))
            episode_return = float(rollout.pop("episode_return"))
            episode_length = int(rollout.pop("episode_length"))
            episode_id = int(rollout.pop("episode_id"))
            update_stats = self._update(rollout)
            self._log_update(global_step, rollout, update_stats)
            certainty_mean = float(rollout["certainty"].mean().item()) if rollout["certainty"].numel() else math.nan
            certainty_std = float(rollout["certainty"].std(unbiased=False).item()) if rollout["certainty"].numel() else math.nan
            if certainty_std < 1e-8 and (certainty_mean < 0.01 or certainty_mean > 0.99):
                print(f"warning: certainty collapse suspected at step {global_step}")
            while global_step >= next_checkpoint_step:
                saved_policy_checkpoints.append(self._save_checkpoint(next_checkpoint_step))
                next_checkpoint_step += self.config.checkpoint_interval
        final_policy_checkpoint = self._save_checkpoint(global_step, suffix="final")
        saved_policy_checkpoints.append(final_policy_checkpoint)
        eval_rows = self._evaluate_checkpoints(saved_policy_checkpoints)
        best_eval = max(eval_rows, key=lambda row: float(row["eval_return_mean"])) if eval_rows else {}
        challenge_rows = self._evaluate_challenge_suite(best_eval)
        summary = {
            "method": self.method,
            "mode": self.mode,
            "seed": self.seed,
            "total_steps": global_step,
            "best_checkpoint_by_eval_return": best_eval,
            "challenge_evaluations": challenge_rows,
        }
        summary_with_config = {
            **summary,
            "config": asdict(self.config),
            "runtime": self._runtime_metadata(),
            "reward_mode": self.config.reward_mode,
            "reward_semantics": "Dense reward for learning: r_t_train=r_t_env at every step." if self.config.reward_mode == "DENSE" else SPARSE_REWARD_SEMANTICS,
            "reward_noise_semantics": "REWARD_NOISE can convert a raw terminal success into policy_success=0; sparse mode flips the terminal binary reward, dense mode leaves per-step shaping intact and only affects the binary outcome label.",
            "certainty_gate_semantics": "AC uses standard PPO with detached certainty-gated advantages; runner-up statistics are used only to train the certainty network.",
            "unmixed_group_semantics": "If a grouped update contains no mixed-outcome groups, the sampled batch is retained for logging/critic/certainty updates, but actor updates are skipped by default because all-success/all-fail groups have no contrast.",
        }
        self.summary_log.write_text(json.dumps(summary_with_config, indent=2), encoding="utf-8")
        self.env.close()
        return summary

    def _collect_rollout(
        self, obs: np.ndarray, global_step: int, episode_id: int, episode_return: float, episode_length: int
    ) -> dict[str, torch.Tensor | np.ndarray | int | float]:
        cfg = self.config
        obs_buf, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
        entropies, deltas, certainties = [], [], []
        action_probs, runner_up_probs, runner_up_actions, old_mixture_probs = [], [], [], []
        successes, outcome_masks, terminal_masks, episode_ids = [], [], [], []
        episode_rows, step_rows = [], []
        current_episode_start = 0
        episode_train_return = 0.0
        current_episode_certainties: list[float] = []
        current_episode_deltas: list[float] = []
        current_episode_action_probs: list[float] = []
        current_episode_runner_up_probs: list[float] = []
        for _ in range(cfg.steps_per_update):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, entropy, _, value = self.policy.act(obs_t)
                probs = self.policy.distribution(obs_t).probs
                action_prob, runner_prob, runner_action = runner_up_stats(probs, action)
                delta = self._runner_up_margin(action_prob, runner_prob)
                certainty, _ = self.certainty(obs_t)
                old_mixture_prob = self._mixture_probability(certainty, action_prob, runner_prob)
            next_obs, reward, terminated, truncated, info = self.env.step(int(action.item()))
            done = terminated or truncated
            obs_buf.append(obs)
            actions.append(action.item())
            log_probs.append(log_prob.item())
            rewards.append(self._train_reward(reward, done, 0.0))
            dones.append(float(done))
            values.append(value.item())
            entropies.append(entropy.item())
            deltas.append(delta.item())
            certainties.append(certainty.item())
            action_probs.append(action_prob.item())
            runner_up_probs.append(runner_prob.item())
            runner_up_actions.append(runner_action.item())
            old_mixture_probs.append(old_mixture_prob.item())
            episode_ids.append(episode_id)
            successes.append(0.0)
            outcome_masks.append(0.0)
            terminal_masks.append(0.0)
            step_rows.append([global_step, episode_id, episode_length, entropy.item(), action_prob.item(), runner_prob.item(), delta.item(), certainty.item(), old_mixture_prob.item()])
            current_episode_certainties.append(float(certainty.item()))
            current_episode_deltas.append(float(delta.item()))
            current_episode_action_probs.append(float(action_prob.item()))
            current_episode_runner_up_probs.append(float(runner_prob.item()))
            episode_train_return += rewards[-1]
            episode_return += float(reward)
            episode_length += 1
            global_step += 1
            obs = next_obs
            if done:
                outcome = self.env.episode_outcome(episode_return, info)
                rewards[-1] = self._train_reward(reward, done, float(outcome.policy_success))
                if cfg.reward_mode == "SPARSE":
                    episode_train_return += rewards[-1]
                else:
                    episode_train_return += rewards[-1] - float(reward)
                for idx in range(current_episode_start, len(successes)):
                    successes[idx] = float(outcome.logged_success)
                    outcome_masks[idx] = 1.0
                terminal_masks[-1] = 1.0
                current_episode_start = len(successes)
                mean_certainty, corr_delta, corr_action, corr_runner = self._episode_certainty_summary(
                    current_episode_certainties,
                    current_episode_deltas,
                    current_episode_action_probs,
                    current_episode_runner_up_probs,
                )
                episode_rows.append(
                    [
                        global_step,
                        episode_id,
                        episode_return,
                        episode_train_return,
                        outcome.logged_success,
                        outcome.raw_success,
                        episode_length,
                        mean_certainty,
                        corr_delta,
                        corr_action,
                        corr_runner,
                    ]
                )
                obs, _ = self.env.reset()
                episode_return = 0.0
                episode_train_return = 0.0
                episode_length = 0
                episode_id += 1
                current_episode_certainties = []
                current_episode_deltas = []
                current_episode_action_probs = []
                current_episode_runner_up_probs = []
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
            "entropy_old": torch.tensor(entropies, dtype=torch.float32, device=self.device),
            "delta_old": torch.tensor(deltas, dtype=torch.float32, device=self.device),
            "action_prob_old": torch.tensor(action_probs, dtype=torch.float32, device=self.device),
            "runner_up_prob_old": torch.tensor(runner_up_probs, dtype=torch.float32, device=self.device),
            "runner_up_actions": torch.tensor(runner_up_actions, dtype=torch.long, device=self.device),
            "old_mixture_probs": torch.tensor(old_mixture_probs, dtype=torch.float32, device=self.device),
            "episode_ids": torch.tensor(episode_ids, dtype=torch.long, device=self.device),
            "success": torch.tensor(successes, dtype=torch.float32, device=self.device),
            "outcome_mask": torch.tensor(outcome_masks, dtype=torch.float32, device=self.device),
            "terminal_mask": torch.tensor(terminal_masks, dtype=torch.float32, device=self.device),
            "certainty": torch.tensor(certainties, dtype=torch.float32, device=self.device),
            "next_obs": obs,
            "global_step": global_step,
            "episode_return": episode_return,
            "episode_length": episode_length,
            "episode_id": episode_id,
        }

    def _collect_grouped_rollout(
        self, obs: np.ndarray, global_step: int, episode_id: int, episode_return: float, episode_length: int, dynamic_sampling_active: bool
    ) -> dict[str, torch.Tensor | np.ndarray | int | float]:
        cfg = self.config
        kept_steps: list[dict[str, float | int | np.ndarray]] = []
        fallback_steps: list[dict[str, float | int | np.ndarray]] = []
        all_step_rows, episode_rows = [], []
        group_total = 0
        group_discarded = 0
        group_mixed = 0
        group_success_counts: list[float] = []
        target_steps = min(cfg.steps_per_update, cfg.total_steps - global_step)
        episode_train_return = 0.0
        current_episode_certainties: list[float] = []
        current_episode_deltas: list[float] = []
        current_episode_action_probs: list[float] = []
        current_episode_runner_up_probs: list[float] = []
        while global_step < cfg.total_steps and len(all_step_rows) < target_steps:
            group_steps: list[dict[str, float | int | np.ndarray]] = []
            group_successes: list[int] = []
            for _ in range(cfg.group_size):
                episode_start = len(group_steps)
                done = False
                while not done:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        action, log_prob, entropy, _, value = self.policy.act(obs_t, temperature=cfg.rollout_temperature)
                        probs = self.policy.distribution(obs_t, temperature=cfg.rollout_temperature).probs
                        action_prob, runner_prob, runner_action = runner_up_stats(probs, action)
                        delta = self._runner_up_margin(action_prob, runner_prob)
                        certainty, _ = self.certainty(obs_t)
                        old_mixture_prob = self._mixture_probability(certainty, action_prob, runner_prob)
                    next_obs, reward, terminated, truncated, info = self.env.step(int(action.item()))
                    done = terminated or truncated
                    train_reward = self._train_reward(reward, done, 0.0)
                    group_steps.append(
                        {
                            "obs": obs,
                            "action": int(action.item()),
                            "log_prob": float(log_prob.item()),
                            "reward": train_reward,
                            "done": float(done),
                            "value": float(value.item()),
                            "entropy": float(entropy.item()),
                            "delta": float(delta.item()),
                            "action_prob": float(action_prob.item()),
                            "runner_up_prob": float(runner_prob.item()),
                            "runner_up_action": int(runner_action.item()),
                            "old_mixture_prob": float(old_mixture_prob.item()),
                            "certainty": float(certainty.item()),
                            "success": 0.0,
                            "outcome_mask": 0.0,
                            "terminal_mask": 0.0,
                            "episode_id": episode_id,
                        }
                    )
                    all_step_rows.append([global_step, episode_id, episode_length, entropy.item(), action_prob.item(), runner_prob.item(), delta.item(), certainty.item(), old_mixture_prob.item()])
                    current_episode_certainties.append(float(certainty.item()))
                    current_episode_deltas.append(float(delta.item()))
                    current_episode_action_probs.append(float(action_prob.item()))
                    current_episode_runner_up_probs.append(float(runner_prob.item()))
                    episode_train_return += train_reward
                    episode_return += float(reward)
                    episode_length += 1
                    global_step += 1
                    obs = next_obs
                outcome = self.env.episode_outcome(episode_return, info)
                group_steps[-1]["reward"] = self._train_reward(reward, done, float(outcome.policy_success))
                if cfg.reward_mode == "SPARSE":
                    episode_train_return += group_steps[-1]["reward"]
                else:
                    episode_train_return += group_steps[-1]["reward"] - float(reward)
                for idx in range(episode_start, len(group_steps)):
                    group_steps[idx]["success"] = float(outcome.logged_success)
                    group_steps[idx]["outcome_mask"] = 1.0
                group_steps[-1]["terminal_mask"] = 1.0
                group_successes.append(outcome.logged_success)
                mean_certainty, corr_delta, corr_action, corr_runner = self._episode_certainty_summary(
                    current_episode_certainties,
                    current_episode_deltas,
                    current_episode_action_probs,
                    current_episode_runner_up_probs,
                )
                episode_rows.append(
                    [
                        global_step,
                        episode_id,
                        episode_return,
                        episode_train_return,
                        outcome.logged_success,
                        outcome.raw_success,
                        episode_length,
                        mean_certainty,
                        corr_delta,
                        corr_action,
                        corr_runner,
                    ]
                )
                obs, _ = self.env.reset()
                episode_return = 0.0
                episode_train_return = 0.0
                episode_length = 0
                episode_id += 1
                current_episode_certainties = []
                current_episode_deltas = []
                current_episode_action_probs = []
                current_episode_runner_up_probs = []
            group_total += 1
            success_count = sum(group_successes)
            group_success_counts.append(float(success_count))
            mixed = 0 < success_count < cfg.group_size
            if mixed:
                group_mixed += 1
                kept_steps.extend(group_steps)
            elif dynamic_sampling_active:
                group_discarded += 1
                fallback_steps.extend(group_steps)
            else:
                kept_steps.extend(group_steps)
        used_fallback = False
        if dynamic_sampling_active and group_total > 0 and group_mixed == 0 and fallback_steps:
            kept_steps = fallback_steps
            used_fallback = True
            print(f"warning: no mixed groups at step {global_step}; falling back to sampled groups for this update")
        self._append_rows(self.step_log, all_step_rows)
        self._append_rows(self.episode_log, episode_rows)
        with torch.no_grad():
            next_value = self.policy.value(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)).item()
        rewards_t = torch.tensor([float(s["reward"]) for s in kept_steps], dtype=torch.float32, device=self.device)
        dones_t = torch.tensor([float(s["done"]) for s in kept_steps], dtype=torch.float32, device=self.device)
        values_t = torch.tensor([float(s["value"]) for s in kept_steps], dtype=torch.float32, device=self.device)
        advantages = self._gae(rewards_t, dones_t, values_t, next_value)
        return {
            "obs": torch.tensor(np.asarray([s["obs"] for s in kept_steps]), dtype=torch.float32, device=self.device),
            "actions": torch.tensor([int(s["action"]) for s in kept_steps], dtype=torch.long, device=self.device),
            "old_log_probs": torch.tensor([float(s["log_prob"]) for s in kept_steps], dtype=torch.float32, device=self.device),
            "returns": advantages + values_t,
            "advantages": advantages,
            "entropy_old": torch.tensor([float(s["entropy"]) for s in kept_steps], dtype=torch.float32, device=self.device),
            "delta_old": torch.tensor([float(s["delta"]) for s in kept_steps], dtype=torch.float32, device=self.device),
            "action_prob_old": torch.tensor([float(s["action_prob"]) for s in kept_steps], dtype=torch.float32, device=self.device),
            "runner_up_prob_old": torch.tensor([float(s["runner_up_prob"]) for s in kept_steps], dtype=torch.float32, device=self.device),
            "runner_up_actions": torch.tensor([int(s["runner_up_action"]) for s in kept_steps], dtype=torch.long, device=self.device),
            "old_mixture_probs": torch.tensor([float(s["old_mixture_prob"]) for s in kept_steps], dtype=torch.float32, device=self.device),
            "episode_ids": torch.tensor([int(s["episode_id"]) for s in kept_steps], dtype=torch.long, device=self.device),
            "success": torch.tensor([float(s["success"]) for s in kept_steps], dtype=torch.float32, device=self.device),
            "outcome_mask": torch.tensor([float(s["outcome_mask"]) for s in kept_steps], dtype=torch.float32, device=self.device),
            "terminal_mask": torch.tensor([float(s["terminal_mask"]) for s in kept_steps], dtype=torch.float32, device=self.device),
            "certainty": torch.tensor([float(s["certainty"]) for s in kept_steps], dtype=torch.float32, device=self.device),
            "group_total": torch.tensor(float(group_total), dtype=torch.float32, device=self.device),
            "group_discarded": torch.tensor(float(group_discarded), dtype=torch.float32, device=self.device),
            "group_mixed": torch.tensor(float(group_mixed), dtype=torch.float32, device=self.device),
            "group_successes": torch.tensor(group_success_counts, dtype=torch.float32, device=self.device),
            "used_dynamic_fallback": torch.tensor(float(used_fallback), dtype=torch.float32, device=self.device),
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

    def _update(self, rollout: dict[str, torch.Tensor]) -> dict[str, float]:
        advantages = rollout["advantages"]
        if advantages.numel() == 0:
            return {
                "policy_loss": math.nan,
                "certainty_loss": math.nan,
                "cert_loss_step": math.nan,
                "cert_loss_traj": math.nan,
                "grad_norm_theta": math.nan,
                "grad_norm_psi": math.nan,
                "c_bar_mean": math.nan,
            }
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        n = len(advantages)
        idxs = torch.arange(n, device=self.device)
        last_policy_loss = math.nan
        last_policy_grad_norm = math.nan
        last_certainty_loss = math.nan
        last_mixture_loss = math.nan
        last_trajectory_loss = math.nan
        last_certainty_grad_norm = math.nan
        last_mean_trajectory_certainty = math.nan
        ratio_means: list[float] = []
        ratio_maxes: list[float] = []
        delta_mins: list[float] = []
        delta_maxes: list[float] = []
        skip_actor_update = (
            self.config.skip_policy_update_on_unmixed_fallback
            and float(rollout.get("group_total", torch.tensor(0.0, device=self.device)).item()) > 0.0
            and float(rollout.get("group_mixed", torch.tensor(0.0, device=self.device)).item()) == 0.0
        )
        for _ in range(self.config.update_epochs):
            perm = idxs[torch.randperm(n, device=self.device)]
            for start in range(0, n, self.config.batch_size):
                mb = perm[start : start + self.config.batch_size]
                obs_mb = rollout["obs"][mb]
                actions_mb = rollout["actions"][mb]
                runner_up_actions_mb = rollout["runner_up_actions"][mb]
                self.policy_optimizer.zero_grad()
                self.certainty_optimizer.zero_grad()
                dist = self.policy.distribution(obs_mb)
                new_log_prob = dist.log_prob(actions_mb)
                ratio = (new_log_prob - rollout["old_log_probs"][mb]).exp()
                certainty_mb, _ = self.certainty(obs_mb)
                mixture_ratio_mean = ratio.mean().item()
                mixture_ratio_max = ratio.abs().max().item()
                ratio_means.append(float(mixture_ratio_mean))
                ratio_maxes.append(float(mixture_ratio_max))
                delta_mins.append(float(rollout["delta_old"][mb].min().item()))
                delta_maxes.append(float(rollout["delta_old"][mb].max().item()))
                gated_advantages = certainty_mb.detach() * advantages[mb]
                pg_loss = -torch.min(
                    gated_advantages * ratio,
                    gated_advantages * torch.clamp(ratio, 1.0 - self.config.epsilon_low, 1.0 + self.config.epsilon_high),
                ).mean()
                value_loss = nn.functional.mse_loss(self.policy.value(obs_mb), rollout["returns"][mb])
                if skip_actor_update:
                    policy_loss = self.config.value_coef * value_loss
                else:
                    policy_loss = pg_loss + self.config.value_coef * value_loss - self.config.entropy_coef * dist.entropy().mean()
                if not torch.isfinite(policy_loss):
                    raise FloatingPointError("NaN loss detected")
                if not self.config.freeze_pretrained_policy:
                    policy_loss.backward()
                    policy_grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                    self.policy_optimizer.step()
                    last_policy_grad_norm = float(policy_grad_norm.item() if hasattr(policy_grad_norm, "item") else policy_grad_norm)
                last_policy_loss = float(policy_loss.item())
            self.policy_optimizer.zero_grad()
            self.certainty_optimizer.zero_grad()
            dist_cert = self.policy.distribution(rollout["obs"])
            probs_cert = F.softmax(dist_cert.logits, dim=1)
            certainty_cert, _ = self.certainty(rollout["obs"])
            action_probs_cert, runner_up_probs_cert, _ = runner_up_stats(probs_cert, rollout["actions"])
            mixture_loss = per_episode_mixture_nll(
                certainty_cert,
                action_probs_cert,
                runner_up_probs_cert,
                rollout["episode_ids"],
            )
            trajectory_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            mean_trajectory_certainty = math.nan
            if self.method == "AC_FULL" and self.config.reward_mode == "SPARSE":
                trajectory_loss = outcome_nll(
                    certainty_cert,
                    rollout["episode_ids"],
                    rollout["success"],
                    rollout["terminal_mask"],
                )
                completed_means = [
                    certainty_cert[rollout["episode_ids"] == episode_id].mean()
                    for episode_id in torch.unique(rollout["episode_ids"])
                    if bool(((rollout["episode_ids"] == episode_id) & rollout["terminal_mask"].bool()).any().item())
                ]
                if completed_means:
                    last_mean_trajectory_certainty = float(torch.stack(completed_means).mean().item())
            certainty_loss = mixture_loss + trajectory_loss
            if not torch.isfinite(certainty_loss):
                raise FloatingPointError("NaN loss detected")
            certainty_loss.backward()
            certainty_grad_norm = nn.utils.clip_grad_norm_(self.certainty.parameters(), self.config.max_grad_norm)
            self.certainty_optimizer.step()
            last_certainty_loss = float(certainty_loss.item())
            last_mixture_loss = float(mixture_loss.item())
            last_trajectory_loss = float(trajectory_loss.item())
            last_certainty_grad_norm = float(certainty_grad_norm.item() if hasattr(certainty_grad_norm, "item") else certainty_grad_norm)
            if self.method != "AC_FULL":
                last_mean_trajectory_certainty = math.nan
        return {
            "policy_loss": last_policy_loss,
            "certainty_loss": last_certainty_loss,
            "cert_loss_step": last_mixture_loss,
            "cert_loss_traj": last_trajectory_loss,
            "grad_norm_theta": last_policy_grad_norm,
            "grad_norm_psi": last_certainty_grad_norm,
            "c_bar_mean": last_mean_trajectory_certainty,
            "policy_update_skipped": float(skip_actor_update),
            "mixture_ratio_mean": (sum(ratio_means) / len(ratio_means)) if ratio_means else math.nan,
            "mixture_ratio_max": max(ratio_maxes) if ratio_maxes else math.nan,
            "delta_min": min(delta_mins) if delta_mins else math.nan,
            "delta_max": max(delta_maxes) if delta_maxes else math.nan,
        }

    def _init_logs(self) -> None:
        with self.episode_log.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    "step",
                    "episode_id",
                    "return_env",
                    "return_train",
                    "outcome_policy",
                    "outcome_raw",
                    "episode_length",
                    "mean_certainty",
                    "certainty_delta_corr",
                    "certainty_action_prob_corr",
                    "certainty_runner_up_prob_corr",
                ]
            )
        with self.step_log.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["step", "episode_id", "timestep", "entropy", "action_prob", "runner_up_prob", "delta", "certainty", "mixture_prob"])
        with self.update_log.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    "step",
                    "discarded_group_fraction",
                    "mixed_group_fraction",
                    "mean_successes_per_group",
                    "mean_policy_entropy",
                    "kept_steps",
                    "fallback_used",
                    "policy_loss",
                    "certainty_loss",
                    "cert_loss_step",
                    "cert_loss_traj",
                    "grad_norm_theta",
                    "grad_norm_psi",
                    "mean_delta",
                    "delta_min",
                    "delta_max",
                    "certainty_mean",
                    "c_bar_mean",
                    "policy_update_skipped",
                    "mixture_ratio_mean",
                    "mixture_ratio_max",
                    "kept_steps_frac",
                ]
            )

    def _log_update(self, global_step: int, rollout: dict[str, torch.Tensor], update_stats: dict[str, float]) -> None:
        group_total = float(rollout["group_total"].item())
        group_discarded = float(rollout["group_discarded"].item())
        group_mixed = float(rollout["group_mixed"].item())
        group_successes = rollout["group_successes"]
        row = [
            global_step,
            group_discarded / group_total if group_total else 0.0,
            group_mixed / group_total if group_total else 0.0,
            float(group_successes.mean().item()) if group_successes.numel() else 0.0,
            float(rollout["entropy_old"].mean().item()) if rollout["entropy_old"].numel() else math.nan,
            int(rollout["obs"].shape[0]),
            int(float(rollout["used_dynamic_fallback"].item())),
            update_stats.get("policy_loss", math.nan),
            update_stats.get("certainty_loss", math.nan),
            update_stats.get("cert_loss_step", math.nan),
            update_stats.get("cert_loss_traj", math.nan),
            update_stats.get("grad_norm_theta", math.nan),
            update_stats.get("grad_norm_psi", math.nan),
            float(rollout["delta_old"].mean().item()) if rollout["delta_old"].numel() else math.nan,
            update_stats.get("delta_min", math.nan),
            update_stats.get("delta_max", math.nan),
            float(rollout["certainty"].mean().item()) if rollout["certainty"].numel() else math.nan,
            update_stats.get("c_bar_mean", math.nan),
            update_stats.get("policy_update_skipped", 0.0),
            update_stats.get("mixture_ratio_mean", math.nan),
            update_stats.get("mixture_ratio_max", math.nan),
            1.0 - (group_discarded / group_total if group_total else 0.0),
        ]
        self._append_rows(self.update_log, [row])

    @staticmethod
    def _append_rows(path: Path, rows: list[list[object]]) -> None:
        if rows:
            with path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(rows)

    def _load_pretrained_policy(self) -> None:
        if not self.config.pretrained_policy_path:
            return
        state = torch.load(Path(self.config.pretrained_policy_path), map_location=self.device)
        if self.config.load_pretrained_critic:
            self.policy.load_state_dict(state)
        else:
            actor_state = {key: value for key, value in state.items() if key.startswith("actor.")}
            missing, unexpected = self.policy.load_state_dict(actor_state, strict=False)
            unexpected = [key for key in unexpected if not key.startswith("critic.")]
            if unexpected:
                raise RuntimeError(f"Unexpected pretrained actor keys: {unexpected}")
            missing = [key for key in missing if key.startswith("actor.")]
            if missing:
                raise RuntimeError(f"Missing pretrained actor keys: {missing}")
        if self.config.freeze_pretrained_policy:
            for parameter in self.policy.parameters():
                parameter.requires_grad_(False)

    def _runtime_metadata(self) -> dict[str, object]:
        return {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(self.device),
            "pretrained_policy_path": self.config.pretrained_policy_path,
            "load_pretrained_critic": self.config.load_pretrained_critic,
            "freeze_pretrained_policy": self.config.freeze_pretrained_policy,
        }

    def _save_checkpoint(self, step: int, suffix: str | None = None) -> Path:
        label = suffix or f"step{step:07d}"
        policy_path = self.checkpoint_dir / f"{self.run_id}_{label}_policy.pt"
        certainty_path = self.checkpoint_dir / f"{self.run_id}_{label}_certainty.pt"
        torch.save(self.policy.state_dict(), policy_path)
        torch.save(self.certainty.state_dict(), certainty_path)
        return policy_path

    def _evaluate_checkpoints(self, checkpoints: list[Path]) -> list[dict[str, object]]:
        eval_path = self.log_dir / f"{self.run_id}_checkpoint_eval.csv"
        eval_rows: list[dict[str, object]] = []
        if self.config.pretrained_policy_path:
            eval_rows.append(
                evaluate_policy_checkpoint(
                    Path(self.config.pretrained_policy_path),
                    self.mode,
                    self.config,
                    self.device,
                    eval_path,
                    checkpoint_label="checkpoint_0_pretrained",
                    eval_name="selection",
                )
            )
        for path in checkpoints:
            eval_rows.append(evaluate_policy_checkpoint(path, self.mode, self.config, self.device, eval_path, eval_name="selection"))
        return eval_rows

    def _evaluate_challenge_suite(self, best_eval: dict[str, object]) -> list[dict[str, object]]:
        eval_path = self.log_dir / f"{self.run_id}_checkpoint_eval.csv"
        rows: list[dict[str, object]] = []
        seen: set[tuple[str, str]] = set()
        checkpoint_specs: list[tuple[str, Path]] = []
        if self.config.pretrained_policy_path:
            checkpoint_specs.append(("checkpoint_0_pretrained", Path(self.config.pretrained_policy_path)))
        if best_eval.get("checkpoint_path"):
            checkpoint_specs.append((str(best_eval.get("checkpoint")), Path(str(best_eval["checkpoint_path"]))))
        test_conditions = [("test_clean", "CLEAN", self.config.obs_noise_sigma)]
        levels = tuple(float(v) for v in self.config.test_eval_obs_noise_levels)
        if levels:
            test_conditions.append(("test_obs_noise", "OBS_NOISE", levels[0]))
        if len(levels) > 1:
            test_conditions.append(("test_obs_noise_hard", "OBS_NOISE", levels[1]))
        for checkpoint_label, checkpoint_path in checkpoint_specs:
            for eval_name, eval_mode, sigma in test_conditions:
                key = (checkpoint_label, eval_name)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    evaluate_policy_checkpoint(
                        checkpoint_path,
                        self.mode,
                        self.config,
                        self.device,
                        eval_path,
                        checkpoint_label=checkpoint_label,
                        eval_name=eval_name,
                        eval_mode_override=eval_mode,
                        eval_obs_noise_sigma=sigma,
                        eval_seeds_override=self.config.test_eval_seeds,
                        eval_episodes_override=self.config.test_eval_episodes_per_seed,
                    )
                )
        return rows
