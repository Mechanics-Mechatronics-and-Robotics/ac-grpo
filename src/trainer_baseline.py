from __future__ import annotations

import csv
import json
import math
import platform
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.config import CHECKPOINT_DIR, LOG_DIR, TrainConfig, ensure_output_dirs
from src.env import LunarLanderDiagnosticEnv
from src.evaluation import evaluate_policy_checkpoint
from src.policy_net import PolicyNet


SPARSE_REWARD_SEMANTICS = (
    "Sparse terminal reward for learning: r_t_train=0 before episode end and "
    "r_T_train=policy_success at termination. Dense LunarLander return is logged only."
)
DENSE_REWARD_SEMANTICS = "Dense reward for learning: r_t_train=r_t_env at every step."


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class PPOBaselineTrainer:
    def __init__(
        self,
        mode: str,
        seed: int,
        config: TrainConfig | None = None,
        device: str | None = None,
        output_dir: Path | None = None,
        run_name: str = "BASELINE",
    ) -> None:
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
        self._load_pretrained_policy()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        self.run_id = f"{run_name}_{mode}_seed{seed}"
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

    def _train_reward(self, env_reward: float, done: bool, outcome_policy_success: float) -> float:
        if self.config.reward_mode == "DENSE":
            return float(env_reward)
        return float(outcome_policy_success) if done else 0.0

    def train(self) -> dict[str, float]:
        self._init_logs()
        obs, _ = self.env.reset(seed=self.seed)
        global_step = 0
        episode_return = 0.0
        episode_length = 0
        episode_id = 0
        next_checkpoint_step = self.config.checkpoint_interval
        saved_checkpoints: list[Path] = []
        while global_step < self.config.total_steps:
            step_budget = self.config.total_steps - global_step
            dynamic_sampling_active = self.config.dynamic_sampling and global_step >= self.config.dynamic_sampling_warmup_steps
            if self.config.grouped_rollouts or self.config.dynamic_sampling:
                rollout = self._collect_grouped_rollout(
                    obs,
                    global_step,
                    episode_id,
                    episode_return,
                    episode_length,
                    step_budget,
                    dynamic_sampling_active,
                )
            else:
                rollout = self._collect_rollout(obs, global_step, episode_id, episode_return, episode_length, step_budget)
            obs = rollout.pop("next_obs")
            global_step = int(rollout.pop("global_step"))
            episode_return = float(rollout.pop("episode_return"))
            episode_length = int(rollout.pop("episode_length"))
            episode_id = int(rollout.pop("episode_id"))
            update_stats = self._update(rollout) if int(rollout["obs"].shape[0]) > 0 else {"loss": math.nan, "grad_norm": math.nan}
            self._log_update(global_step, rollout, update_stats)
            if rollout["rewards"].numel() and not math.isfinite(float(rollout["rewards"].mean())):
                raise FloatingPointError("NaN detected in rollout rewards")
            while global_step >= next_checkpoint_step:
                saved_checkpoints.append(self._save_checkpoint(next_checkpoint_step))
                next_checkpoint_step += self.config.checkpoint_interval
        final_checkpoint = self._save_checkpoint(global_step, suffix="final")
        saved_checkpoints.append(final_checkpoint)
        eval_rows = self._evaluate_checkpoints(saved_checkpoints)
        best_eval = max(eval_rows, key=lambda row: float(row["eval_return_mean"])) if eval_rows else {}
        challenge_rows = self._evaluate_challenge_suite(best_eval)
        summary = {
            "method": "BASELINE",
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
            "reward_semantics": DENSE_REWARD_SEMANTICS if self.config.reward_mode == "DENSE" else SPARSE_REWARD_SEMANTICS,
            "reward_noise_semantics": "REWARD_NOISE can convert a raw terminal success into policy_success=0; sparse mode flips the terminal binary reward, dense mode leaves per-step shaping intact and only affects the binary outcome label.",
        }
        self.summary_log.write_text(json.dumps(summary_with_config, indent=2), encoding="utf-8")
        self.env.close()
        return summary

    def _collect_rollout(
        self, obs: np.ndarray, global_step: int, episode_id: int, episode_return: float, episode_length: int, step_budget: int
    ) -> dict[str, torch.Tensor | np.ndarray | int | float]:
        cfg = self.config
        obs_buf, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
        entropies, deltas, episode_ids, timesteps = [], [], [], []
        step_rows = []
        episode_rows = []
        episode_train_return = 0.0
        for _ in range(min(cfg.steps_per_update, step_budget)):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, entropy, delta, value = self.policy.act(obs_t, temperature=cfg.rollout_temperature)
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
            episode_ids.append(episode_id)
            timesteps.append(episode_length)
            step_rows.append([global_step, episode_id, episode_length, entropy.item(), delta.item(), ""])
            episode_return += float(reward)
            episode_length += 1
            global_step += 1
            obs = next_obs
            episode_train_return += rewards[-1]
            if done:
                outcome = self.env.episode_outcome(episode_return, info)
                rewards[-1] = self._train_reward(reward, done, float(outcome.policy_success))
                if self.config.reward_mode == "SPARSE":
                    episode_train_return += rewards[-1]
                else:
                    episode_train_return += rewards[-1] - float(reward)
                episode_rows.append(
                    [
                        global_step,
                        episode_id,
                        episode_return,
                        episode_train_return,
                        outcome.logged_success,
                        outcome.raw_success,
                        episode_length,
                        "",
                        "",
                        "",
                        "",
                    ]
                )
                obs, _ = self.env.reset()
                episode_return = 0.0
                episode_train_return = 0.0
                episode_length = 0
                episode_id += 1
        self._append_rows(self.step_log, step_rows)
        self._append_rows(self.episode_log, episode_rows)
        with torch.no_grad():
            next_value = self.policy.value(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)).item()
        if not rewards:
            empty = torch.tensor([], dtype=torch.float32, device=self.device)
            return {
                "obs": torch.empty((0, cfg.obs_size), dtype=torch.float32, device=self.device),
                "actions": torch.empty((0,), dtype=torch.long, device=self.device),
                "old_log_probs": empty,
                "returns": empty,
                "advantages": empty,
                "rewards": empty,
                "entropy": empty,
                "group_total": torch.tensor(0.0, dtype=torch.float32, device=self.device),
                "group_discarded": torch.tensor(0.0, dtype=torch.float32, device=self.device),
                "group_mixed": torch.tensor(0.0, dtype=torch.float32, device=self.device),
                "group_successes": torch.tensor([], dtype=torch.float32, device=self.device),
                "next_obs": obs,
                "global_step": global_step,
                "episode_return": episode_return,
                "episode_length": episode_length,
                "episode_id": episode_id,
            }
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
            "entropy": torch.tensor(entropies, dtype=torch.float32, device=self.device),
            "group_total": torch.tensor(0.0, dtype=torch.float32, device=self.device),
            "group_discarded": torch.tensor(0.0, dtype=torch.float32, device=self.device),
            "group_mixed": torch.tensor(0.0, dtype=torch.float32, device=self.device),
            "group_successes": torch.tensor([], dtype=torch.float32, device=self.device),
            "next_obs": obs,
            "global_step": global_step,
            "episode_return": episode_return,
            "episode_length": episode_length,
            "episode_id": episode_id,
        }

    def _collect_grouped_rollout(
        self,
        obs: np.ndarray,
        global_step: int,
        episode_id: int,
        episode_return: float,
        episode_length: int,
        step_budget: int,
        dynamic_sampling_active: bool,
    ) -> dict[str, torch.Tensor | np.ndarray | int | float]:
        cfg = self.config
        kept_steps: list[dict[str, float | int | np.ndarray]] = []
        fallback_steps: list[dict[str, float | int | np.ndarray]] = []
        all_step_rows, episode_rows = [], []
        group_total = 0
        group_discarded = 0
        group_mixed = 0
        group_success_counts: list[float] = []
        attempts = 0
        target_steps = min(cfg.steps_per_update, step_budget)
        episode_train_return = 0.0
        while global_step < self.config.total_steps and len(all_step_rows) < target_steps and attempts < cfg.max_group_attempts_per_update:
            attempts += 1
            group_steps: list[dict[str, float | int | np.ndarray]] = []
            group_successes: list[int] = []
            for _ in range(cfg.group_size):
                episode_start = len(group_steps)
                done = False
                # Once a group starts, finish full episodes so dynamic sampling has
                # valid success/fail labels. This may overshoot the per-update
                # step target, but avoids empty updates from partial groups.
                while not done:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        action, log_prob, entropy, delta, value = self.policy.act(obs_t, temperature=cfg.rollout_temperature)
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
                            "step": global_step,
                            "episode_id": episode_id,
                            "timestep": episode_length,
                        }
                    )
                    all_step_rows.append([global_step, episode_id, episode_length, entropy.item(), delta.item(), ""])
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
                    group_steps[idx]["episode_success"] = float(outcome.logged_success)
                group_successes.append(outcome.logged_success)
                episode_rows.append(
                    [
                        global_step,
                        episode_id,
                        episode_return,
                        episode_train_return,
                        outcome.logged_success,
                        outcome.raw_success,
                        episode_length,
                        "",
                        "",
                        "",
                        "",
                    ]
                )
                obs, _ = self.env.reset()
                episode_return = 0.0
                episode_train_return = 0.0
                episode_length = 0
                episode_id += 1
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
        if dynamic_sampling_active and group_total > 0 and group_mixed == 0:
            if cfg.dynamic_sampling_fallback_on_empty and fallback_steps:
                kept_steps = fallback_steps
                used_fallback = True
                print(f"warning: no mixed groups at step {global_step}; falling back to sampled groups for this update")
            else:
                print(f"warning: all {group_total} groups discarded at step {global_step}; skipping this update")
        self._append_rows(self.step_log, all_step_rows)
        self._append_rows(self.episode_log, episode_rows)
        with torch.no_grad():
            next_value = self.policy.value(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)).item()
        if not kept_steps:
            empty = torch.tensor([], dtype=torch.float32, device=self.device)
            return {
                "obs": torch.empty((0, cfg.obs_size), dtype=torch.float32, device=self.device),
                "actions": torch.empty((0,), dtype=torch.long, device=self.device),
                "old_log_probs": empty,
                "returns": empty,
                "advantages": empty,
                "rewards": empty,
                "entropy": empty,
                "group_total": torch.tensor(float(group_total), dtype=torch.float32, device=self.device),
                "group_discarded": torch.tensor(float(group_discarded), dtype=torch.float32, device=self.device),
                "group_mixed": torch.tensor(float(group_mixed), dtype=torch.float32, device=self.device),
                "used_dynamic_fallback": torch.tensor(float(used_fallback), dtype=torch.float32, device=self.device),
                "group_successes": torch.tensor(group_success_counts, dtype=torch.float32, device=self.device),
                "next_obs": obs,
                "global_step": global_step,
                "episode_return": episode_return,
                "episode_length": episode_length,
                "episode_id": episode_id,
            }
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
            "rewards": rewards_t,
            "entropy": torch.tensor([float(s["entropy"]) for s in kept_steps], dtype=torch.float32, device=self.device),
            "group_total": torch.tensor(float(group_total), dtype=torch.float32, device=self.device),
            "group_discarded": torch.tensor(float(group_discarded), dtype=torch.float32, device=self.device),
            "group_mixed": torch.tensor(float(group_mixed), dtype=torch.float32, device=self.device),
            "used_dynamic_fallback": torch.tensor(float(used_fallback), dtype=torch.float32, device=self.device),
            "group_successes": torch.tensor(group_success_counts, dtype=torch.float32, device=self.device),
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
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        n = len(advantages)
        idxs = torch.arange(n, device=self.device)
        last_loss = math.nan
        last_grad_norm = math.nan
        for _ in range(self.config.update_epochs):
            perm = idxs[torch.randperm(n, device=self.device)]
            for start in range(0, n, self.config.batch_size):
                mb = perm[start : start + self.config.batch_size]
                dist = self.policy.distribution(rollout["obs"][mb])
                new_log_prob = dist.log_prob(rollout["actions"][mb])
                ratio = (new_log_prob - rollout["old_log_probs"][mb]).exp()
                clipped_ratio = torch.clamp(ratio, 1.0 - self.config.epsilon_low, 1.0 + self.config.epsilon_high)
                pg_loss = -torch.min(
                    advantages[mb] * ratio,
                    advantages[mb] * clipped_ratio,
                ).mean()
                value_loss = nn.functional.mse_loss(self.policy.value(rollout["obs"][mb]), rollout["returns"][mb])
                entropy_loss = dist.entropy().mean()
                loss = pg_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("NaN loss detected")
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                last_loss = float(loss.item())
                last_grad_norm = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)
        return {"loss": last_loss, "grad_norm": last_grad_norm}

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
            csv.writer(f).writerow(["step", "episode_id", "timestep", "entropy", "delta", "certainty"])
        with self.update_log.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    "step",
                    "discarded_group_fraction",
                    "mixed_group_fraction",
                    "mean_successes_per_group",
                    "mean_policy_entropy",
                    "kept_steps",
                    "used_dynamic_fallback",
                    "loss",
                    "grad_norm",
                ]
            )

    def _log_update(self, global_step: int, rollout: dict[str, torch.Tensor], update_stats: dict[str, float]) -> None:
        group_total = float(rollout["group_total"].item())
        group_discarded = float(rollout["group_discarded"].item())
        group_mixed = float(rollout["group_mixed"].item())
        group_successes = rollout["group_successes"]
        discarded_fraction = group_discarded / group_total if group_total else 0.0
        mixed_fraction = group_mixed / group_total if group_total else 0.0
        mean_successes = float(group_successes.mean().item()) if group_successes.numel() else 0.0
        entropy = rollout["entropy"]
        mean_entropy = float(entropy.mean().item()) if entropy.numel() else math.nan
        row = [
            global_step,
            discarded_fraction,
            mixed_fraction,
            mean_successes,
            mean_entropy,
            int(rollout["obs"].shape[0]),
            int(float(rollout.get("used_dynamic_fallback", torch.tensor(0.0, device=self.device)).item())),
            update_stats.get("loss", math.nan),
            update_stats.get("grad_norm", math.nan),
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
        path = self.checkpoint_dir / f"{self.run_id}_{label}.pt"
        torch.save(self.policy.state_dict(), path)
        return path

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
