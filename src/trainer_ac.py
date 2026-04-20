from __future__ import annotations

import csv
import json
import math
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.certainty_net import CertaintyNet
from src.config import CHECKPOINT_DIR, LOG_DIR, TrainConfig, ensure_output_dirs
from src.env import LunarLanderDiagnosticEnv
from src.evaluation import evaluate_policy_checkpoint
from src.losses import alignment_loss, dispersion_proxy_loss, outcome_loss
from src.policy_net import PolicyNet
from src.trainer_baseline import SPARSE_REWARD_SEMANTICS, set_seed


class ACPPOTrainer:
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
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        self.certainty_optimizer = torch.optim.Adam(self.certainty.parameters(), lr=self.config.learning_rate)
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
            self._update(rollout)
            self._log_update(global_step, rollout)
            if float(rollout["certainty"].std(unbiased=False)) < 1e-8:
                print(f"warning: certainty collapse suspected at step {global_step}")
            while global_step >= next_checkpoint_step:
                saved_policy_checkpoints.append(self._save_checkpoint(next_checkpoint_step))
                next_checkpoint_step += self.config.checkpoint_interval
        final_policy_checkpoint = self._save_checkpoint(global_step, suffix="final")
        saved_policy_checkpoints.append(final_policy_checkpoint)
        eval_rows = self._evaluate_checkpoints(saved_policy_checkpoints)
        best_eval = max(eval_rows, key=lambda row: float(row["eval_return_mean"])) if eval_rows else {}
        summary = {
            "method": self.method,
            "mode": self.mode,
            "seed": self.seed,
            "total_steps": global_step,
            "best_checkpoint_by_eval_return": best_eval,
        }
        summary_with_config = {
            **summary,
            "config": asdict(self.config),
            "runtime": self._runtime_metadata(),
            "reward_semantics": SPARSE_REWARD_SEMANTICS,
            "reward_noise_semantics": "REWARD_NOISE can convert a raw terminal success into policy_success=0; no dense reward is used by PPO/GAE.",
            "certainty_gate_semantics": "policy advantage gate uses effective_c = c * (1 - c_min) + c_min.",
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
        successes, outcome_masks = [], []
        episode_rows, step_rows = [], []
        current_episode_start = 0
        for _ in range(cfg.steps_per_update):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, entropy, delta, value = self.policy.act(obs_t)
                certainty, _ = self.certainty(obs_t)
            next_obs, reward, terminated, truncated, info = self.env.step(int(action.item()))
            done = terminated or truncated
            obs_buf.append(obs)
            actions.append(action.item())
            log_probs.append(log_prob.item())
            rewards.append(0.0)
            dones.append(float(done))
            values.append(value.item())
            entropies.append(entropy.item())
            deltas.append(delta.item())
            certainties.append(certainty.item())
            successes.append(0.0)
            outcome_masks.append(0.0)
            step_rows.append([global_step, episode_id, episode_length, entropy.item(), delta.item(), certainty.item()])
            episode_return += float(reward)
            episode_length += 1
            global_step += 1
            obs = next_obs
            if done:
                outcome = self.env.episode_outcome(episode_return, info)
                rewards[-1] = float(outcome.policy_success)
                for idx in range(current_episode_start, len(successes)):
                    successes[idx] = float(outcome.logged_success)
                    outcome_masks[idx] = 1.0
                current_episode_start = len(successes)
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
            "entropy_old": torch.tensor(entropies, dtype=torch.float32, device=self.device),
            "delta_old": torch.tensor(deltas, dtype=torch.float32, device=self.device),
            "success": torch.tensor(successes, dtype=torch.float32, device=self.device),
            "outcome_mask": torch.tensor(outcome_masks, dtype=torch.float32, device=self.device),
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
        while global_step < cfg.total_steps and len(all_step_rows) < target_steps:
            group_steps: list[dict[str, float | int | np.ndarray]] = []
            group_successes: list[int] = []
            for _ in range(cfg.group_size):
                episode_start = len(group_steps)
                done = False
                while not done:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        action, log_prob, entropy, delta, value = self.policy.act(obs_t, temperature=cfg.rollout_temperature)
                        certainty, _ = self.certainty(obs_t)
                    next_obs, reward, terminated, truncated, info = self.env.step(int(action.item()))
                    done = terminated or truncated
                    group_steps.append(
                        {
                            "obs": obs,
                            "action": int(action.item()),
                            "log_prob": float(log_prob.item()),
                            "reward": 0.0,
                            "done": float(done),
                            "value": float(value.item()),
                            "entropy": float(entropy.item()),
                            "delta": float(delta.item()),
                            "certainty": float(certainty.item()),
                            "success": 0.0,
                            "outcome_mask": 0.0,
                        }
                    )
                    all_step_rows.append([global_step, episode_id, episode_length, entropy.item(), delta.item(), certainty.item()])
                    episode_return += float(reward)
                    episode_length += 1
                    global_step += 1
                    obs = next_obs
                outcome = self.env.episode_outcome(episode_return, info)
                group_steps[-1]["reward"] = float(outcome.policy_success)
                for idx in range(episode_start, len(group_steps)):
                    group_steps[idx]["success"] = float(outcome.logged_success)
                    group_steps[idx]["outcome_mask"] = 1.0
                group_successes.append(outcome.logged_success)
                episode_rows.append([global_step, episode_id, episode_return, outcome.logged_success, outcome.raw_success, episode_length])
                obs, _ = self.env.reset()
                episode_return = 0.0
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
            "success": torch.tensor([float(s["success"]) for s in kept_steps], dtype=torch.float32, device=self.device),
            "outcome_mask": torch.tensor([float(s["outcome_mask"]) for s in kept_steps], dtype=torch.float32, device=self.device),
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

    def _update(self, rollout: dict[str, torch.Tensor]) -> None:
        advantages = rollout["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        n = len(advantages)
        idxs = torch.arange(n, device=self.device)
        for _ in range(self.config.update_epochs):
            perm = idxs[torch.randperm(n, device=self.device)]
            for start in range(0, n, self.config.batch_size):
                mb = perm[start : start + self.config.batch_size]
                obs_mb = rollout["obs"][mb]
                actions_mb = rollout["actions"][mb]
                certainty_mb, logits_mb = self.certainty(obs_mb)
                dist = self.policy.distribution(obs_mb)
                new_log_prob = dist.log_prob(actions_mb)
                ratio = (new_log_prob - rollout["old_log_probs"][mb]).exp()
                effective_certainty = certainty_mb.detach() * (1.0 - self.config.certainty_min_gate) + self.config.certainty_min_gate
                gated_adv = effective_certainty * advantages[mb]
                pg_loss = -torch.min(
                    gated_adv * ratio,
                    gated_adv * torch.clamp(ratio, 1.0 - self.config.epsilon_low, 1.0 + self.config.epsilon_high),
                ).mean()
                value_loss = nn.functional.mse_loss(self.policy.value(obs_mb), rollout["returns"][mb])
                policy_loss = pg_loss + self.config.value_coef * value_loss - self.config.entropy_coef * dist.entropy().mean()
                cert_terms = [
                    alignment_loss(
                        rollout["delta_old"][mb],
                        certainty_mb,
                        self.config.action_size,
                        self.config.ac_loss_temperature,
                    )
                ]
                if self.method == "AC_FULL":
                    outcome = outcome_loss(rollout["success"][mb], certainty_mb, self.config.alpha)
                    mask = rollout["outcome_mask"][mb]
                    if float(mask.sum().item()) > 0.0:
                        cert_terms.append((outcome * mask).sum() / mask.sum().clamp_min(1.0))
                    cert_terms.append(dispersion_proxy_loss(rollout["entropy_old"][mb], logits_mb, self.config.beta))
                certainty_loss = torch.stack([term.mean() for term in cert_terms]).sum()
                if not torch.isfinite(policy_loss) or not torch.isfinite(certainty_loss):
                    raise FloatingPointError("NaN loss detected")
                if not self.config.freeze_pretrained_policy:
                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                    self.policy_optimizer.step()
                self.certainty_optimizer.zero_grad()
                certainty_loss.backward()
                nn.utils.clip_grad_norm_(self.certainty.parameters(), self.config.max_grad_norm)
                self.certainty_optimizer.step()

    def _init_logs(self) -> None:
        with self.episode_log.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["step", "episode_id", "return", "success", "raw_success", "episode_length"])
        with self.step_log.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["step", "episode_id", "timestep", "entropy", "delta", "certainty"])
        with self.update_log.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["step", "discarded_group_fraction", "mixed_group_fraction", "mean_successes_per_group", "mean_policy_entropy", "kept_steps", "used_dynamic_fallback"])

    def _log_update(self, global_step: int, rollout: dict[str, torch.Tensor]) -> None:
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
        self.policy.load_state_dict(state)
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
                )
            )
        for path in checkpoints:
            eval_rows.append(evaluate_policy_checkpoint(path, self.mode, self.config, self.device, eval_path))
        return eval_rows
