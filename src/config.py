from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SEEDS = (42, 0, 17, 9, 3)
MODES = ("CLEAN", "REWARD_NOISE", "OBS_NOISE")
REWARD_MODES = ("SPARSE", "DENSE")
METHODS = ("BASELINE", "AC_LITE", "AC_FULL")
FINAL_EXPERIMENT_GRID = (
    "baseline_clean",
    "baseline_obs_noise",
    "baseline_reward_noise",
    "ac_lite_clean",
    "ac_lite_obs_noise",
    "ac_lite_reward_noise",
    "ac_full_clean",
    "ac_full_obs_noise",
    "ac_full_reward_noise",
)
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs"
LOG_DIR = OUTPUTS_DIR / "logs"
PLOT_DIR = OUTPUTS_DIR / "plots"
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints"
PRETRAINED_DIR = REPO_ROOT / "pretrained_models"
DEFAULT_PRETRAINED_POLICY = PRETRAINED_DIR / "lunarlander_baseline_clean_seed42.pt"
BEST_EXPERIMENT_SETTINGS = {
    "dynamic_sampling": True,
    "grouped_rollouts": True,
    "rollout_temperature": 1.0,
    "epsilon_low": 0.2,
    "epsilon_high": 0.2,
    "total_steps": 250_000,
    "dynamic_sampling_warmup_steps": 10_000,
    "reward_mode": "SPARSE",
}
BEST_BASELINE_REPAIR = BEST_EXPERIMENT_SETTINGS
BASELINE_EXPERIMENTS = {
    "baseline_clean_dynsample": {
        "mode": "CLEAN",
        **BEST_EXPERIMENT_SETTINGS,
    },
}
METHOD_MODE_EXPERIMENTS = {
    f"{method.lower()}_{mode.lower()}": {
        "method": method,
        "mode": mode,
        **BEST_EXPERIMENT_SETTINGS,
    }
    for method in METHODS
    for mode in MODES
}


@dataclass(frozen=True)
class TrainConfig:
    env_id: str = "LunarLander-v2"
    reward_mode: str = "SPARSE"
    steps_per_update: int = 2048
    batch_size: int = 64
    total_steps: int = 60_000
    learning_rate: float = 1e-4
    policy_lr: float = 1e-4
    certainty_lr: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epsilon_low: float = 0.2
    epsilon_high: float = 0.2
    update_epochs: int = 4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    reward_noise_p: float = 0.2
    obs_noise_sigma: float = 0.1
    obs_size: int = 8
    action_size: int = 4
    hidden_size: int = 128
    pretrained_policy_path: str | None = str(DEFAULT_PRETRAINED_POLICY)
    load_pretrained_critic: bool = False
    freeze_pretrained_policy: bool = False
    checkpoint_interval: int = 10_000
    eval_seeds: tuple[int, ...] = (101, 102, 103)
    eval_episodes_per_seed: int = 5
    grouped_rollouts: bool = True
    dynamic_sampling: bool = True
    group_size: int = 4
    rollout_temperature: float = 1.0
    max_group_attempts_per_update: int = 256
    dynamic_sampling_warmup_steps: int = 150_000
    dynamic_sampling_fallback_on_empty: bool = True
    skip_policy_update_on_unmixed_fallback: bool = True


def ensure_output_dirs() -> None:
    for path in (LOG_DIR, PLOT_DIR, CHECKPOINT_DIR, PRETRAINED_DIR):
        path.mkdir(parents=True, exist_ok=True)
