from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SEEDS = (42, 0, 17, 9, 3)
MODES = ("CLEAN", "REWARD_NOISE", "OBS_NOISE")
METHODS = ("BASELINE", "AC_LITE", "AC_FULL")
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs"
LOG_DIR = OUTPUTS_DIR / "logs"
PLOT_DIR = OUTPUTS_DIR / "plots"
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints"
BEST_EXPERIMENT_SETTINGS = {
    "dynamic_sampling": True,
    "grouped_rollouts": True,
    "rollout_temperature": 1.0,
    "epsilon_low": 0.2,
    "epsilon_high": 0.2,
    "total_steps": 900_000,
    "dynamic_sampling_warmup_steps": 300_000,
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
    steps_per_update: int = 2048
    batch_size: int = 64
    total_steps: int = 60_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
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
    alpha: float = 1.0
    beta: float = 1.0
    ac_loss_temperature: float = 1.0
    grouped_rollouts: bool = True
    dynamic_sampling: bool = True
    group_size: int = 4
    rollout_temperature: float = 1.0
    max_group_attempts_per_update: int = 256
    dynamic_sampling_warmup_steps: int = 150_000
    dynamic_sampling_fallback_on_empty: bool = True


def ensure_output_dirs() -> None:
    for path in (LOG_DIR, PLOT_DIR, CHECKPOINT_DIR):
        path.mkdir(parents=True, exist_ok=True)
