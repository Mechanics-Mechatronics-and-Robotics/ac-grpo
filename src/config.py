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


@dataclass(frozen=True)
class TrainConfig:
    env_id: str = "LunarLander-v2"
    steps_per_update: int = 2048
    batch_size: int = 64
    total_steps: int = 200_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
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


def ensure_output_dirs() -> None:
    for path in (LOG_DIR, PLOT_DIR, CHECKPOINT_DIR):
        path.mkdir(parents=True, exist_ok=True)
