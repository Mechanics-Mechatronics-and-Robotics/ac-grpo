from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import MODES, TrainConfig
from src.trainer_baseline import PPOBaselineTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BASELINE PPO on LunarLander-v2.")
    parser.add_argument("--mode", choices=MODES, default="CLEAN")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int, default=TrainConfig().total_steps)
    parser.add_argument("--grouped-rollouts", action="store_true")
    parser.add_argument("--dynamic-sampling", action="store_true")
    parser.add_argument("--group-size", type=int, default=TrainConfig().group_size)
    parser.add_argument("--rollout-temperature", type=float, default=TrainConfig().rollout_temperature)
    parser.add_argument("--epsilon-low", type=float, default=TrainConfig().epsilon_low)
    parser.add_argument("--epsilon-high", type=float, default=TrainConfig().epsilon_high)
    parser.add_argument("--dynamic-sampling-warmup-steps", type=int, default=TrainConfig().dynamic_sampling_warmup_steps)
    parser.add_argument("--strict-dynamic-sampling", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", default="BASELINE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        total_steps=args.total_steps,
        grouped_rollouts=args.grouped_rollouts,
        dynamic_sampling=args.dynamic_sampling,
        group_size=args.group_size,
        rollout_temperature=args.rollout_temperature,
        epsilon_low=args.epsilon_low,
        epsilon_high=args.epsilon_high,
        dynamic_sampling_warmup_steps=args.dynamic_sampling_warmup_steps,
        dynamic_sampling_fallback_on_empty=not args.strict_dynamic_sampling,
    )
    summary = PPOBaselineTrainer(
        mode=args.mode,
        seed=args.seed,
        config=config,
        output_dir=args.output_dir,
        run_name=args.run_name,
    ).train()
    print(summary)


if __name__ == "__main__":
    main()
