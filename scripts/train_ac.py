from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import MODES, REWARD_MODES, TrainConfig
from src.trainer_ac import ACPPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AC_LITE or AC_FULL PPO on LunarLander-v2.")
    parser.add_argument("--method", choices=("AC_LITE", "AC_FULL"), default="AC_LITE")
    parser.add_argument("--mode", choices=MODES, default="CLEAN")
    parser.add_argument("--reward-mode", choices=REWARD_MODES, default=TrainConfig().reward_mode)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int, default=TrainConfig().total_steps)
    parser.add_argument("--policy-lr", type=float, default=TrainConfig().policy_lr)
    parser.add_argument("--certainty-lr", type=float, default=TrainConfig().certainty_lr)
    parser.add_argument("--grouped-rollouts", action="store_true", default=TrainConfig().grouped_rollouts)
    parser.add_argument("--dynamic-sampling", action="store_true", default=TrainConfig().dynamic_sampling)
    parser.add_argument("--group-size", type=int, default=TrainConfig().group_size)
    parser.add_argument("--rollout-temperature", type=float, default=TrainConfig().rollout_temperature)
    parser.add_argument("--epsilon-low", type=float, default=TrainConfig().epsilon_low)
    parser.add_argument("--epsilon-high", type=float, default=TrainConfig().epsilon_high)
    parser.add_argument("--dynamic-sampling-warmup-steps", type=int, default=TrainConfig().dynamic_sampling_warmup_steps)
    parser.add_argument("--pretrained-policy-path", default=TrainConfig().pretrained_policy_path)
    parser.add_argument("--freeze-pretrained-policy", action="store_true")
    parser.add_argument("--no-detach-certainty-in-policy-loss", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        total_steps=args.total_steps,
        reward_mode=args.reward_mode,
        policy_lr=args.policy_lr,
        certainty_lr=args.certainty_lr,
        grouped_rollouts=args.grouped_rollouts,
        dynamic_sampling=args.dynamic_sampling,
        group_size=args.group_size,
        rollout_temperature=args.rollout_temperature,
        epsilon_low=args.epsilon_low,
        epsilon_high=args.epsilon_high,
        dynamic_sampling_warmup_steps=args.dynamic_sampling_warmup_steps,
        pretrained_policy_path=args.pretrained_policy_path,
        freeze_pretrained_policy=args.freeze_pretrained_policy,
        detach_certainty_in_policy_loss=not args.no_detach_certainty_in_policy_loss,
    )
    summary = ACPPOTrainer(
        method=args.method,
        mode=args.mode,
        seed=args.seed,
        config=config,
        output_dir=args.output_dir,
        run_name=args.run_name or f"{args.method}_{args.reward_mode}",
    ).train()
    print(summary)


if __name__ == "__main__":
    main()
