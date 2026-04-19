from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import MODES, TrainConfig
from src.trainer_ac import ACPPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AC_LITE or AC_FULL PPO on LunarLander-v2.")
    parser.add_argument("--method", choices=("AC_LITE", "AC_FULL"), default="AC_LITE")
    parser.add_argument("--mode", choices=MODES, default="CLEAN")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int, default=TrainConfig().total_steps)
    parser.add_argument("--ac-loss-temperature", type=float, default=TrainConfig().ac_loss_temperature)
    parser.add_argument("--certainty-min-gate", type=float, default=TrainConfig().certainty_min_gate)
    parser.add_argument("--pretrained-policy-path", default=TrainConfig().pretrained_policy_path)
    parser.add_argument("--freeze-pretrained-policy", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        total_steps=args.total_steps,
        ac_loss_temperature=args.ac_loss_temperature,
        certainty_min_gate=args.certainty_min_gate,
        pretrained_policy_path=args.pretrained_policy_path,
        freeze_pretrained_policy=args.freeze_pretrained_policy,
    )
    summary = ACPPOTrainer(
        method=args.method,
        mode=args.mode,
        seed=args.seed,
        config=config,
        output_dir=args.output_dir,
        run_name=args.run_name,
    ).train()
    print(summary)


if __name__ == "__main__":
    main()
