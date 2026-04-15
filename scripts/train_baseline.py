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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(total_steps=args.total_steps)
    summary = PPOBaselineTrainer(mode=args.mode, seed=args.seed, config=config).train()
    print(summary)


if __name__ == "__main__":
    main()
