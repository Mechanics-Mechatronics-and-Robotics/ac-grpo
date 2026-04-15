from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import MODES, SEEDS, TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed 5-seed AC-GRPO diagnostic sweep.")
    parser.add_argument("--total-steps", type=int, default=TrainConfig().total_steps)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    runs: list[list[str]] = []
    for seed in SEEDS:
        for mode in MODES:
            runs.append([sys.executable, "scripts/train_baseline.py", "--mode", mode, "--seed", str(seed), "--total-steps", str(args.total_steps)])
            for method in ("AC_LITE", "AC_FULL"):
                runs.append([sys.executable, "scripts/train_ac.py", "--method", method, "--mode", mode, "--seed", str(seed), "--total-steps", str(args.total_steps)])
    for command in runs:
        print("running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=repo, check=True)
    subprocess.run([sys.executable, "scripts/analyze.py"], cwd=repo, check=True)


if __name__ == "__main__":
    main()
