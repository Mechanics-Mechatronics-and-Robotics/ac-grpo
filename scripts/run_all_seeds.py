from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import BASELINE_EXPERIMENTS, DEFAULT_PRETRAINED_POLICY, METHOD_MODE_EXPERIMENTS, OUTPUTS_DIR, SEEDS


def parse_args() -> argparse.Namespace:
    experiments = {**BASELINE_EXPERIMENTS, **METHOD_MODE_EXPERIMENTS}
    parser = argparse.ArgumentParser(description="Run one fixed-seed LunarLander experiment.")
    parser.add_argument("--experiment", choices=tuple(experiments), default=None)
    parser.add_argument("--all", action="store_true", help="Run the full method x mode matrix and make one combined report.")
    parser.add_argument("--total-steps", type=int, default=None, help="Override the experiment default.")
    parser.add_argument("--pretrained-policy-path", default=str(DEFAULT_PRETRAINED_POLICY))
    parser.add_argument("--freeze-pretrained-policy", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run seed 42 only.")
    return parser.parse_args()


def selected_experiments(args: argparse.Namespace) -> list[str]:
    if args.all:
        return list(METHOD_MODE_EXPERIMENTS)
    if args.experiment is None:
        raise SystemExit("Provide --experiment NAME or --all.")
    return [args.experiment]


def write_yaml(path: Path, data: dict[str, object]) -> None:
    lines = []
    for key, value in data.items():
        if isinstance(value, (list, tuple)):
            lines.append(f"{key}: [{', '.join(str(v) for v in value)}]")
        else:
            lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_command(command: list[str], cwd: Path) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def summarize_seed(seed_dir: Path, seed: int) -> dict[str, float | int]:
    episode_paths = list((seed_dir / "logs").glob("*_episodes.csv"))
    update_paths = list((seed_dir / "logs").glob("*_updates.csv"))
    row: dict[str, float | int] = {"seed": seed}
    if episode_paths:
        with episode_paths[0].open("r", newline="", encoding="utf-8") as f:
            episodes = list(csv.DictReader(f))
        tail = episodes[-20:] if len(episodes) >= 20 else episodes
        if tail:
            returns = [float(r["return"]) for r in tail]
            successes = [float(r["success"]) for r in tail]
            row["final_return_last20"] = sum(returns) / len(returns)
            row["final_success_last20"] = sum(successes) / len(successes)
        row["episodes"] = len(episodes)
    if update_paths:
        with update_paths[0].open("r", newline="", encoding="utf-8") as f:
            updates = list(csv.DictReader(f))
        if updates:
            row["discarded_group_fraction_mean"] = sum(float(r["discarded_group_fraction"]) for r in updates) / len(updates)
            row["mixed_group_fraction_mean"] = sum(float(r["mixed_group_fraction"]) for r in updates) / len(updates)
            row["mean_successes_per_group"] = sum(float(r["mean_successes_per_group"]) for r in updates) / len(updates)
            row["policy_entropy_mean"] = sum(float(r["mean_policy_entropy"]) for r in updates if r["mean_policy_entropy"] != "nan") / max(
                1, sum(1 for r in updates if r["mean_policy_entropy"] != "nan")
            )
    return row


def write_aggregate(run_dir: Path, seeds: tuple[int, ...]) -> list[dict[str, float | int]]:
    rows = [summarize_seed(run_dir / f"seed_{seed}", seed) for seed in seeds]
    fieldnames = sorted({key for row in rows for key in row})
    with (run_dir / "aggregate_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run_one_experiment(
    experiment_name: str,
    parent_dir: Path,
    repo: Path,
    seeds: tuple[int, ...],
    total_steps_override: int | None,
    analyze_single: bool,
    pretrained_policy_path: str | None,
    freeze_pretrained_policy: bool,
) -> Path:
    experiments = {**BASELINE_EXPERIMENTS, **METHOD_MODE_EXPERIMENTS}
    settings = dict(experiments[experiment_name])
    method = str(settings.get("method", "BASELINE"))
    if total_steps_override is not None:
        settings["total_steps"] = total_steps_override
    run_dir = parent_dir / experiment_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "plots").mkdir()
    config_payload = {"experiment": experiment_name, "seeds": list(seeds), **settings}
    write_yaml(run_dir / "config.yaml", config_payload)
    for seed in seeds:
        if method == "BASELINE":
            command = [
                sys.executable,
                "scripts/train_baseline.py",
                "--mode",
                str(settings["mode"]),
                "--seed",
                str(seed),
                "--total-steps",
                str(settings["total_steps"]),
                "--group-size",
                str(settings.get("group_size", 4)),
                "--rollout-temperature",
                str(settings["rollout_temperature"]),
                "--epsilon-low",
                str(settings["epsilon_low"]),
                "--epsilon-high",
                str(settings["epsilon_high"]),
                "--dynamic-sampling-warmup-steps",
                str(settings.get("dynamic_sampling_warmup_steps", 0)),
                "--output-dir",
                str(run_dir),
                "--run-name",
                "BASELINE",
            ]
            if pretrained_policy_path:
                command.extend(["--pretrained-policy-path", pretrained_policy_path])
            if freeze_pretrained_policy:
                command.append("--freeze-pretrained-policy")
            if settings["grouped_rollouts"]:
                command.append("--grouped-rollouts")
            if settings["dynamic_sampling"]:
                command.append("--dynamic-sampling")
        else:
            command = [
                sys.executable,
                "scripts/train_ac.py",
                "--method",
                method,
                "--mode",
                str(settings["mode"]),
                "--seed",
                str(seed),
                "--total-steps",
                str(settings["total_steps"]),
                "--output-dir",
                str(run_dir),
                "--run-name",
                method,
            ]
            if pretrained_policy_path:
                command.extend(["--pretrained-policy-path", pretrained_policy_path])
            if freeze_pretrained_policy:
                command.append("--freeze-pretrained-policy")
        run_command(command, repo)
    aggregate_rows = write_aggregate(run_dir, seeds)
    summary = {"experiment": experiment_name, "run_dir": str(run_dir), "seeds": list(seeds), "settings": settings, "aggregate": aggregate_rows}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if analyze_single:
        run_command([sys.executable, "scripts/analyze.py", "--log-dir", str(run_dir), "--plot-dir", str(run_dir / "plots")], repo)
    return run_dir


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    names = selected_experiments(args)
    seeds = (42,) if args.smoke else SEEDS
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    label = "all_experiments" if args.all else names[0]
    parent_dir = OUTPUTS_DIR / f"{stamp}_{label}"
    parent_dir.mkdir(parents=True, exist_ok=False)
    write_yaml(parent_dir / "config.yaml", {
        "experiments": names,
        "seeds": list(seeds),
        "total_steps_override": args.total_steps,
        "pretrained_policy_path": args.pretrained_policy_path,
        "freeze_pretrained_policy": args.freeze_pretrained_policy,
    })
    run_dirs = [
        run_one_experiment(
            name,
            parent_dir,
            repo,
            seeds,
            args.total_steps,
            analyze_single=not args.all,
            pretrained_policy_path=args.pretrained_policy_path,
            freeze_pretrained_policy=args.freeze_pretrained_policy,
        )
        for name in names
    ]
    summary = {
        "experiments": names,
        "run_dirs": [str(path) for path in run_dirs],
        "seeds": list(seeds),
        "pretrained_policy_path": args.pretrained_policy_path,
        "freeze_pretrained_policy": args.freeze_pretrained_policy,
    }
    (parent_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    run_command([sys.executable, "scripts/analyze.py", "--log-dir", str(parent_dir), "--plot-dir", str(parent_dir / "plots")], repo)


if __name__ == "__main__":
    main()
