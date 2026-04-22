from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import BASELINE_EXPERIMENTS, DEFAULT_PRETRAINED_POLICY, FINAL_EXPERIMENT_GRID, METHOD_MODE_EXPERIMENTS, OUTPUTS_DIR, REWARD_MODES, SEEDS, TrainConfig


def parse_args() -> argparse.Namespace:
    experiments = {**BASELINE_EXPERIMENTS, **METHOD_MODE_EXPERIMENTS}
    parser = argparse.ArgumentParser(description="Run one fixed-seed LunarLander experiment.")
    parser.add_argument("--experiment", choices=tuple(experiments), default=None)
    parser.add_argument("--all", action="store_true", help="Run the full method x mode matrix and make one combined report.")
    parser.add_argument("--total-steps", type=int, default=None, help="Override the experiment default.")
    parser.add_argument("--pretrained-policy-path", default=str(DEFAULT_PRETRAINED_POLICY))
    parser.add_argument("--freeze-pretrained-policy", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run seed 42 only.")
    parser.add_argument("--max-parallel-seeds", type=int, default=None, help="Run up to this many seeds concurrently. Defaults to all selected seeds.")
    parser.add_argument("--reward-mode", choices=REWARD_MODES, default=None, help="Override the experiment reward mode.")
    return parser.parse_args()


def selected_experiments(args: argparse.Namespace) -> list[str]:
    if args.all:
        return list(FINAL_EXPERIMENT_GRID)
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


def build_seed_command(
    method: str,
    settings: dict[str, object],
    seed: int,
    run_dir: Path,
    pretrained_policy_path: str | None,
    freeze_pretrained_policy: bool,
    config_defaults: TrainConfig,
) -> list[str]:
    reward_mode = str(settings.get("reward_mode", config_defaults.reward_mode))
    run_name = f"{method}_{reward_mode}"
    if method == "BASELINE":
        command = [
            sys.executable,
            "scripts/train_baseline.py",
            "--mode",
            str(settings["mode"]),
            "--reward-mode",
            reward_mode,
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
            run_name,
        ]
        if pretrained_policy_path:
            command.extend(["--pretrained-policy-path", pretrained_policy_path])
        if freeze_pretrained_policy:
            command.append("--freeze-pretrained-policy")
        if settings["grouped_rollouts"]:
            command.append("--grouped-rollouts")
        if settings["dynamic_sampling"]:
            command.append("--dynamic-sampling")
        return command
    command = [
        sys.executable,
        "scripts/train_ac.py",
        "--method",
        method,
        "--mode",
        str(settings["mode"]),
        "--reward-mode",
        reward_mode,
        "--seed",
        str(seed),
        "--total-steps",
        str(settings["total_steps"]),
        "--policy-lr",
        str(settings.get("policy_lr", config_defaults.policy_lr)),
        "--certainty-lr",
        str(settings.get("certainty_lr", config_defaults.certainty_lr)),
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
        run_name,
    ]
    if settings["grouped_rollouts"]:
        command.append("--grouped-rollouts")
    if settings["dynamic_sampling"]:
        command.append("--dynamic-sampling")
    if pretrained_policy_path:
        command.extend(["--pretrained-policy-path", pretrained_policy_path])
    if freeze_pretrained_policy:
        command.append("--freeze-pretrained-policy")
    return command


def run_seed_commands(commands: list[list[str]], cwd: Path, max_parallel: int) -> None:
    pending = list(commands)
    running: dict[subprocess.Popen[str], list[str]] = {}
    while pending or running:
        while pending and len(running) < max_parallel:
            command = pending.pop(0)
            print("running:", " ".join(command), flush=True)
            proc = subprocess.Popen(command, cwd=cwd)
            running[proc] = command
        if not running:
            continue
        finished = []
        for proc, command in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            finished.append((proc, command, code))
        if not finished:
            time.sleep(0.2)
            continue
        for proc, command, code in finished:
            running.pop(proc, None)
            if code != 0:
                for other in list(running):
                    other.terminate()
                for other in list(running):
                    other.wait()
                raise subprocess.CalledProcessError(code, command)


def summarize_seed(seed_dir: Path, seed: int) -> dict[str, float | int]:
    episode_paths = list((seed_dir / "logs").glob("*_episodes.csv"))
    update_paths = list((seed_dir / "logs").glob("*_updates.csv"))
    row: dict[str, float | int] = {"seed": seed}
    if episode_paths:
        with episode_paths[0].open("r", newline="", encoding="utf-8") as f:
            episodes = list(csv.DictReader(f))
        tail = episodes[-20:] if len(episodes) >= 20 else episodes
        if tail:
            returns = [float(r.get("return_env", r.get("return", "nan"))) for r in tail]
            successes = [float(r.get("outcome_policy", r.get("success", "nan"))) for r in tail]
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
            if "mean_policy_entropy" in updates[0]:
                valid_entropy = [float(r["mean_policy_entropy"]) for r in updates if r["mean_policy_entropy"] != "nan"]
                row["policy_entropy_mean"] = sum(valid_entropy) / max(1, len(valid_entropy))
    eval_paths = list((seed_dir / "logs").glob("*_checkpoint_eval.csv"))
    if eval_paths:
        with eval_paths[0].open("r", newline="", encoding="utf-8") as f:
            eval_rows = list(csv.DictReader(f))
        if eval_rows:
            by_checkpoint: dict[str, list[dict[str, str]]] = {}
            for eval_row in eval_rows:
                by_checkpoint.setdefault(eval_row["checkpoint"], []).append(eval_row)
            best_name = ""
            best_return = float("-inf")
            best_success = 0.0
            for checkpoint, rows in by_checkpoint.items():
                mean_return = sum(float(r["return"]) for r in rows) / len(rows)
                mean_success = sum(float(r["success"]) for r in rows) / len(rows)
                if mean_return > best_return:
                    best_name = checkpoint
                    best_return = mean_return
                    best_success = mean_success
            row["best_eval_return"] = best_return
            row["best_eval_success"] = best_success
            row["best_eval_checkpoint"] = best_name
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
    max_parallel_seeds: int,
    reward_mode_override: str | None,
) -> Path:
    experiments = {**BASELINE_EXPERIMENTS, **METHOD_MODE_EXPERIMENTS}
    settings = dict(experiments[experiment_name])
    config_defaults = TrainConfig()
    method = str(settings.get("method", "BASELINE"))
    if total_steps_override is not None:
        settings["total_steps"] = total_steps_override
    if reward_mode_override is not None:
        settings["reward_mode"] = reward_mode_override
    run_dir = parent_dir / experiment_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "plots").mkdir()
    config_payload = {
        "experiment": experiment_name,
        "seeds": list(seeds),
        "pretrained_policy_path": pretrained_policy_path,
        "freeze_pretrained_policy": freeze_pretrained_policy,
        **settings,
    }
    write_yaml(run_dir / "config.yaml", config_payload)
    commands = [
        build_seed_command(
            method,
            settings,
            seed,
            run_dir,
            pretrained_policy_path,
            freeze_pretrained_policy,
            config_defaults,
        )
        for seed in seeds
    ]
    run_seed_commands(commands, repo, max_parallel=max_parallel_seeds)
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
    max_parallel_seeds = max(1, min(args.max_parallel_seeds or len(seeds), len(seeds)))
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    label = "all_experiments" if args.all else names[0]
    parent_dir = OUTPUTS_DIR / f"{stamp}_{label}"
    parent_dir.mkdir(parents=True, exist_ok=False)
    write_yaml(parent_dir / "config.yaml", {
        "experiments": names,
        "seeds": list(seeds),
        "total_steps_override": args.total_steps,
        "reward_mode_override": args.reward_mode,
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
            analyze_single=True,
            pretrained_policy_path=args.pretrained_policy_path,
            freeze_pretrained_policy=args.freeze_pretrained_policy,
            max_parallel_seeds=max_parallel_seeds,
            reward_mode_override=args.reward_mode,
        )
        for name in names
    ]
    summary = {
        "experiments": names,
        "run_dirs": [str(path) for path in run_dirs],
        "seeds": list(seeds),
        "reward_mode": args.reward_mode,
        "pretrained_policy_path": args.pretrained_policy_path,
        "freeze_pretrained_policy": args.freeze_pretrained_policy,
    }
    (parent_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    run_command([sys.executable, "scripts/analyze.py", "--log-dir", str(parent_dir), "--plot-dir", str(parent_dir / "plots")], repo)


if __name__ == "__main__":
    main()
