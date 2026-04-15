from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import LOG_DIR, PLOT_DIR, ensure_output_dirs
from src.metrics import safe_auroc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate exactly the required AC-GRPO diagnostic plots.")
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--plot-dir", type=Path, default=PLOT_DIR)
    return parser.parse_args()


def parse_run_name(path: Path) -> tuple[str, str, int]:
    stem = path.stem.removesuffix("_episodes").removesuffix("_steps")
    prefix, seed_part = stem.rsplit("_seed", 1)
    seed = int(seed_part)
    for mode in ("REWARD_NOISE", "OBS_NOISE", "CLEAN"):
        suffix = f"_{mode}"
        if prefix.endswith(suffix):
            return prefix[: -len(suffix)], mode, seed
    raise ValueError(f"Could not parse run name: {path.name}")


def load_logs(log_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_frames = []
    for path in log_dir.glob("*_episodes.csv"):
        df = pd.read_csv(path)
        method, mode, seed = parse_run_name(path)
        df["method"], df["mode"], df["seed"] = method, mode, seed
        episode_frames.append(df)
    step_frames = []
    for path in log_dir.glob("*_steps.csv"):
        df = pd.read_csv(path)
        method, mode, seed = parse_run_name(path)
        df["method"], df["mode"], df["seed"] = method, mode, seed
        step_frames.append(df)
    episodes = pd.concat(episode_frames, ignore_index=True) if episode_frames else pd.DataFrame()
    steps = pd.concat(step_frames, ignore_index=True) if step_frames else pd.DataFrame()
    return episodes, steps


def rolling_mean(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def save_return_vs_steps(episodes: pd.DataFrame, plot_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    for name, group in episodes.groupby(["method", "mode"]):
        g = group.sort_values("step")
        plt.plot(g["step"], rolling_mean(g["return"]), label=f"{name[0]} {name[1]}")
    plt.xlabel("steps")
    plt.ylabel("return")
    plt.title("Return vs steps")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "01_return_vs_steps.png")
    plt.close()


def save_success_vs_steps(episodes: pd.DataFrame, plot_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    for name, group in episodes.groupby(["method", "mode"]):
        g = group.sort_values("step")
        plt.plot(g["step"], rolling_mean(g["success"]), label=f"{name[0]} {name[1]}")
    plt.xlabel("steps")
    plt.ylabel("success rate")
    plt.title("Success rate vs steps")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "02_success_rate_vs_steps.png")
    plt.close()


def save_certainty_histogram(steps: pd.DataFrame, plot_dir: Path) -> None:
    data = pd.to_numeric(steps["certainty"], errors="coerce").dropna()
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=40)
    plt.xlabel("certainty")
    plt.ylabel("count")
    plt.title("Certainty histogram")
    plt.tight_layout()
    plt.savefig(plot_dir / "03_certainty_histogram.png")
    plt.close()


def save_scatter(steps: pd.DataFrame, x: str, y: str, filename: str, title: str, plot_dir: Path) -> None:
    df = steps[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) > 20_000:
        df = df.sample(20_000, random_state=42)
    plt.figure(figsize=(8, 6))
    plt.scatter(df[x], df[y], s=4, alpha=0.25)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(plot_dir / filename)
    plt.close()


def save_clean_vs_noisy_return(episodes: pd.DataFrame, plot_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    noisy = episodes[episodes["mode"].isin(["CLEAN", "REWARD_NOISE", "OBS_NOISE"])].copy()
    for name, group in noisy.groupby(["method", "mode"]):
        g = group.sort_values("step")
        plt.plot(g["step"], rolling_mean(g["return"]), label=f"{name[0]} {name[1]}")
    plt.xlabel("steps")
    plt.ylabel("return")
    plt.title("Clean vs noisy return comparison")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "06_clean_vs_noisy_return_comparison.png")
    plt.close()


def write_summary(episodes: pd.DataFrame, steps: pd.DataFrame, plot_dir: Path) -> None:
    lines = [
        "# AC-GRPO Diagnostic Summary",
        "",
        "Dispersion proxy note: AC_FULL uses policy entropy as a discrete-action surrogate, not the exact continuous Gaussian orbit likelihood.",
        "",
    ]
    if not episodes.empty and not steps.empty and "certainty" in steps:
        cert_steps = steps.copy()
        cert_steps["certainty"] = pd.to_numeric(cert_steps["certainty"], errors="coerce")
        cert_steps = cert_steps.dropna(subset=["certainty"])
        means = cert_steps.groupby(["method", "mode", "seed", "episode_id"])["certainty"].mean().reset_index(name="mean_certainty")
        merged = means.merge(episodes[["method", "mode", "seed", "episode_id", "success"]], on=["method", "mode", "seed", "episode_id"])
        auc = safe_auroc(merged["success"], merged["mean_certainty"])
        late = cert_steps.merge(episodes[["method", "mode", "seed", "episode_id", "episode_length"]], on=["method", "mode", "seed", "episode_id"])
        late["late_phase"] = late["timestep"] > 0.8 * late["episode_length"]
        step_auc = safe_auroc(late["late_phase"], 1.0 - late["certainty"])
        lines.append(f"Trajectory AUROC: {auc if not math.isnan(auc) else 'undefined one-class input'}")
        lines.append(f"Timestep AUROC: {step_auc if not math.isnan(step_auc) else 'undefined one-class input'}")
    (plot_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    episodes, steps = load_logs(args.log_dir)
    if episodes.empty:
        raise SystemExit(f"No episode logs found in {args.log_dir}")
    save_return_vs_steps(episodes, args.plot_dir)
    save_success_vs_steps(episodes, args.plot_dir)
    if not steps.empty and "certainty" in steps:
        save_certainty_histogram(steps, args.plot_dir)
        save_scatter(steps, "entropy", "certainty", "04_certainty_vs_entropy_scatter.png", "Certainty vs entropy scatter", args.plot_dir)
        save_scatter(steps, "delta", "certainty", "05_certainty_vs_delta_t_scatter.png", "Certainty vs delta_t scatter", args.plot_dir)
    save_clean_vs_noisy_return(episodes, args.plot_dir)
    write_summary(episodes, steps, args.plot_dir)


if __name__ == "__main__":
    main()
