from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import LOG_DIR, PLOT_DIR, ensure_output_dirs
from src.metrics import safe_auroc


METHOD_COLORS = {
    "BASELINE": "red",
    "AC_LITE": "green",
    "AC_FULL": "blue",
}

MODE_LINESTYLES = {
    "CLEAN": "-",
    "REWARD_NOISE": ":",
    "OBS_NOISE": "--",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate exactly the required AC-GRPO diagnostic plots.")
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--plot-dir", type=Path, default=PLOT_DIR)
    return parser.parse_args()


def parse_run_name(path: Path) -> tuple[str, str, int]:
    # Python 3.7 compatibility: str.removesuffix was added in Python 3.9
    stem = path.stem
    for suffix in ("_checkpoint_eval", "_episodes", "_steps"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    prefix, seed_part = stem.rsplit("_seed", 1)
    seed = int(seed_part)
    for mode in ("REWARD_NOISE", "OBS_NOISE", "CLEAN"):
        suffix = f"_{mode}"
        if prefix.endswith(suffix):
            return prefix[: -len(suffix)], mode, seed
    raise ValueError(f"Could not parse run name: {path.name}")


def load_logs(log_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_frames = []
    for path in log_dir.rglob("*_episodes.csv"):
        df = pd.read_csv(path)
        method, mode, seed = parse_run_name(path)
        df["method"], df["mode"], df["seed"] = method, mode, seed
        episode_frames.append(df)
    step_frames = []
    for path in log_dir.rglob("*_steps.csv"):
        df = pd.read_csv(path)
        method, mode, seed = parse_run_name(path)
        df["method"], df["mode"], df["seed"] = method, mode, seed
        step_frames.append(df)
    episodes = pd.concat(episode_frames, ignore_index=True) if episode_frames else pd.DataFrame()
    steps = pd.concat(step_frames, ignore_index=True) if step_frames else pd.DataFrame()
    return episodes, steps


def load_eval_logs(log_dir: Path) -> pd.DataFrame:
    frames = []
    for path in log_dir.rglob("*_checkpoint_eval.csv"):
        df = pd.read_csv(path)
        method, mode, seed = parse_run_name(path)
        df["method"], df["mode"], df["seed"] = method, mode, seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def rolling_mean(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def _seed_mean_curve(
    episodes: pd.DataFrame,
    y_col: str,
    window: int = 20,
    grid_points: int = 250,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mean±std curve over seeds for a given y_col.

    IMPORTANT: 'step' starts at 0 for every seed/run, so we must compute curves per
    seed and then aggregate; sorting by 'step' after concatenation will interleave
    seeds and produce misleading plots.
    """
    if episodes.empty:
        return np.array([]), np.array([]), np.array([])

    # Common x grid in [0, max_step]
    max_step = float(pd.to_numeric(episodes["step"], errors="coerce").max())
    if not np.isfinite(max_step) or max_step <= 0:
        return np.array([]), np.array([]), np.array([])
    grid = np.linspace(0.0, max_step, grid_points)

    seed_series = []
    for seed in sorted(episodes["seed"].unique().tolist()):
        g = episodes[episodes["seed"] == seed].sort_values("step")
        x = pd.to_numeric(g["step"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(g[y_col], errors="coerce")
        y = rolling_mean(y, window=window).to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if len(x) < 2:
            continue
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        # Ensure unique x for interpolation
        x_unique, idx = np.unique(x, return_index=True)
        y_unique = y[idx]
        if len(x_unique) < 2:
            continue
        seed_series.append(np.interp(grid, x_unique, y_unique))

    if not seed_series:
        return np.array([]), np.array([]), np.array([])
    mat = np.vstack(seed_series)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0, ddof=0)
    return grid, mean, std


def _format_mean_std(mean: float, std: float, digits: int = 1) -> str:
    if not (math.isfinite(mean) and math.isfinite(std)):
        return "n/a"
    fmt = f"{{:.{digits}f}}"
    return f"{fmt.format(mean)} ± {fmt.format(std)}"


def _method_color(method: str) -> str | None:
    return METHOD_COLORS.get(method)


def _mode_linestyle(mode: str) -> str:
    return MODE_LINESTYLES.get(mode, "-")


def save_return_vs_steps(episodes: pd.DataFrame, plot_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    for (method, mode), group in episodes.groupby(["method", "mode"]):
        x, mean, std = _seed_mean_curve(group, "return", window=20, grid_points=250)
        if len(x) == 0:
            continue
        label = f"{method} {mode}"
        color = _method_color(method)
        plt.plot(x, mean, label=label, color=color, linestyle=_mode_linestyle(mode))
        plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
    plt.xlabel("steps")
    plt.ylabel("return")
    plt.title("Return vs steps")
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "01_return_vs_steps.png")
    plt.close()


def save_success_vs_steps(episodes: pd.DataFrame, plot_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    for (method, mode), group in episodes.groupby(["method", "mode"]):
        x, mean, std = _seed_mean_curve(group, "success", window=20, grid_points=250)
        if len(x) == 0:
            continue
        label = f"{method} {mode}"
        color = _method_color(method)
        plt.plot(x, mean, label=label, color=color, linestyle=_mode_linestyle(mode))
        plt.fill_between(x, np.clip(mean - std, 0.0, 1.0), np.clip(mean + std, 0.0, 1.0), color=color, alpha=0.15)
    plt.xlabel("steps")
    plt.ylabel("success rate")
    plt.title("Success rate vs steps")
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "02_success_rate_vs_steps.png")
    plt.close()


def save_metric_by_mode_subplots(episodes: pd.DataFrame, y_col: str, ylabel: str, title: str, filename: str, plot_dir: Path) -> None:
    modes = ("CLEAN", "OBS_NOISE", "REWARD_NOISE")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, mode in zip(axes, modes):
        mode_df = episodes[episodes["mode"] == mode]
        for method, group in mode_df.groupby("method"):
            x, mean, std = _seed_mean_curve(group, y_col, window=20, grid_points=250)
            if len(x) == 0:
                continue
            color = _method_color(method)
            ax.plot(x, mean, label=method, color=color)
            if y_col == "success":
                lower = np.clip(mean - std, 0.0, 1.0)
                upper = np.clip(mean + std, 0.0, 1.0)
            else:
                lower = mean - std
                upper = mean + std
            ax.fill_between(x, lower, upper, color=color, alpha=0.15)
        ax.set_title(mode)
        ax.set_xlabel("steps")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(ylabel)
    handles, labels = [], []
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            break
    if handles:
        axes[-1].legend(handles, labels, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(plot_dir / filename)
    plt.close(fig)


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


def _compute_auc_tables(episodes: pd.DataFrame, steps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if episodes.empty or steps.empty or "certainty" not in steps:
        return pd.DataFrame(), pd.DataFrame()

    cert_steps = steps.copy()
    cert_steps["certainty"] = pd.to_numeric(cert_steps["certainty"], errors="coerce")
    cert_steps = cert_steps.dropna(subset=["certainty"])
    if cert_steps.empty:
        return pd.DataFrame(), pd.DataFrame()

    means = (
        cert_steps.groupby(["method", "mode", "seed", "episode_id"])["certainty"]
        .mean()
        .reset_index(name="mean_certainty")
    )
    merged = means.merge(
        episodes[["method", "mode", "seed", "episode_id", "success"]],
        on=["method", "mode", "seed", "episode_id"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    traj_series = merged.groupby(["method", "mode"]).apply(lambda g: safe_auroc(g["success"], g["mean_certainty"]))
    traj = traj_series.to_frame("trajectory_auroc").reset_index()

    late = cert_steps.merge(
        episodes[["method", "mode", "seed", "episode_id", "episode_length"]],
        on=["method", "mode", "seed", "episode_id"],
        how="inner",
    )
    if late.empty:
        return traj, pd.DataFrame()
    late["late_phase"] = pd.to_numeric(late["timestep"], errors="coerce") > 0.8 * pd.to_numeric(late["episode_length"], errors="coerce")

    step_series = late.groupby(["method", "mode"]).apply(lambda g: safe_auroc(g["late_phase"], 1.0 - g["certainty"]))
    step = step_series.to_frame("timestep_auroc").reset_index()
    return traj, step


def _compute_episode_tables(episodes: pd.DataFrame, last_n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-run (method, mode, seed) summary and aggregated (method, mode) summary.
    """
    if episodes.empty:
        return pd.DataFrame(), pd.DataFrame()

    def per_run(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("step")
        tail = g.tail(last_n)
        # best rolling window of size last_n (requires >= last_n episodes)
        if len(g) >= last_n:
            best_ret = float(pd.to_numeric(g["return"], errors="coerce").rolling(last_n, min_periods=last_n).mean().max())
            best_succ = float(pd.to_numeric(g["success"], errors="coerce").rolling(last_n, min_periods=last_n).mean().max())
        else:
            best_ret = float(pd.to_numeric(g["return"], errors="coerce").mean())
            best_succ = float(pd.to_numeric(g["success"], errors="coerce").mean())
        return pd.Series(
            {
                "episodes": len(g),
                "final_return_mean": float(pd.to_numeric(tail["return"], errors="coerce").mean()),
                "final_success_mean": float(pd.to_numeric(tail["success"], errors="coerce").mean()),
                "final_len_mean": float(pd.to_numeric(tail["episode_length"], errors="coerce").mean()),
                "best_window_return_mean": best_ret,
                "best_window_success_mean": best_succ,
            }
        )

    per_seed = episodes.groupby(["method", "mode", "seed"]).apply(per_run).reset_index()

    agg = (
        per_seed.groupby(["method", "mode"])
        .agg(
            seeds=("seed", "nunique"),
            final_return_mean=("final_return_mean", "mean"),
            final_return_std=("final_return_mean", "std"),
            final_success_mean=("final_success_mean", "mean"),
            final_success_std=("final_success_mean", "std"),
            final_len_mean=("final_len_mean", "mean"),
            best_window_return_mean=("best_window_return_mean", "mean"),
            best_window_success_mean=("best_window_success_mean", "mean"),
        )
        .reset_index()
    )

    return per_seed, agg


def write_report(episodes: pd.DataFrame, steps: pd.DataFrame, report_dir: Path) -> None:
    # Keep the original summary.md, but make it more informative + reproducible.
    lines = [
        "# RL Diagnostic Summary",
        "",
        "This summary is generated from CSV logs in the selected output folder.",
        "",
    ]

    traj_auc, step_auc = _compute_auc_tables(episodes, steps)
    per_seed, agg = _compute_episode_tables(episodes, last_n=20)
    evals = load_eval_logs(report_dir)

    if not agg.empty:
        lines += [
            "## Final metrics (last 20 episodes per run; mean ± std over seeds)",
            "",
            "| method | mode | final return | final success | best-window return | best-window success |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for _, r in agg.sort_values(["mode", "method"]).iterrows():
            lines.append(
                "| {method} | {mode} | {ret} | {succ} | {bret:.1f} | {bsucc:.3f} |".format(
                    method=r["method"],
                    mode=r["mode"],
                    ret=_format_mean_std(float(r["final_return_mean"]), float(r["final_return_std"]), digits=1),
                    succ=_format_mean_std(float(r["final_success_mean"]), float(r["final_success_std"]), digits=3),
                    bret=float(r["best_window_return_mean"]),
                    bsucc=float(r["best_window_success_mean"]),
                )
            )
        lines.append("")

    if not traj_auc.empty or not step_auc.empty:
        auc_table = traj_auc.merge(step_auc, on=["method", "mode"], how="outer")
        lines += [
            "## AUROC diagnostics (by method × mode)",
            "",
            "| method | mode | trajectory AUROC (success ~ mean certainty) | timestep AUROC (late_phase ~ 1-certainty) |",
            "|---|---|---:|---:|",
        ]
        for _, r in auc_table.sort_values(["mode", "method"]).iterrows():
            ta = r.get("trajectory_auroc", math.nan)
            sa = r.get("timestep_auroc", math.nan)
            lines.append(
                "| {method} | {mode} | {ta} | {sa} |".format(
                    method=r["method"],
                    mode=r["mode"],
                    ta=("n/a" if not math.isfinite(float(ta)) else f"{float(ta):.3f}"),
                    sa=("n/a" if not math.isfinite(float(sa)) else f"{float(sa):.3f}"),
                )
            )
        lines.append("")

    # summary.md is intentionally not written; report.md contains the full summary.

    # Paper-ready extended report.
    report = []
    report += [
        "# RL Experiment Report",
        "",
        "This report summarizes the selected sweep from the CSV logs.",
        "",
        f"Source folder: `{report_dir}`",
        "",
        "Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.",
        "",
        "## Notes on experimental modes",
        "",
        "- **Reward semantics**: PPO/GAE uses sparse terminal binary reward only (`0` before termination, terminal `policy_success` at episode end); dense LunarLander return is logged for diagnostics only.",
        "- **REWARD_NOISE**: false-negative successes set terminal `policy_success` to `0`, so the sparse policy update sees the corrupted outcome directly.",
        "- **OBS_NOISE**: adds Gaussian noise \(\\sigma=0.1\\) to observations at every step.",
        "",
        "## Seed aggregation",
        "",
        "Learning curves are computed **per seed** and then aggregated (mean ± std). This avoids interleaving seeds (each run resets `step` to 0).",
        "",
    ]

    if not agg.empty:
        report += [
            "## Summary table (mean ± std over 5 seeds)",
            "",
            "| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for _, r in agg.sort_values(["mode", "method"]).iterrows():
            report.append(
                "| {mode} | {method} | {ret} | {succ} | {bret:.1f} | {bsucc:.3f} |".format(
                    mode=r["mode"],
                    method=r["method"],
                    ret=_format_mean_std(float(r["final_return_mean"]), float(r["final_return_std"]), digits=1),
                    succ=_format_mean_std(float(r["final_success_mean"]), float(r["final_success_std"]), digits=3),
                    bret=float(r["best_window_return_mean"]),
                    bsucc=float(r["best_window_success_mean"]),
                )
            )
        report.append("")

    if not per_seed.empty:
        report += [
            "## Per-seed finals (last 20 episodes)",
            "",
            "| mode | method | seed | final return | final success |",
            "|---|---|---:|---:|---:|",
        ]
        for _, r in per_seed.sort_values(["mode", "method", "seed"]).iterrows():
            report.append(
                "| {mode} | {method} | {seed} | {ret:.1f} | {succ:.3f} |".format(
                    mode=r["mode"],
                    method=r["method"],
                    seed=int(r["seed"]),
                    ret=float(r["final_return_mean"]),
                    succ=float(r["final_success_mean"]),
                )
            )
        report.append("")

    if not evals.empty:
        eval_summary = (
            evals.groupby(["method", "mode", "seed", "checkpoint"])
            .agg(eval_return=("return", "mean"), eval_success=("success", "mean"), eval_length=("episode_length", "mean"))
            .reset_index()
        )
        best_rows = eval_summary.sort_values("eval_return", ascending=False).groupby(["method", "mode", "seed"]).head(1)
        report += [
            "## Best checkpoint by greedy held-out evaluation",
            "",
            "Checkpoints are evaluated greedily on fixed held-out seeds. The final checkpoint is not assumed to be best.",
            "",
            "| mode | method | seed | checkpoint | eval return | eval success |",
            "|---|---|---:|---|---:|---:|",
        ]
        for _, r in best_rows.sort_values(["mode", "method", "seed"]).iterrows():
            report.append(
                "| {mode} | {method} | {seed} | {checkpoint} | {ret:.1f} | {succ:.3f} |".format(
                    mode=r["mode"],
                    method=r["method"],
                    seed=int(r["seed"]),
                    checkpoint=r["checkpoint"],
                    ret=float(r["eval_return"]),
                    succ=float(r["eval_success"]),
                )
            )
        report.append("")

    if not traj_auc.empty or not step_auc.empty:
        auc_table = traj_auc.merge(step_auc, on=["method", "mode"], how="outer")
        report += [
            "## Certainty AUROC diagnostics",
            "",
            "- **Trajectory AUROC**: episode success predicted by mean certainty over the trajectory.",
            "- **Timestep AUROC**: late-phase indicator predicted by \(1 - certainty\\) (diagnostic).",
            "",
            "| mode | method | trajectory AUROC | timestep AUROC |",
            "|---|---|---:|---:|",
        ]
        for _, r in auc_table.sort_values(["mode", "method"]).iterrows():
            ta = r.get("trajectory_auroc", math.nan)
            sa = r.get("timestep_auroc", math.nan)
            report.append(
                "| {mode} | {method} | {ta} | {sa} |".format(
                    mode=r["mode"],
                    method=r["method"],
                    ta=("n/a" if not math.isfinite(float(ta)) else f"{float(ta):.3f}"),
                    sa=("n/a" if not math.isfinite(float(sa)) else f"{float(sa):.3f}"),
                )
            )
        report.append("")

    report += [
        "## Plots",
        "",
        "The following plots are generated in the `plots/` subfolder:",
        "",
        "1. `01_return_vs_steps.png`",
        "2. `02_success_rate_vs_steps.png`",
        "3. `03_certainty_histogram.png`",
        "4. `04_certainty_vs_entropy_scatter.png`",
        "5. `05_certainty_vs_delta_t_scatter.png`",
        "6. `06_return_by_mode_subplots.png`",
        "7. `07_success_by_mode_subplots.png`",
        "",
    ]
    (report_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    episodes, steps = load_logs(args.log_dir)
    if episodes.empty:
        raise SystemExit(f"No episode logs found in {args.log_dir}")
    save_return_vs_steps(episodes, args.plot_dir)
    save_success_vs_steps(episodes, args.plot_dir)
    save_metric_by_mode_subplots(episodes, "return", "return", "Return vs steps by mode", "06_return_by_mode_subplots.png", args.plot_dir)
    save_metric_by_mode_subplots(episodes, "success", "success rate", "Success rate vs steps by mode", "07_success_by_mode_subplots.png", args.plot_dir)
    if not steps.empty and "certainty" in steps:
        save_certainty_histogram(steps, args.plot_dir)
        save_scatter(steps, "entropy", "certainty", "04_certainty_vs_entropy_scatter.png", "Certainty vs entropy scatter", args.plot_dir)
        save_scatter(steps, "delta", "certainty", "05_certainty_vs_delta_t_scatter.png", "Certainty vs delta_t scatter", args.plot_dir)
    write_report(episodes, steps, args.log_dir)


if __name__ == "__main__":
    main()
