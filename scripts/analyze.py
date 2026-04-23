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
    "BASELINE_SPARSE": "#c0392b",
    "BASELINE_DENSE": "#e67e22",
    "AC_LITE_SPARSE": "#2e8b57",
    "AC_LITE_DENSE": "#16a085",
    "AC_FULL_SPARSE": "#1f77b4",
}

MODE_LINESTYLES = {
    "CLEAN": "-",
    "REWARD_NOISE": ":",
    "OBS_NOISE": "--",
}

METHOD_ORDER = (
    "BASELINE_SPARSE",
    "BASELINE_DENSE",
    "AC_LITE_SPARSE",
    "AC_LITE_DENSE",
    "AC_FULL_SPARSE",
)

MODE_ORDER = ("CLEAN", "OBS_NOISE", "REWARD_NOISE")

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AC-PPO diagnostic plots and report.")
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--plot-dir", type=Path, default=PLOT_DIR)
    return parser.parse_args()


def _split_method_reward(core: str) -> tuple[str, str]:
    for reward_mode in ("SPARSE", "DENSE"):
        suffix = f"_{reward_mode}"
        if core.endswith(suffix):
            return core[: -len(suffix)], reward_mode
    return core, "SPARSE"


def parse_run_name(path: Path) -> tuple[str, str, str, int, str]:
    stem = path.stem
    for suffix in ("_checkpoint_eval", "_episodes", "_steps", "_updates", "_summary"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    prefix, seed_part = stem.rsplit("_seed", 1)
    seed = int(seed_part)
    for mode in ("REWARD_NOISE", "OBS_NOISE", "CLEAN"):
        mode_suffix = f"_{mode}"
        if prefix.endswith(mode_suffix):
            core = prefix[: -len(mode_suffix)]
            method_family, reward_mode = _split_method_reward(core)
            method_label = f"{method_family}_{reward_mode}"
            return method_family, reward_mode, mode, seed, method_label
    raise ValueError(f"Could not parse run name: {path.name}")


def _coerce_episode_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "return_env" in df.columns and "return" not in df.columns:
        df["return"] = pd.to_numeric(df["return_env"], errors="coerce")
    if "outcome_policy" in df.columns and "success" not in df.columns:
        df["success"] = pd.to_numeric(df["outcome_policy"], errors="coerce")
    if "outcome_raw" in df.columns and "raw_success" not in df.columns:
        df["raw_success"] = pd.to_numeric(df["outcome_raw"], errors="coerce")
    return df


def load_logs(log_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    episode_frames = []
    step_frames = []
    update_frames = []
    for path in log_dir.rglob("*_episodes.csv"):
        df = _coerce_episode_columns(pd.read_csv(path))
        method_family, reward_mode, mode, seed, method_label = parse_run_name(path)
        df["method_family"] = method_family
        df["reward_mode"] = reward_mode
        df["method"] = method_label
        df["mode"] = mode
        df["seed"] = seed
        episode_frames.append(df)
    for path in log_dir.rglob("*_steps.csv"):
        df = pd.read_csv(path)
        method_family, reward_mode, mode, seed, method_label = parse_run_name(path)
        df["method_family"] = method_family
        df["reward_mode"] = reward_mode
        df["method"] = method_label
        df["mode"] = mode
        df["seed"] = seed
        step_frames.append(df)
    for path in log_dir.rglob("*_updates.csv"):
        df = pd.read_csv(path)
        method_family, reward_mode, mode, seed, method_label = parse_run_name(path)
        df["method_family"] = method_family
        df["reward_mode"] = reward_mode
        df["method"] = method_label
        df["mode"] = mode
        df["seed"] = seed
        update_frames.append(df)
    episodes = pd.concat(episode_frames, ignore_index=True) if episode_frames else pd.DataFrame()
    steps = pd.concat(step_frames, ignore_index=True) if step_frames else pd.DataFrame()
    updates = pd.concat(update_frames, ignore_index=True) if update_frames else pd.DataFrame()
    return episodes, steps, updates


def load_eval_logs(log_dir: Path) -> pd.DataFrame:
    frames = []
    for path in log_dir.rglob("*_checkpoint_eval.csv"):
        df = pd.read_csv(path)
        if "eval_name" not in df.columns:
            df["eval_name"] = "selection"
        if "eval_mode" not in df.columns:
            _, _, mode, _, _ = parse_run_name(path)
            df["eval_mode"] = mode
        if "eval_obs_noise_sigma" not in df.columns:
            df["eval_obs_noise_sigma"] = np.nan
        method_family, reward_mode, mode, seed, method_label = parse_run_name(path)
        df["method_family"] = method_family
        df["reward_mode"] = reward_mode
        df["method"] = method_label
        df["mode"] = mode
        df["seed"] = seed
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
    if episodes.empty:
        return np.array([]), np.array([]), np.array([])
    max_step = float(pd.to_numeric(episodes["step"], errors="coerce").max())
    if not np.isfinite(max_step) or max_step <= 0:
        return np.array([]), np.array([]), np.array([])
    grid = np.linspace(0.0, max_step, grid_points)
    seed_series = []
    for seed in sorted(episodes["seed"].unique().tolist()):
        g = episodes[episodes["seed"] == seed].sort_values("step")
        x = pd.to_numeric(g["step"], errors="coerce").to_numpy(dtype=float)
        y = rolling_mean(pd.to_numeric(g[y_col], errors="coerce"), window=window).to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if len(x) < 2:
            continue
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        x_unique, idx = np.unique(x, return_index=True)
        y_unique = y[idx]
        if len(x_unique) < 2:
            continue
        seed_series.append(np.interp(grid, x_unique, y_unique))
    if not seed_series:
        return np.array([]), np.array([]), np.array([])
    mat = np.vstack(seed_series)
    return grid, mat.mean(axis=0), mat.std(axis=0, ddof=0)


def _compute_reward_auc_tables(episodes: pd.DataFrame, window: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    if episodes.empty:
        return pd.DataFrame(), pd.DataFrame()

    per_seed_rows = []
    for (method, mode, seed), group in episodes.groupby(["method", "mode", "seed"]):
        g = group.sort_values("step")
        x = pd.to_numeric(g["step"], errors="coerce").to_numpy(dtype=float)
        y = rolling_mean(pd.to_numeric(g["return"], errors="coerce"), window=window).to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if len(x) < 2:
            auc = math.nan
        else:
            order = np.argsort(x)
            x = x[order]
            y = y[order]
            x_unique, idx = np.unique(x, return_index=True)
            y_unique = y[idx]
            auc = math.nan
            if len(x_unique) >= 2 and x_unique[-1] > x_unique[0]:
                auc = float(np.trapz(y_unique, x_unique) / (x_unique[-1] - x_unique[0]))
        per_seed_rows.append({"method": method, "mode": mode, "seed": seed, "reward_auc": auc})

    per_seed = pd.DataFrame(per_seed_rows)
    if per_seed.empty:
        return pd.DataFrame(), pd.DataFrame()
    agg = (
        per_seed.groupby(["method", "mode"])
        .agg(reward_auc_mean=("reward_auc", "mean"), reward_auc_std=("reward_auc", "std"))
        .reset_index()
    )
    return per_seed, agg


def _format_mean_std(mean: float, std: float, digits: int = 1) -> str:
    if math.isfinite(mean) and not math.isfinite(std):
        std = 0.0
    if not (math.isfinite(mean) and math.isfinite(std)):
        return "n/a"
    fmt = f"{{:.{digits}f}}"
    return f"{fmt.format(mean)} ± {fmt.format(std)}"


def _ordered_groups(df: pd.DataFrame, group_cols: list[str]) -> list[tuple[tuple[object, ...], pd.DataFrame]]:
    groups = []
    for key, group in df.groupby(group_cols):
        groups.append((key if isinstance(key, tuple) else (key,), group))
    method_idx = {name: idx for idx, name in enumerate(METHOD_ORDER)}
    mode_idx = {name: idx for idx, name in enumerate(MODE_ORDER)}
    def sort_key(item: tuple[tuple[object, ...], pd.DataFrame]) -> tuple[int, int]:
        key = item[0]
        if len(key) == 2:
            method, mode = key
        else:
            method = key[0]
            mode = "CLEAN"
        return method_idx.get(str(method), 999), mode_idx.get(str(mode), 999)
    return sorted(groups, key=sort_key)


def _method_linewidth(method: str) -> float:
    return 3.0 if str(method).endswith("_DENSE") else 2.0


def _auc_label(method: str, mode: str, reward_auc: pd.DataFrame) -> str:
    row = reward_auc[(reward_auc["method"] == method) & (reward_auc["mode"] == mode)]
    if row.empty:
        return f"{method} {mode}"
    mean = float(row.iloc[0]["reward_auc_mean"])
    std = float(row.iloc[0]["reward_auc_std"])
    return f"{method} {mode} | AUC {_format_mean_std(mean, std, digits=1)}"


def save_return_vs_steps(episodes: pd.DataFrame, reward_auc: pd.DataFrame, plot_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    for (method, mode), group in _ordered_groups(episodes, ["method", "mode"]):
        x, mean, std = _seed_mean_curve(group, "return", window=20, grid_points=250)
        if len(x) == 0:
            continue
        color = METHOD_COLORS.get(str(method))
        plt.plot(
            x,
            mean,
            label=_auc_label(str(method), str(mode), reward_auc),
            color=color,
            linestyle=MODE_LINESTYLES.get(str(mode), "-"),
            linewidth=_method_linewidth(str(method)),
        )
        plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
    plt.xlabel("steps")
    plt.ylabel("return")
    plt.title("Return vs steps")
    plt.grid(alpha=0.25)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "01_return_vs_steps.png")
    plt.close()


def save_success_vs_steps(episodes: pd.DataFrame, reward_auc: pd.DataFrame, plot_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    for (method, mode), group in _ordered_groups(episodes, ["method", "mode"]):
        x, mean, std = _seed_mean_curve(group, "success", window=20, grid_points=250)
        if len(x) == 0:
            continue
        color = METHOD_COLORS.get(str(method))
        plt.plot(
            x,
            mean,
            label=_auc_label(str(method), str(mode), reward_auc),
            color=color,
            linestyle=MODE_LINESTYLES.get(str(mode), "-"),
            linewidth=_method_linewidth(str(method)),
        )
        plt.fill_between(x, np.clip(mean - std, 0.0, 1.0), np.clip(mean + std, 0.0, 1.0), color=color, alpha=0.15)
    plt.xlabel("steps")
    plt.ylabel("success rate")
    plt.title("Success rate vs steps")
    plt.grid(alpha=0.25)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "02_success_rate_vs_steps.png")
    plt.close()


def save_metric_by_mode_subplots(
    episodes: pd.DataFrame,
    reward_auc: pd.DataFrame,
    y_col: str,
    ylabel: str,
    title: str,
    filename: str,
    plot_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    for ax, mode in zip(axes, MODE_ORDER):
        mode_df = episodes[episodes["mode"] == mode]
        for (method,), group in _ordered_groups(mode_df, ["method"]):
            x, mean, std = _seed_mean_curve(group, y_col, window=20, grid_points=250)
            if len(x) == 0:
                continue
            color = METHOD_COLORS.get(str(method))
            auc_row = reward_auc[(reward_auc["method"] == str(method)) & (reward_auc["mode"] == mode)]
            auc_text = ""
            if not auc_row.empty:
                auc_text = f" | AUC {_format_mean_std(float(auc_row.iloc[0]['reward_auc_mean']), float(auc_row.iloc[0]['reward_auc_std']), digits=1)}"
            ax.plot(x, mean, label=f"{method}{auc_text}", color=color, linewidth=_method_linewidth(str(method)))
            lower = np.clip(mean - std, 0.0, 1.0) if y_col == "success" else mean - std
            upper = np.clip(mean + std, 0.0, 1.0) if y_col == "success" else mean + std
            ax.fill_between(x, lower, upper, color=color, alpha=0.15)
        ax.set_title(mode)
        ax.set_xlabel("steps")
        ax.grid(alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=8)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(plot_dir / filename)
    plt.close(fig)


def save_certainty_histogram(episodes: pd.DataFrame, steps: pd.DataFrame, plot_dir: Path) -> None:
    if steps.empty or "certainty" not in steps.columns:
        return
    cert_df = steps.copy()
    cert_df["certainty"] = pd.to_numeric(cert_df["certainty"], errors="coerce")
    cert_df = cert_df.merge(
        episodes[["method", "mode", "seed", "episode_id", "success"]],
        on=["method", "mode", "seed", "episode_id"],
        how="left",
    )
    cert_df["success"] = pd.to_numeric(cert_df["success"], errors="coerce")
    cert_df = cert_df.dropna(subset=["certainty", "success"])
    if cert_df.empty:
        return
    x_min = float(cert_df["certainty"].min())
    x_max = float(cert_df["certainty"].max())
    if not math.isfinite(x_min) or not math.isfinite(x_max):
        return
    if x_max <= x_min:
        x_min, x_max = max(0.0, x_min - 0.05), min(1.0, x_max + 0.05)
    pad = max(0.01, 0.05 * (x_max - x_min))
    x_low = max(0.0, x_min - pad)
    x_high = min(1.0, x_max + pad)
    bins = np.linspace(x_low, x_high, 81)
    success_vals = cert_df[cert_df["success"] >= 0.5]["certainty"].to_numpy(dtype=float)
    fail_vals = cert_df[cert_df["success"] < 0.5]["certainty"].to_numpy(dtype=float)
    plt.figure(figsize=(8, 6))
    grid = np.linspace(x_low, x_high, 300)
    bin_width = bins[1] - bins[0]

    def smooth_counts(values: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return np.zeros_like(grid)
        bw = max(0.01, np.std(values) * 0.2)
        density = np.exp(-0.5 * ((grid[:, None] - values[None, :]) / bw) ** 2).sum(axis=1)
        density /= max(len(values), 1) * bw * math.sqrt(2.0 * math.pi)
        return density * len(values) * bin_width

    if len(success_vals):
        plt.hist(success_vals, bins=bins, color="blue", alpha=0.40, label="success")
        plt.plot(grid, smooth_counts(success_vals), color="blue", linewidth=1.2)
    if len(fail_vals):
        plt.hist(fail_vals, bins=bins, color="red", alpha=0.40, label="failure")
        plt.plot(grid, smooth_counts(fail_vals), color="red", linewidth=1.2)
    plt.xlabel("timestep certainty")
    plt.ylabel("count")
    plt.title("Timestep certainty histogram")
    plt.xlim(x_low, x_high)
    plt.grid(alpha=0.25)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "03_certainty_histogram.png")
    plt.close()


def save_scatter(steps: pd.DataFrame, x: str, y: str, filename: str, title: str, plot_dir: Path) -> None:
    if x not in steps.columns or y not in steps.columns:
        return
    df = steps[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if df.empty:
        return
    if len(df) > 20_000:
        df = df.sample(20_000, random_state=42)
    plt.figure(figsize=(8, 6))
    plt.scatter(df[x], df[y], s=4, alpha=0.25)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(plot_dir / filename)
    plt.close()


def _compute_auc_tables(episodes: pd.DataFrame, steps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if episodes.empty or steps.empty or "certainty" not in steps.columns:
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
    traj = (
        merged.groupby(["method", "mode"])
        .apply(lambda g: safe_auroc(g["success"], g["mean_certainty"]))
        .to_frame("trajectory_auroc")
        .reset_index()
    )
    late = cert_steps.merge(
        episodes[["method", "mode", "seed", "episode_id", "episode_length"]],
        on=["method", "mode", "seed", "episode_id"],
        how="inner",
    )
    if late.empty:
        return traj, pd.DataFrame()
    late["late_phase"] = pd.to_numeric(late["timestep"], errors="coerce") > 0.8 * pd.to_numeric(late["episode_length"], errors="coerce")
    step = (
        late.groupby(["method", "mode"])
        .apply(lambda g: safe_auroc(g["late_phase"], 1.0 - g["certainty"]))
        .to_frame("timestep_auroc")
        .reset_index()
    )
    return traj, step


def _compute_episode_tables(episodes: pd.DataFrame, last_n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    if episodes.empty:
        return pd.DataFrame(), pd.DataFrame()

    def per_run(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("step")
        tail = g.tail(last_n)
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


def _compute_certainty_episode_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    needed = ["mean_certainty", "certainty_delta_corr", "certainty_action_prob_corr", "certainty_runner_up_prob_corr"]
    if episodes.empty or any(col not in episodes.columns for col in needed):
        return pd.DataFrame()
    df = episodes.copy()
    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["mean_certainty"], how="any")
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(["method", "mode"])
        .agg(
            mean_episode_certainty=("mean_certainty", "mean"),
            certainty_delta_corr_mean=("certainty_delta_corr", "mean"),
            certainty_action_prob_corr_mean=("certainty_action_prob_corr", "mean"),
            certainty_runner_up_prob_corr_mean=("certainty_runner_up_prob_corr", "mean"),
        )
        .reset_index()
    )


def _method_display_name(method: str) -> str:
    mapping = {
        "BASELINE_SPARSE": "Baseline (Sparse)",
        "BASELINE_DENSE": "Baseline (Dense)",
        "AC_LITE_SPARSE": "AC-LITE (Sparse)",
        "AC_LITE_DENSE": "AC-LITE (Dense)",
        "AC_FULL_SPARSE": "AC-FULL (Sparse)",
    }
    return mapping.get(method, method)


def _selection_best_rows(evals: pd.DataFrame) -> pd.DataFrame:
    if evals.empty:
        return pd.DataFrame()
    selection = evals[evals["eval_name"] == "selection"].copy()
    if selection.empty:
        return pd.DataFrame()
    summary = (
        selection.groupby(["method", "mode", "seed", "checkpoint", "eval_name", "eval_mode", "eval_obs_noise_sigma"])
        .agg(eval_return=("return", "mean"), eval_success=("success", "mean"), eval_length=("episode_length", "mean"))
        .reset_index()
    )
    summary["anchor_preference"] = (summary["checkpoint"] == "checkpoint_0_pretrained").astype(int)
    return (
        summary.sort_values(["eval_return", "anchor_preference"], ascending=[False, False])
        .groupby(["method", "mode", "seed"])
        .head(1)
        .drop(columns=["anchor_preference"])
    )


def _challenge_summary(evals: pd.DataFrame) -> pd.DataFrame:
    if evals.empty:
        return pd.DataFrame()
    challenge = evals[evals["eval_name"] != "selection"].copy()
    if challenge.empty:
        return pd.DataFrame()
    return (
        challenge.groupby(["method", "mode", "checkpoint", "eval_name", "eval_mode", "eval_obs_noise_sigma"])
        .agg(
            eval_return_mean=("return", "mean"),
            eval_return_std=("return", "std"),
            eval_success_mean=("success", "mean"),
            eval_success_std=("success", "std"),
            episodes=("episode", "count"),
        )
        .reset_index()
    )


def _best_checkpoint_challenge(evals: pd.DataFrame, best_rows: pd.DataFrame) -> pd.DataFrame:
    if evals.empty or best_rows.empty:
        return pd.DataFrame()
    merged = evals.merge(best_rows[["method", "mode", "seed", "checkpoint"]], on=["method", "mode", "seed", "checkpoint"], how="inner")
    merged = merged[merged["eval_name"] != "selection"]
    if merged.empty:
        return pd.DataFrame()
    return (
        merged.groupby(["method", "mode", "eval_name", "eval_mode", "eval_obs_noise_sigma"])
        .agg(
            eval_return_mean=("return", "mean"),
            eval_return_std=("return", "std"),
            eval_success_mean=("success", "mean"),
            eval_success_std=("success", "std"),
            episodes=("episode", "count"),
        )
        .reset_index()
    )


def _checkpoint0_win_rate(best_rows: pd.DataFrame) -> pd.DataFrame:
    if best_rows.empty:
        return pd.DataFrame()
    df = best_rows.copy()
    df["anchor_won"] = (df["checkpoint"] == "checkpoint_0_pretrained").astype(float)
    return (
        df.groupby(["method", "mode"])
        .agg(anchor_win_fraction=("anchor_won", "mean"), anchor_win_count=("anchor_won", "sum"), seeds=("seed", "nunique"))
        .reset_index()
    )


def _lines_from_protocol(episodes: pd.DataFrame, evals: pd.DataFrame) -> list[str]:
    methods = ", ".join(sorted(episodes["method"].dropna().unique().tolist())) if not episodes.empty else "n/a"
    modes = ", ".join(sorted(episodes["mode"].dropna().unique().tolist())) if not episodes.empty else "n/a"
    reward_modes = ", ".join(sorted(episodes["reward_mode"].dropna().unique().tolist())) if not episodes.empty else "n/a"
    seeds = ", ".join(str(int(v)) for v in sorted(episodes["seed"].dropna().unique().tolist())) if not episodes.empty else "n/a"
    challenge = evals[evals["eval_name"] != "selection"] if not evals.empty else pd.DataFrame()
    challenge_episodes = int(challenge.groupby(["eval_name", "eval_seed"]).size().max()) if not challenge.empty else 0
    return [
        "## Experiment protocol",
        "",
        f"- Implemented method variants in this run: {methods}",
        f"- Training modes: {modes}",
        f"- Reward modes represented: {reward_modes}",
        f"- Training seeds: {seeds}",
        "- All branches start from the shared pretrained anchor when a pretrained path is provided.",
        "- Training uses grouped rollouts with dynamic sampling fallback when no mixed-outcome groups are available.",
        "- Checkpoints are saved during training and the pretrained anchor is treated as checkpoint 0.",
        "- Checkpoint selection uses greedy held-out evaluation under the branch's primary selection condition.",
        "- Additional challenge tests evaluate selected checkpoints under CLEAN, OBS_NOISE (typical), and OBS_NOISE (hard).",
        f"- Challenge tests currently use up to {challenge_episodes} episodes per evaluation seed in the generated logs." if challenge_episodes else "- Challenge-test episodes were not found in the current logs.",
        "",
    ]


def _lines_from_auto_analysis(agg: pd.DataFrame, best_rows: pd.DataFrame, best_challenge: pd.DataFrame, certainty_summary: pd.DataFrame) -> list[str]:
    lines = ["## Result analysis", ""]
    if not agg.empty:
        lines.append("The final training-window summary shows the following strongest method per mode:")
        lines.append("")
        for mode in MODE_ORDER:
            mode_df = agg[agg["mode"] == mode]
            if mode_df.empty:
                continue
            top = mode_df.sort_values("final_return_mean", ascending=False).iloc[0]
            lines.append(
                f"- {mode}: `{top['method']}` has the highest mean final return ({top['final_return_mean']:.1f}) with mean final success {top['final_success_mean']:.3f}."
            )
        lines.append("")
    if not best_rows.empty:
        anchor = _checkpoint0_win_rate(best_rows)
        if not anchor.empty:
            lines.append("Checkpoint selection versus the pretrained anchor:")
            lines.append("")
            for _, row in anchor.sort_values(["mode", "method"]).iterrows():
                lines.append(
                    f"- {row['mode']} / {row['method']}: checkpoint 0 wins in {int(row['anchor_win_count'])} of {int(row['seeds'])} seeds ({row['anchor_win_fraction']:.2f})."
                )
            lines.append("")
    if not best_challenge.empty:
        lines.append("Best-checkpoint challenge testing:")
        lines.append("")
        for eval_name in ("test_clean", "test_obs_noise", "test_obs_noise_hard"):
            eval_df = best_challenge[best_challenge["eval_name"] == eval_name]
            if eval_df.empty:
                continue
            top = eval_df.sort_values("eval_return_mean", ascending=False).iloc[0]
            sigma = top["eval_obs_noise_sigma"]
            sigma_text = "" if not math.isfinite(float(sigma)) else f", sigma={float(sigma):.2f}"
            lines.append(
                f"- {eval_name} ({top['eval_mode']}{sigma_text}): best mean return is `{top['method']}` in `{top['mode']}` with {top['eval_return_mean']:.1f} ± {top['eval_return_std']:.1f} and success {top['eval_success_mean']:.3f} ± {top['eval_success_std']:.3f}."
            )
        lines.append("")
    if not certainty_summary.empty:
        lines.append("Episode-level certainty behavior:")
        lines.append("")
        for _, row in certainty_summary.sort_values(["mode", "method"]).iterrows():
            lines.append(
                f"- {row['mode']} / {row['method']}: mean episode certainty {row['mean_episode_certainty']:.3f}, mean corr(certainty, delta) {row['certainty_delta_corr_mean']:.3f}, mean corr(certainty, action_prob) {row['certainty_action_prob_corr_mean']:.3f}, mean corr(certainty, runner_up_prob) {row['certainty_runner_up_prob_corr_mean']:.3f}."
            )
        lines.append("")
    return lines


def _cross_test_obs_table(best_challenge: pd.DataFrame) -> pd.DataFrame:
    if best_challenge.empty:
        return pd.DataFrame()
    mapping = {
        "CLEAN": "Train: CLEAN → Test OBS",
        "OBS_NOISE": "Train: OBS → Test OBS",
        "REWARD_NOISE": "Train: REWARD → Test OBS",
    }
    obs = best_challenge[best_challenge["eval_name"] == "test_obs_noise"].copy()
    if obs.empty:
        return pd.DataFrame()
    obs["column"] = obs["mode"].map(mapping)
    obs["display_method"] = obs["method"].map(_method_display_name)
    return obs


def write_report(episodes: pd.DataFrame, steps: pd.DataFrame, updates: pd.DataFrame, report_dir: Path) -> None:
    evals = load_eval_logs(report_dir)
    traj_auc, step_auc = _compute_auc_tables(episodes, steps)
    per_seed, agg = _compute_episode_tables(episodes, last_n=20)
    reward_auc_per_seed, reward_auc = _compute_reward_auc_tables(episodes, window=20)
    certainty_summary = _compute_certainty_episode_summary(episodes)
    best_rows = _selection_best_rows(evals)
    best_challenge = _best_checkpoint_challenge(evals, best_rows)

    report = [
        "# RL Experiment Report",
        "",
        "This report summarizes the selected sweep from the generated CSV logs.",
        "",
        f"Source folder: `{report_dir}`",
        "",
        "Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.",
        "",
    ]
    report += _lines_from_protocol(episodes, evals)

    if not agg.empty:
        report += [
            "## Summary table (mean ± std over seeds)",
            "",
            "| mode | method | final return (last 20 eps) | final success (last 20 eps) | reward AUC | best rolling-20 return | best rolling-20 success |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        merged = agg.merge(reward_auc, on=["method", "mode"], how="left")
        for _, r in merged.sort_values(["mode", "method"]).iterrows():
            report.append(
                "| {mode} | {method} | {ret} | {succ} | {auc} | {bret:.1f} | {bsucc:.3f} |".format(
                    mode=r["mode"],
                    method=r["method"],
                    ret=_format_mean_std(float(r["final_return_mean"]), float(r["final_return_std"]), digits=1),
                    succ=_format_mean_std(float(r["final_success_mean"]), float(r["final_success_std"]), digits=3),
                    auc=_format_mean_std(float(r.get("reward_auc_mean", math.nan)), float(r.get("reward_auc_std", math.nan)), digits=1),
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

    if not best_rows.empty:
        report += [
            "## Best checkpoint by primary greedy held-out selection",
            "",
            "| mode | method | seed | checkpoint | eval mode | eval return | eval success |",
            "|---|---|---:|---|---|---:|---:|",
        ]
        for _, r in best_rows.sort_values(["mode", "method", "seed"]).iterrows():
            report.append(
                "| {mode} | {method} | {seed} | {checkpoint} | {eval_mode} | {ret:.1f} | {succ:.3f} |".format(
                    mode=r["mode"],
                    method=r["method"],
                    seed=int(r["seed"]),
                    checkpoint=r["checkpoint"],
                    eval_mode=r["eval_mode"],
                    ret=float(r["eval_return"]),
                    succ=float(r["eval_success"]),
                )
            )
        report.append("")

    if not best_challenge.empty:
        report += [
            "## Best-checkpoint challenge tests",
            "",
            "| training mode | method | test condition | eval mode | obs sigma | return | success |",
            "|---|---|---|---|---:|---:|---:|",
        ]
        for _, r in best_challenge.sort_values(["mode", "method", "eval_name"]).iterrows():
            sigma = float(r["eval_obs_noise_sigma"]) if math.isfinite(float(r["eval_obs_noise_sigma"])) else math.nan
            report.append(
                "| {mode} | {method} | {eval_name} | {eval_mode} | {sigma} | {ret} | {succ} |".format(
                    mode=r["mode"],
                    method=r["method"],
                    eval_name=r["eval_name"],
                    eval_mode=r["eval_mode"],
                    sigma=("n/a" if not math.isfinite(sigma) else f"{sigma:.2f}"),
                    ret=_format_mean_std(float(r["eval_return_mean"]), float(r["eval_return_std"]), digits=1),
                    succ=_format_mean_std(float(r["eval_success_mean"]), float(r["eval_success_std"]), digits=3),
                )
            )
        report.append("")

    cross_obs = _cross_test_obs_table(best_challenge)
    if not cross_obs.empty:
        columns = ["Train: CLEAN → Test OBS", "Train: OBS → Test OBS", "Train: REWARD → Test OBS"]
        report += [
            "## Cross-test summary on OBS evaluation",
            "",
            "| Method | Train: CLEAN → Test OBS | Train: OBS → Test OBS | Train: REWARD → Test OBS |",
            "|---|---:|---:|---:|",
        ]
        for method in [_method_display_name(m) for m in METHOD_ORDER]:
            row = [f"| {method} "]
            method_df = cross_obs[cross_obs["display_method"] == method]
            for col in columns:
                cell_df = method_df[method_df["column"] == col]
                if cell_df.empty:
                    row.append("| n/a ")
                else:
                    r = cell_df.iloc[0]
                    cell = (
                        f"| {_format_mean_std(float(r['eval_return_mean']), float(r['eval_return_std']), digits=1)} / "
                        f"{_format_mean_std(float(r['eval_success_mean']), float(r['eval_success_std']), digits=3)} "
                    )
                    row.append(cell)
            report.append("".join(row) + "|")
        report.append("")

    if not certainty_summary.empty:
        report += [
            "## Episode-level certainty summary",
            "",
            "| mode | method | mean c_i | corr(c, delta) | corr(c, action prob) | corr(c, runner-up prob) |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for _, r in certainty_summary.sort_values(["mode", "method"]).iterrows():
            report.append(
                "| {mode} | {method} | {c:.3f} | {cd:.3f} | {ca:.3f} | {cr:.3f} |".format(
                    mode=r["mode"],
                    method=r["method"],
                    c=float(r["mean_episode_certainty"]),
                    cd=float(r["certainty_delta_corr_mean"]),
                    ca=float(r["certainty_action_prob_corr_mean"]),
                    cr=float(r["certainty_runner_up_prob_corr_mean"]),
                )
            )
        report.append("")

    if not traj_auc.empty or not step_auc.empty:
        auc_table = traj_auc.merge(step_auc, on=["method", "mode"], how="outer")
        report += [
            "## Certainty AUROC diagnostics",
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

    report += _lines_from_auto_analysis(agg, best_rows, best_challenge, certainty_summary)
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
    episodes, steps, updates = load_logs(args.log_dir)
    if episodes.empty:
        raise SystemExit(f"No episode logs found in {args.log_dir}")
    _, reward_auc = _compute_reward_auc_tables(episodes, window=20)
    save_return_vs_steps(episodes, reward_auc, args.plot_dir)
    save_success_vs_steps(episodes, reward_auc, args.plot_dir)
    save_metric_by_mode_subplots(episodes, reward_auc, "return", "return", "Return vs steps by mode", "06_return_by_mode_subplots.png", args.plot_dir)
    save_metric_by_mode_subplots(episodes, reward_auc, "success", "success rate", "Success rate vs steps by mode", "07_success_by_mode_subplots.png", args.plot_dir)
    save_certainty_histogram(episodes, steps, args.plot_dir)
    save_scatter(steps, "entropy", "certainty", "04_certainty_vs_entropy_scatter.png", "Certainty vs entropy scatter", args.plot_dir)
    save_scatter(steps, "delta", "certainty", "05_certainty_vs_delta_t_scatter.png", "Certainty vs delta_t scatter", args.plot_dir)
    write_report(episodes, steps, updates, args.log_dir)


if __name__ == "__main__":
    main()
