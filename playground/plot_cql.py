"""
Load CQL (or other offline RL) training runs from an artifacts directory and
build a pandas DataFrame for plotting, with optional averaging across seeds.

Directory layout:
  <root_dir>/
    <game>/           # e.g. SeaquestNoFrameskip-v4
      <run_name>/     # e.g. 20260207_205205_RNvn
        environment.csv
        config.yaml
"""

import os
from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str) -> dict:
    """Load a single config.yaml into a flat dict (for DataFrame columns)."""
    with open(config_path) as f:
        data = yaml.safe_load(f)
    # Flatten: put args at top level with prefix if needed to avoid clashes
    flat = {}
    if "unique_tag" in data:
        flat["unique_tag"] = data["unique_tag"]
    if "args" in data:
        for k, v in data["args"].items():
            flat[f"args_{k}"] = v
    return flat


def load_environment_csv(csv_path: str) -> pd.DataFrame:
    """Load environment.csv (epoch, step, value) with no header."""
    return pd.read_csv(
        csv_path,
        header=None,
        names=["epoch", "step", "value"],
    )


def load_runs_from_directory(root_dir: str) -> pd.DataFrame:
    """
    Scan root_dir for <game>/<run_name>/ and load environment.csv + config.yaml
    for each run. Return one DataFrame with all runs (long format).

    Columns include: game, run_name, seed (from config), step, value, and
    other config fields (args_*) so you can group by experiment and average
    over seeds.
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    rows = []

    for game_dir in sorted(root.iterdir()):
        if not game_dir.is_dir():
            continue
        game = game_dir.name

        for run_dir in sorted(game_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run_name = run_dir.name

            env_csv = run_dir / "environment.csv"
            config_yaml = run_dir / "config.yaml"

            if not env_csv.exists():
                continue
            if not config_yaml.exists():
                config_flat = {}
            else:
                config_flat = load_config(str(config_yaml))

            df_env = load_environment_csv(str(env_csv))
            seed = config_flat.get("args_seed", None)

            df_env["game"] = game
            df_env["run_name"] = run_name
            df_env["seed"] = seed
            for k, v in config_flat.items():
                df_env[k] = v

            rows.append(df_env)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def aggregate_by_experiment(
    df: pd.DataFrame,
    experiment_cols: list[str] | None = None,
    value_col: str = "value",
) -> pd.DataFrame:
    """
    Average metrics across seeds within each experiment.

    experiment_cols: columns that define an experiment (e.g. game + args_encoder_weights).
                     If None, uses ['game', 'run_name'] (no averaging).
    """
    if experiment_cols is None:
        experiment_cols = ["game", "run_name"]
    # Ensure we have step (or epoch) for x-axis
    group_cols = [c for c in experiment_cols if c in df.columns]
    if "step" in df.columns:
        group_cols = group_cols + ["step"]
    else:
        group_cols = group_cols + ["epoch"]

    agg = df.groupby(group_cols, dropna=False)[value_col].agg(["mean", "std", "count"])
    agg = agg.reset_index()
    agg = agg.rename(columns={"mean": value_col, "std": f"{value_col}_std"})
    return agg


def all_environment_csv_last_n_mean(
    root_dir: str,
    last_n: int = 20,
    col_index: int = 2,
) -> list[tuple[str, float]]:
    """
    Find all environment.csv files under root_dir (<game>/<run_name>/environment.csv).
    For each, compute the mean of the (col_index+1)-th column over the last last_n rows.
    Returns list of (csv_path, mean_value).
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    results = []
    for game_dir in sorted(root.iterdir()):
        if not game_dir.is_dir():
            continue
        for run_dir in sorted(game_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            csv_path = run_dir / "environment.csv"
            if not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path, header=None)
                if len(df) == 0 or df.shape[1] <= col_index:
                    continue
                tail = df.iloc[-last_n:, col_index]
                mean_val = tail.astype(float).mean()
                results.append((str(csv_path), mean_val))
            except Exception:
                results.append((str(csv_path), float("nan")))
    return results


def last_n_run_average(
    df: pd.DataFrame,
    n: int = 10,
    value_col: str = "value",
    run_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    For each run, take the last n data points (by step) and average their value.
    Returns one row per run with column last_n_avg (and run identifiers).
    """
    if run_cols is None:
        run_cols = ["game", "run_name"]
    run_cols = [c for c in run_cols if c in df.columns]

    def tail_mean(g: pd.DataFrame) -> float:
        tail = g.nlargest(n, "step")[value_col]
        return tail.mean()

    out = (
        df.groupby(run_cols, dropna=False)
        .apply(tail_mean)
        .reset_index(name="last_n_avg")
    )
    # Restore any other columns that are constant per run (e.g. args_*)
    first = df.groupby(run_cols).first().reset_index()
    merge_cols = [c for c in first.columns if c not in out.columns or c in run_cols]
    out = out.merge(first[merge_cols], on=run_cols, how="left")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load CQL runs into a DataFrame")
    parser.add_argument(
        "root_dir",
        type=str,
        default="/u/mrudolph/documents/d3rlpy/artifacts/offline_rl/BREAKOUT_POST_EXPERT_02_04_2026",
        nargs="?",
        help="Root directory containing <game>/<run_name>/ with environment.csv and config.yaml",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to save the raw DataFrame (CSV).",
    )
    parser.add_argument(
        "--experiment-cols",
        type=str,
        nargs="+",
        default=None,
        help="Columns to define an experiment for averaging across seeds (e.g. game args_encoder_weights).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot value vs step, averaged by experiment (requires --experiment-cols).",
    )
    parser.add_argument(
        "--plot-by",
        type=str,
        default=None,
        help="Column to use for separate lines/curves (e.g. args_encoder_weights). Used with --plot.",
    )
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="Column to group by; for each run take running avg of last N points, then print mean ± std per key.",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=10,
        help="Number of last data points to average per run (default: 10).",
    )
    parser.add_argument(
        "--print-csv-last-n",
        type=int,
        default=0,
        metavar="N",
        help="If set, load each environment.csv under root_dir, compute mean of 3rd column over last N rows, and print (default: 0 = off).",
    )
    args = parser.parse_args()

    if args.print_csv_last_n > 0:
        results = all_environment_csv_last_n_mean(args.root_dir, last_n=args.print_csv_last_n, col_index=2)
        print(f"environment.csv: mean of 3rd column over last {args.print_csv_last_n} rows:\n")
        for path, val in results:
            print(f"  {path}: {val}")
        exit(0)

    df = load_runs_from_directory(args.root_dir)
    if df.empty:
        print("No runs found.")
        exit(1)

    print("Raw runs DataFrame:")
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"Games: {df['game'].unique().tolist()}")
    print(f"Runs: {df['run_name'].nunique()}, seeds: {df['seed'].unique().tolist()}")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nSaved raw DataFrame to {args.out}")

    if args.key:
        run_avg = last_n_run_average(df, n=args.last_n, run_cols=["game", "run_name"])
        if args.key not in run_avg.columns:
            print(f"Key '{args.key}' not in DataFrame. Available: {[c for c in run_avg.columns]}")
        else:
            by_key = run_avg.groupby(args.key)["last_n_avg"].agg(["mean", "std", "count"])
            by_key = by_key.reset_index()
            print(f"\nLast {args.last_n} points average per run, then mean ± std over key '{args.key}':")
            for _, row in by_key.iterrows():
                k, m, s, c = row[args.key], row["mean"], row["std"], row["count"]
                std_str = f" ± {s:.2f}" if pd.notna(s) and s > 0 else ""
                print(f"  {k}: {m:.2f}{std_str}  (n={int(c)} runs)")

    if args.experiment_cols:
        df_agg = aggregate_by_experiment(df, experiment_cols=args.experiment_cols)
        print("\nAggregated (mean ± std across seeds):")
        print(df_agg.head(10))
        if args.out:
            out_agg = args.out.replace(".csv", "_agg.csv") if args.out.endswith(".csv") else args.out + "_agg.csv"
            df_agg.to_csv(out_agg, index=False)
            print(f"Saved aggregated DataFrame to {out_agg}")

        if args.plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            plot_col = args.plot_by
            if plot_col and plot_col not in df_agg.columns:
                plot_col = None
            if plot_col and plot_col in df_agg.columns:
                for key, g in df_agg.groupby(plot_col):
                    ax.plot(g["step"], g["value"], label=str(key))
                    if "value_std" in g.columns:
                        ax.fill_between(
                            g["step"],
                            g["value"] - g["value_std"],
                            g["value"] + g["value_std"],
                            alpha=0.2,
                        )
                ax.legend()
            else:
                for (game,), g in df_agg.groupby(["game"]):
                    ax.plot(g["step"], g["value"], label=game)
                    if "value_std" in g.columns:
                        ax.fill_between(
                            g["step"],
                            g["value"] - g["value_std"],
                            g["value"] + g["value_std"],
                            alpha=0.2,
                        )
                ax.legend()
            ax.set_xlabel("step")
            ax.set_ylabel("environment (eval return)")
            ax.set_title("Runs averaged across seeds")
            plt.tight_layout()
            plt.show()
