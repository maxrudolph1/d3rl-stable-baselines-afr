"""Load environment.csv with config.yaml metadata for each row."""

import pandas as pd
import yaml
from pathlib import Path


def flatten_config(config: dict, parent_key: str = "", sep: str = "_") -> dict:
    """Flatten nested config dict into a single level with concatenated keys."""
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_run_with_config(run_path: str | Path) -> pd.DataFrame:
    """
    Load environment.csv from a run folder and add all config.yaml values to each row.

    Path structure: artifacts/offline_rl/<group_id>/<game_name>/<run_name>/
    The path can point to the run folder or directly to environment.csv.

    Returns:
        DataFrame with environment.csv columns (epoch, step, value) plus all config values.
    """
    path = Path(run_path)
    if path.suffix == ".csv":
        run_dir = path.parent
        csv_path = path
    else:
        run_dir = path
        csv_path = run_dir / "environment.csv"

    config_path = run_dir / "config.yaml"
    if not csv_path.exists():
        raise FileNotFoundError(f"environment.csv not found: {csv_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found: {config_path}")

    # Load environment.csv (epoch, step, value) with no header
    df = pd.read_csv(csv_path, header=None, names=["epoch", "step", "value"])

    # Load and flatten config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    flat_config = flatten_config(config)

    # Add config values to every row
    for key, val in flat_config.items():
        df[key] = val

    return df


def load_game_runs(game_path: str | Path) -> pd.DataFrame:
    """
    Find all run folders under a game directory and load them into one DataFrame.

    Path structure: artifacts/offline_rl/<group_id>/<game_name>/
    Each subdirectory of game_path is a run (e.g. 20260207_205205_RNvn) with
    environment.csv and config.yaml.

    Returns:
        DataFrame with all runs concatenated. Each row has epoch, step, value
        plus config columns (including unique_tag to identify the run).
    """
    game_dir = Path(game_path)
    if not game_dir.is_dir():
        raise FileNotFoundError(f"Game directory not found: {game_dir}")

    dfs = []
    for run_dir in sorted(game_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        csv_path = run_dir / "environment.csv"
        config_path = run_dir / "config.yaml"
        if csv_path.exists() and config_path.exists():
            dfs.append(load_run_with_config(run_dir))
        else:
            continue  # Skip dirs that don't look like runs

    if not dfs:
        raise FileNotFoundError(
            f"No valid runs found under {game_dir} "
            "(need environment.csv and config.yaml in each subdirectory)"
        )
    return pd.concat(dfs, ignore_index=True)


def plot_step_vs_value(
    df: pd.DataFrame,
    group_by: str | list[str] | None = None,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot step vs value: mean with ± std error shaded across runs.

    Args:
        df: DataFrame from load_game_runs (has step, value, and config columns).
        group_by: Config column(s) to group runs by. Each group gets its own line with
            mean ± sem. E.g. "args_freeze_encoder" or ["args_freeze_encoder", "args_encoder_weights"].
            If None, all runs are aggregated into one line.
        save_path: Where to save the figure.
    """
    import matplotlib.pyplot as plt

    if group_by is not None:
        group_cols = [group_by] if isinstance(group_by, str) else list(group_by)
        missing = [c for c in group_cols if c not in df.columns]
        if missing:
            raise ValueError(f"group_by columns not in DataFrame: {missing}. Available: {list(df.columns)}")
    else:
        group_cols = []

    fig, ax = plt.subplots()
    # Only do grouped lines if group_cols is not empty and not None
    if group_cols and len(group_cols) > 0:
        # within this, only group if there's more than one group value or more than 1 group col
        # e.g. don't do grouped legend/labels for a single unique value or single group
        # (preserves original intent)
        if (len(group_cols) == 1 and df[group_cols[0]].nunique() > 1) or (len(group_cols) > 1):
            for keys, g in df.groupby(group_cols):
                keys_tuple = (keys,) if len(group_cols) == 1 else keys
                label = ", ".join(f"{c}={v}" for c, v in zip(group_cols, keys_tuple))
                agg = g.groupby("step")["value"].agg(["mean", "sem"]).reset_index()
                step, mean, sem = agg["step"], agg["mean"], agg["sem"]
                ax.plot(step, mean, label=label)
                ax.fill_between(step, mean - sem, mean + sem, alpha=0.3)
            ax.legend()
        else:
            # Single group, no need for legend
            agg = df.groupby("step")["value"].agg(["mean", "sem"]).reset_index()
            step, mean, sem = agg["step"], agg["mean"], agg["sem"]
            ax.plot(step, mean)
            ax.fill_between(step, mean - sem, mean + sem, alpha=0.3)
    else:
        # No group_by: aggregate all runs
        agg = df.groupby("step")["value"].agg(["mean", "sem"]).reset_index()
        step, mean, sem = agg["step"], agg["mean"], agg["sem"]
        ax.plot(step, mean)
        ax.fill_between(step, mean - sem, mean + sem, alpha=0.3)

    ax.set_xlabel("step")
    ax.set_ylabel("value")
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot step vs value from environment.csv with optional grouping by config params."
    )
    parser.add_argument(
        "--path",
        default='artifacts/offline_rl/SEAQUEST_BIG_EXPERT_02_14_2026/SeaquestNoFrameskip-v4',
        help="Path to game folder, e.g. artifacts/offline_rl/SEAQUEST_POST_EXPERT_02_07_2026/SeaquestNoFrameskip-v4",
    )
    parser.add_argument(
        "--group-by",
        nargs="*",
        default=None,
        metavar="COL",
        help="Config columns to group runs by (e.g. args_freeze_encoder args_encoder_weights). "
        "Pass multiple for composite grouping. Omit to aggregate all runs.",
    )
    parser.add_argument(
        "--save",
        default=None,
        metavar="FILE",
        help="Output path for the figure. Default: figures/{group_id}_{game_name}.png",
    )
    args = parser.parse_args()

    df = load_game_runs(args.path)
    group_by = args.group_by if args.group_by else None
    save_path = args.save
    if save_path is None:
        path_parts = Path(args.path).parts
        group_id = path_parts[-2] if len(path_parts) >= 2 else "output"
        game_name = path_parts[-1] if path_parts else "plot"
        save_path = f"figures/{group_id}_{game_name}.png"
    plot_step_vs_value(df, group_by=group_by, save_path=save_path)