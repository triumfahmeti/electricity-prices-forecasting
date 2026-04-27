from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts" / "forecast_plots"


def ensure_output_dir(output_dir: str | Path | None = None) -> Path:
    resolved = DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def to_prediction_frame(y_true: pd.Series, y_pred, split: str) -> pd.DataFrame:
    pred_series = y_pred if isinstance(y_pred, pd.Series) else pd.Series(y_pred, index=y_true.index)
    frame = pd.DataFrame({"true": y_true, "pred": pred_series}, index=y_true.index)
    frame["split"] = split
    return frame


def _format_axis(ax) -> None:
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(True, alpha=0.25)


def save_split_plot(frame: pd.DataFrame, output_path: str | Path, title: str) -> Path:
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(frame.index, frame["true"], label="True", color="#222222", linewidth=1.4)
    ax.plot(frame.index, frame["pred"], label="Prediction", color="#1f77b4", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price")
    _format_axis(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_combined_plot(frames: dict[str, pd.DataFrame], output_path: str | Path, title: str) -> Path:
    output_path = Path(output_path)
    split_colors = {"train": "#d9ead3", "validation": "#fff2cc", "test": "#f4cccc"}
    fig, ax = plt.subplots(figsize=(16, 6))

    first = True
    for split_name in ["train", "validation", "test"]:
        if split_name not in frames:
            continue
        frame = frames[split_name]
        ax.axvspan(frame.index.min(), frame.index.max(), color=split_colors[split_name], alpha=0.4)
        ax.plot(frame.index, frame["true"], color="#222222", linewidth=1.2, label="True" if first else None)
        ax.plot(frame.index, frame["pred"], color="#1f77b4", linewidth=1.1, label="Prediction" if first else None)
        first = False

    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price")
    _format_axis(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_prediction_bundle(
    frames: dict[str, pd.DataFrame],
    model_name: str,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    output_dir = ensure_output_dir(output_dir)
    safe_name = model_name.lower().replace(" ", "_")
    paths: dict[str, str] = {}

    combined_path = output_dir / f"{safe_name}_all_splits.png"
    save_combined_plot(frames, combined_path, f"{model_name} Forecast | Train / Validation / Test")
    paths["all_splits"] = str(combined_path)

    for split_name, frame in frames.items():
        split_path = output_dir / f"{safe_name}_{split_name}.png"
        save_split_plot(frame, split_path, f"{model_name} Forecast | {split_name.title()}")
        paths[split_name] = str(split_path)

    return paths


def save_test_comparison_plot(
    test_frames: dict[str, pd.DataFrame],
    output_path: str | Path,
    title: str = "Test Set Forecast Comparison",
) -> Path:
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(16, 6))

    first_frame = next(iter(test_frames.values()))
    ax.plot(first_frame.index, first_frame["true"], label="True", color="#222222", linewidth=1.6)

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    mae_lines = []
    for idx, (model_name, frame) in enumerate(test_frames.items()):
        color = palette[idx % len(palette)]
        mae = mean_absolute_error(frame["true"], frame["pred"])
        mae_lines.append(f"{model_name}: MAE={mae:.3f}")
        ax.plot(frame.index, frame["pred"], label=model_name, color=color, linewidth=1.15, alpha=0.95)

    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price")
    _format_axis(ax)
    ax.legend(loc="upper left", ncol=2)
    ax.text(
        0.01,
        0.82,
        "\n".join(mae_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
