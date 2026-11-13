"""Deterministic chart rendering with matplotlib.

The writer agent supplies only structured numbers (ChartSpec). This
module validates them and draws the chart — so a chart can never
disagree with its own labels, and a bad spec fails safely (no chart)
instead of producing a broken image.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def validate_chart(chart) -> list[str]:
    problems: list[str] = []
    if chart is None:
        return ["no chart"]
    if chart.chart_type not in ("bar", "line", "pie"):
        problems.append(f"Unknown chart_type {chart.chart_type!r}.")
    if not (2 <= len(chart.labels) <= 8):
        problems.append(f"Need 2-8 labels, got {len(chart.labels)}.")
    if len(chart.labels) != len(chart.values):
        problems.append("labels and values lengths differ.")
    if any(not math.isfinite(v) for v in chart.values):
        problems.append("Non-finite value in chart.")
    if chart.chart_type == "pie" and any(v < 0 for v in chart.values):
        problems.append("Pie chart values must be >= 0.")
    return problems


def render_chart(chart, out_path: Path) -> Path | None:
    """Render one chart to PNG. Returns the path, or None if invalid."""
    if validate_chart(chart):
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = [str(lbl)[:24] for lbl in chart.labels]
    values = list(chart.values)
    if chart.chart_type == "bar":
        ax.bar(labels, values, color=COLORS[: len(labels)])
        ax.set_ylabel(chart.y_label or "Value")
        ax.set_xlabel(chart.x_label or "")
        ax.tick_params(axis="x", rotation=20)
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:g}", ha="center", va="bottom", fontsize=9)
    elif chart.chart_type == "line":
        ax.plot(labels, values, marker="o", color=COLORS[0], linewidth=2)
        ax.set_ylabel(chart.y_label or "Value")
        ax.set_xlabel(chart.x_label or "")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.3)
    else:  # pie
        ax.pie(values, labels=labels, autopct="%1.0f%%", colors=COLORS[: len(labels)], startangle=90)
        ax.axis("equal")
    ax.set_title(chart.title or "Chart", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
