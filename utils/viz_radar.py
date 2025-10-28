# utils/viz_radar.py
from __future__ import annotations
from math import pi
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_STYLE = dict(
    facecolor="white",
    grid_alpha=0.25,
    spine_alpha=0.6,
    label_fontsize=10,
    tick_fontsize=9,
    title_fontsize=14,
    line_width=2.0,
    fill_alpha=0.15,
)

def _angles(n: int) -> np.ndarray:
    return np.linspace(0, 2 * np.pi, n, endpoint=False)

def _normalize_values(values: List[float], ranges: Optional[List[Tuple[float, float]]]) -> List[float]:
    if not ranges:
        return [max(0.0, min(1.0, v / 100.0)) for v in values]
    out = []
    for v, (vmin, vmax) in zip(values, ranges):
        x = 0.0 if vmax == vmin else (v - vmin) / (vmax - vmin)
        out.append(float(np.clip(x, 0.0, 1.0)))
    return out

def radar_chart(
    metrics: Dict[str, float],
    title: str,
    out_path: Path,
    *,
    ranges: Optional[List[Tuple[float, float]]] = None,
    rings: int = 5,
    figsize: Tuple[float, float] = (6.5, 6.5),
    dpi: int = 300,
    style: Dict = None,
    color: Optional[str] = None,
    show: bool = False,
    fmt_ticks: str = "{:.0%}",
) -> None:
    style = {**DEFAULT_STYLE, **(style or {})}
    labels = list(metrics.keys())
    raw_values = [float(metrics[k]) for k in labels]
    values = _normalize_values(raw_values, ranges)

    if len(labels) < 3:
        raise ValueError("Radar wymaga min. 3 metryk.")

    theta = _angles(len(labels))
    theta = np.concatenate([theta, [theta[0]]])
    values_closed = np.concatenate([values, [values[0]]])

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=style["facecolor"])
    ax = plt.subplot(111, polar=True)
    ax.set_rlabel_position(0)
    ax.set_rticks(np.linspace(0.0, 1.0, rings + 1)[1:])
    ax.set_yticklabels([fmt_ticks.format(t) for t in np.linspace(1/(rings+0.0), 1.0, rings)], fontsize=style["tick_fontsize"])
    ax.yaxis.grid(True, alpha=style["grid_alpha"])
    ax.xaxis.grid(True, alpha=style["grid_alpha"])
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(labels, fontsize=style["label_fontsize"])
    for spine in ax.spines.values():
        spine.set_alpha(style["spine_alpha"])

    ax.plot(theta, values_closed, linewidth=style["line_width"], color=color)
    ax.fill(theta, values_closed, alpha=style["fill_alpha"], color=color)
    ax.set_title(title, fontsize=style["title_fontsize"], pad=18)

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".svg":
        plt.savefig(out_path, bbox_inches="tight")
    else:
        plt.savefig(out_path, bbox_inches="tight", dpi=dpi)
    if show: plt.show()
    plt.close(fig)
