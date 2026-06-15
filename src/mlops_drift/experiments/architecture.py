"""Render the architecture diagram (docs/images/architecture.png).

A simple boxes-and-arrows view of the closed loop, drawn with matplotlib (Agg) so it
regenerates deterministically with no extra tooling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from mlops_drift.config import REPO_ROOT
from mlops_drift.utils.io import ensure_dir

DEFAULT_OUT = REPO_ROOT / "docs" / "images" / "architecture.png"

# (key): (x, y, label)
BOXES = {
    "data": (0.5, 3.0, "Data\n(synthetic / CICIDS2017)\ningest · validate · split"),
    "train": (3.0, 3.0, "Training\nPipeline(impute→RF)\nevaluate · MLflow log"),
    "registry": (5.5, 3.0, "Model Registry\n@champion / @challenger\n(aliases)"),
    "serving": (5.5, 1.0, "Serving (FastAPI)\n/predict /health\n/metrics /reload"),
    "reqlog": (3.0, 1.0, "Request log\n(SQLite, raw f*)"),
    "monitor": (3.0, -1.0, "Monitoring\nEvidently drift\nrealized F1 (delayed labels)"),
    "controller": (0.5, 1.0, "Controller\ndrift→retrain→validate\n→promote→reload"),
}

# (src, dst, label, color, rad, label_pos)  — rad curves the arrow; label_pos is the fraction
# along the line for the text (keeps labels off the boxes).
ARROWS = [
    ("data", "train", "", "0.3", 0.0, 0.5),
    ("train", "registry", "register", "0.3", 0.0, 0.5),
    ("registry", "serving", "load @champion", "0.3", 0.0, 0.5),
    ("serving", "reqlog", "log preds", "0.3", 0.0, 0.5),
    ("reqlog", "monitor", "rolling window", "0.3", 0.0, 0.5),
    ("monitor", "controller", "drift signal", "#b00020", 0.0, 0.45),
    ("controller", "train", "retrain", "#0050b0", 0.0, 0.5),
    ("controller", "registry", "promote", "#0050b0", -0.25, 0.5),
    ("controller", "serving", "POST /reload", "#0050b0", -0.55, 0.5),
]

_BW, _BH = 1.9, 0.9


def _center(key):
    x, y, _ = BOXES[key]
    return x + _BW / 2, y + _BH / 2


def render(out: Path | str = DEFAULT_OUT) -> Path:
    fig, ax = plt.subplots(figsize=(11, 7))
    for _key, (x, y, label) in BOXES.items():
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y),
                _BW,
                _BH,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                linewidth=1.5,
                edgecolor="#333",
                facecolor="#eef3fb",
            )
        )
        ax.text(x + _BW / 2, y + _BH / 2, label, ha="center", va="center", fontsize=8.5)

    for src, dst, label, color, rad, lpos in ARROWS:
        x0, y0 = _center(src)
        x1, y1 = _center(dst)
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": 1.6,
                "shrinkA": 26,
                "shrinkB": 26,
                "connectionstyle": f"arc3,rad={rad}",
            },
        )
        if label:
            lx = x0 + (x1 - x0) * lpos
            ly = y0 + (y1 - y0) * lpos - (rad * 2.2)  # nudge off the curved line
            ax.text(
                lx,
                ly,
                label,
                fontsize=7.5,
                color=color,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none"},
            )

    ax.set_xlim(0, 8)
    ax.set_ylim(-1.6, 4.4)
    ax.axis("off")
    ax.set_title("Drift-triggered retraining loop — no human in path", fontsize=12, weight="bold")
    fig.tight_layout()
    out = Path(out)
    ensure_dir(out.parent)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    p = render()
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
