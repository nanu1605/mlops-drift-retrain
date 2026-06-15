"""Phase 6: plot builders write PNGs (pure matplotlib; no MLflow / no server)."""

from __future__ import annotations

from mlops_drift.experiments import architecture
from mlops_drift.experiments.run import _plot_recovery


def test_plot_recovery_writes_png(tmp_path):
    timeline = [
        {"t": 12000 + i * 400, "f1": (0.28 if i < 4 else 0.83), "version": ("1" if i < 4 else "2")}
        for i in range(12)
    ]
    out = _plot_recovery(timeline, detected_t=13600, promoted_t=13600, out=tmp_path / "rec.png")
    assert out.exists() and out.stat().st_size > 0


def test_architecture_renders_png(tmp_path):
    out = architecture.render(out=tmp_path / "arch.png")
    assert out.exists() and out.stat().st_size > 0
