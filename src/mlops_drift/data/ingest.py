"""Dataset acquisition entrypoint (`make data`).

Uses real CICIDS2017 MachineLearningCSV files if present in ``data/raw/``; otherwise
falls back to the synthetic generator so the whole pipeline stays runnable end to end.
Writes a single canonical parquet to ``data/raw/dataset.parquet`` and prints which path
was taken (the caller records it in CHANGELOG.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from mlops_drift.config import Config, get_config
from mlops_drift.data import synthetic
from mlops_drift.utils.io import ensure_dir, write_parquet
from mlops_drift.utils.logging import get_logger
from mlops_drift.utils.seeds import set_seed

log = get_logger("ingest")

CANONICAL = "dataset.parquet"


def _find_cicids_csvs(raw_dir: Path) -> list[Path]:
    """Real dataset detection: any *.csv in raw/ that is not our own output."""
    return sorted(p for p in raw_dir.glob("*.csv") if p.is_file())


def _load_cicids(csvs: list[Path], cfg: Config) -> pd.DataFrame:
    """Collapse CICIDS2017 CSVs to the binary schema, day/session order = time axis.

    NOTE: kept intentionally minimal — real-data path is a documented option; the
    synthetic path is the runnable default in this environment.
    """
    frames = []
    for i, path in enumerate(csvs):
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        label_col = next((c for c in df.columns if c.lower() == "label"), None)
        if label_col is None:
            raise ValueError(f"{path} has no Label column")
        df["_session"] = i  # capture order as the time axis
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values("_session").reset_index(drop=True)
    y = (full[label_col].astype(str).str.upper() != "BENIGN").astype(int)
    feats = full.select_dtypes("number").drop(columns=["_session"], errors="ignore")
    feats = feats.replace([float("inf"), float("-inf")], pd.NA).dropna(axis=1, how="any")
    out = feats.copy()
    out[cfg.data.target_col] = y.values
    out[cfg.data.time_col] = range(len(out))
    out["period"] = "reference"  # refined downstream by features/time split
    return out


def ingest(cfg: Config | None = None) -> tuple[Path, str]:
    """Produce ``data/raw/dataset.parquet``. Returns (path, source) where source is
    ``"cicids2017"`` or ``"synthetic"``."""
    cfg = cfg or get_config()
    set_seed(cfg.seed)
    raw_dir = ensure_dir(cfg.raw_dir)
    csvs = _find_cicids_csvs(raw_dir)

    if csvs:
        log.info("ingest.source", source="cicids2017", n_files=len(csvs))
        df = _load_cicids(csvs, cfg)
        source = "cicids2017"
    else:
        log.info("ingest.source", source="synthetic", reason="no CSVs in data/raw")
        df = synthetic.generate(cfg)
        source = "synthetic"

    out_path = raw_dir / CANONICAL
    write_parquet(df, out_path)
    log.info("ingest.written", path=str(out_path), rows=len(df), cols=df.shape[1])
    return out_path, source


def main() -> int:
    path, source = ingest()
    print(f"data ingested via '{source}' -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
