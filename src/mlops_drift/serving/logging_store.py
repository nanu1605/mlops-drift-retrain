"""SQLite append-only store of serving predictions.

Every ``/predict`` request appends one row per instance: timestamp, live model version,
the raw ``f*`` feature values, the predicted class, and the positive-class probability.
Phase 4 drift detection reads a rolling time window back out of this table and compares it
against the persisted reference window.

SQLite chosen for append-friendliness + queryability in a single file. WAL mode lets the
monitor read while serving writes. A process-level ``Lock`` serializes writes (single
uvicorn worker here; multi-worker would need a real DB — out of scope, see CLAUDE.md).
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from mlops_drift.utils.io import ensure_dir
from mlops_drift.utils.logging import get_logger

log = get_logger("serving.store")

TABLE = "predictions"


class RequestStore:
    """Append predictions to SQLite; read them back by time window."""

    def __init__(self, db_path: Path | str, feature_cols: list[str]):
        self.db_path = Path(db_path)
        self.feature_cols = list(feature_cols)
        self._lock = threading.Lock()
        ensure_dir(self.db_path.parent)
        # check_same_thread=False: the bg refresh / request threads share one connection,
        # guarded by self._lock.
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_table()

    def _feature_ddl(self) -> str:
        return ", ".join(f'"{c}" REAL' for c in self.feature_cols)

    def _create_table(self) -> None:
        with self._lock:
            self._conn.execute(
                f"CREATE TABLE IF NOT EXISTS {TABLE} ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "ts REAL NOT NULL, "
                "model_version TEXT NOT NULL, "
                f"{self._feature_ddl()}, "
                "pred INTEGER NOT NULL, "
                "proba REAL NOT NULL)"
            )
            self._conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_ts ON {TABLE}(ts)")
            self._conn.commit()

    def log(
        self,
        features: pd.DataFrame,
        preds: np.ndarray,
        probas: np.ndarray,
        version: str,
        ts: float,
    ) -> int:
        """Append one row per instance. Returns the number of rows written."""
        cols = ["ts", "model_version", *self.feature_cols, "pred", "proba"]
        placeholders = ", ".join("?" for _ in cols)
        quoted = ", ".join(f'"{c}"' for c in cols)
        sql = f"INSERT INTO {TABLE} ({quoted}) VALUES ({placeholders})"

        feats = features[self.feature_cols].to_numpy(dtype=float)
        rows = [
            [ts, version, *(float(v) for v in feats[i]), int(preds[i]), float(probas[i])]
            for i in range(len(feats))
        ]
        with self._lock:
            self._conn.executemany(sql, rows)
            self._conn.commit()
        return len(rows)

    def read_window(
        self, seconds: float | None = None, now: float | None = None, limit: int | None = None
    ) -> pd.DataFrame:
        """Read logged predictions. ``seconds`` keeps only rows newer than ``now-seconds``
        (Phase 4 rolling window); ``now`` defaults to the max ts in the table so callers
        need not pass a clock. ``limit`` caps the most-recent N rows."""
        clauses, params = "", []
        if seconds is not None:
            if now is None:
                cur = self._conn.execute(f"SELECT MAX(ts) FROM {TABLE}")
                now = cur.fetchone()[0]
            if now is not None:
                clauses = " WHERE ts > ?"
                params.append(now - seconds)
        sql = f"SELECT * FROM {TABLE}{clauses} ORDER BY ts ASC, id ASC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        with self._lock:
            return pd.read_sql_query(sql, self._conn, params=params)

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()
