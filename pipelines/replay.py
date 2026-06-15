"""Spec-path entry for the replay streamer (``make replay``).

The implementation lives in ``mlops_drift.experiments.replay`` (importable + tested); this
thin wrapper keeps the spec's ``pipelines/replay.py`` path.
"""

from __future__ import annotations

import sys

from mlops_drift.experiments.replay import main

if __name__ == "__main__":
    sys.exit(main())
