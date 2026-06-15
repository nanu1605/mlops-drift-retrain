"""Synthetic intrusion-traffic generator with *natural* drift.

Produces a single tabular stream with an integer time axis. Two regimes:

* **reference period** (earlier time): features drawn from distribution A; attacks
  are linearly separable-ish from benign traffic.
* **drift period** (later time): a subset of feature means/correlations shift, the
  attack prevalence rises, and a **new minority attack sub-population** appears whose
  signature the reference model has never seen (so it misclassifies it). This is the
  source of the accuracy dip the experiment demonstrates.

Ground-truth labels exist for every row but are intended to be revealed to the
monitoring layer with a configurable delay (handled downstream via ``time``).

The generator is fully deterministic given a seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlops_drift.config import Config
from mlops_drift.utils.seeds import rng


def _signal(x: np.ndarray, weights: np.ndarray, bias: float, noise: np.ndarray) -> np.ndarray:
    """Logit -> probability of attack for the 'known' attack mechanism."""
    z = x @ weights + bias + noise
    return 1.0 / (1.0 + np.exp(-z))


def generate(cfg: Config) -> pd.DataFrame:
    """Return a DataFrame with feature columns f0..f{n-1}, ``label``, ``t``, ``period``.

    ``t`` is a strictly increasing integer time axis: all reference rows precede all
    drift rows. ``period`` is ``"reference"`` or ``"drift"`` for bookkeeping/EDA.
    """
    d = cfg.data
    g = rng(cfg.seed)
    n_feat = d.n_features
    feat_cols = [f"f{i}" for i in range(n_feat)]

    # Stable "known attack" decision direction shared by both periods.
    weights = g.normal(0, 1.0, size=n_feat)

    # --- reference period ---
    n_ref = d.n_reference
    x_ref = g.normal(0.0, 1.0, size=(n_ref, n_feat))
    noise_ref = g.normal(0.0, 0.5, size=n_ref)
    # bias chosen so attack prevalence ~= attack_rate
    bias = _solve_bias(x_ref, weights, d.attack_rate, g)
    p_ref = _signal(x_ref, weights, bias, noise_ref)
    y_ref = (g.uniform(size=n_ref) < p_ref).astype(int)

    # --- drift period ---
    n_drift = d.n_drift
    x_drift = g.normal(0.0, 1.0, size=(n_drift, n_feat))
    # Shift mean + inflate variance on a subset of features (covariate drift).
    n_shift = max(1, int(round(d.drift_feature_frac * n_feat)))
    shift_idx = g.choice(n_feat, size=n_shift, replace=False)
    x_drift[:, shift_idx] += g.uniform(1.0, 2.0, size=n_shift)
    x_drift[:, shift_idx] *= g.uniform(1.3, 1.8, size=n_shift)

    noise_drift = g.normal(0.0, 0.5, size=n_drift)
    p_drift = _signal(x_drift, weights, bias, noise_drift)
    y_drift = (g.uniform(size=n_drift) < p_drift).astype(int)

    # New minority attack sub-population: a cluster offset in an orthogonal-ish
    # direction the reference model does not associate with attacks.
    new_dir = g.normal(0.0, 1.0, size=n_feat)
    new_dir /= np.linalg.norm(new_dir)
    frac_new = 0.18
    new_mask = g.uniform(size=n_drift) < frac_new
    x_drift[new_mask] += 3.5 * new_dir  # distinct signature
    y_drift[new_mask] = 1  # they ARE attacks (model will miss them)

    # Bump overall attack prevalence toward drift_attack_rate by flipping some
    # benign rows that sit near the boundary.
    _raise_prevalence(y_drift, p_drift, d.drift_attack_rate, g)

    ref = pd.DataFrame(x_ref, columns=feat_cols)
    ref["label"] = y_ref
    ref["period"] = "reference"

    drift = pd.DataFrame(x_drift, columns=feat_cols)
    drift["label"] = y_drift
    drift["period"] = "drift"

    out = pd.concat([ref, drift], ignore_index=True)
    out[cfg.data.time_col] = np.arange(len(out), dtype=int)
    out[cfg.data.target_col] = out["label"].astype(int)
    if cfg.data.target_col != "label":
        out = out.drop(columns=["label"])
    return out


def _solve_bias(
    x: np.ndarray, weights: np.ndarray, target_rate: float, g: np.random.Generator
) -> float:
    """Pick a bias so that mean P(attack) ≈ target_rate (simple bisection)."""
    base = x @ weights
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        rate = float(np.mean(1.0 / (1.0 + np.exp(-(base + mid)))))
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _raise_prevalence(
    y: np.ndarray, p: np.ndarray, target_rate: float, g: np.random.Generator
) -> None:
    """Flip a few highest-probability benign rows to attack to hit target prevalence."""
    cur = float(y.mean())
    if cur >= target_rate:
        return
    need = int(round((target_rate - cur) * len(y)))
    benign = np.where(y == 0)[0]
    if need <= 0 or benign.size == 0:
        return
    order = benign[np.argsort(-p[benign])][:need]
    y[order] = 1
