# Drift Experiment — detect, retrain, recover

Reproduce with `make experiment` → regenerates `docs/images/drift_recovery.png`.

## Hypothesis
A network-intrusion classifier trained on a *reference* traffic period degrades when the
production distribution shifts (covariate shift + a **new attack sub-population** the model has
never seen). A closed loop that (a) detects the shift label-free, (b) retrains on the newly
labelled data, and (c) promotes the challenger only if it beats the champion, will **recover
realized F1 with no human in the path**.

## Setup
- **Data** (`src/mlops_drift/data/synthetic.py`): 20k rows over an integer time axis `t`.
  - *Reference* period (12k rows, attack rate ≈ 18 %): features ~ N(0, 1).
  - *Drift* period (8k rows, attack rate ≈ 49 %): 40 % of features shift mean + inflate variance,
    **plus a new minority attack sub-population** offset along an orthogonal direction that the
    reference model misclassifies. Labels arrive `label_delay_steps = 200` after the prediction.
- **Model**: one family — `Pipeline(SimpleImputer(median) → RandomForestClassifier)` on raw
  `f0…f11`. Registered to MLflow with `@champion` / `@challenger` **aliases**.
- **Method** (`src/mlops_drift/experiments/run.py`): boot the real serving process on a fresh
  reference-only champion, replay the drift period to `POST /predict` in time-ordered batches,
  and run the **real controller** between batches (drift signal → retrain on `reference+drift`
  → champion/challenger gate → `POST /reload`). Each batch's realized F1 (live champion's
  predictions vs that batch's labels) is the timeline.

## Result

![drift recovery](../docs/images/drift_recovery.png)

| phase | realized F1 (per-batch mean) |
|---|---|
| stale champion on drift | **≈ 0.27** |
| after auto-promotion | **≈ 0.78** |

The curve is flat-low while the stale reference champion serves drifted traffic, then jumps at
the **drift-detected → model-promoted** markers and stays recovered — the whole transition
triggered automatically by the controller.

## Thresholds (`configs/thresholds.yaml`)
- `drift.dataset_drift_share = 0.5` — declare drift when ≥ 50 % of features are flagged by
  Evidently. The synthetic shift trips **every** feature (share = 1.0), so detection is decisive;
  a subtler real-world shift would need this tuned down.
- `promotion.f1_margin = 0.01` — the challenger must beat the champion by ≥ 0.01 F1 on the drift
  holdout. Prevents promotion on noise/ties.
- `validation.f1_floor = 0.50` — a model below this in-distribution F1 is blocked (CI gate). The
  deliberately-simple RF scores ≈ 0.576 in-distribution, so 0.50 is a meaningful floor (0.70
  would be unreachable for this model — a reminder to set floors from measured baselines).

## Failure modes & honest limitations
- **Label latency.** The drift *trigger* is label-free (Evidently on features) — correct, because
  realized F1 needs labels that arrive late. The recovery *plot* reveals labels immediately for
  visualization; in production the realized-F1 line would lag by `label_delay_steps`.
- **Optimistic promotion holdout.** The champion/challenger comparison scores on the drift period,
  which overlaps the challenger's retrain data → inflated challenger F1. A stricter setup would
  carve a disjoint future holdout. The honest out-of-sample view is the per-batch realized-F1
  series above (the challenger is scored on *unseen* batches as they stream).
- **Synthetic ≠ real.** No real CICIDS2017 here; the shift is engineered to be detectable. The
  ingest path auto-uses real CSVs if dropped in `data/raw/`.
- **One model family, no HP search.** By design (scope guardrails) — the deliverable is the
  *loop*, not model accuracy.
- **Single-writer assumptions.** SQLite request log + in-process model swap assume one serving
  worker; multi-worker would need a real DB + shared cache.

## Reproducibility
Deterministic: fixed seed, time-aware splits, models logged with the DVC data hash + git SHA.
`make experiment` resets to a fresh baseline champion and clears runtime state, so the recovery
curve is identical run-to-run (only wall-clock timing varies).
