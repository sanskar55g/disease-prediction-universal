"""
Tier 1 trainer: benchmark several model families per disease, keep the best.

For each disease in the registry:
  1. load the real parquet (guarded: refuses to run without provenance)
  2. stratified k-fold CV over 4 candidates
       logistic regression, random forest, XGBoost, LightGBM
  3. pick the winner by mean ROC-AUC (ties broken by Brier score = calibration)
  4. refit the winner on all rows, save models/tier1/<disease>.joblib
  5. append the full benchmark table to models/tier1/benchmark.json (committed)

Why this design: the literature survey (Liu et al. 2025; Kavakiotis et al. 2017;
Shickel et al. 2018) says boosted trees usually win on tabular clinical data but
not always, and calibration matters as much as AUC. So we measure per disease
instead of assuming.

Run:  python backend/scripts/train_tier1.py            # all
      python backend/scripts/train_tier1.py heart_disease
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data_guard import MODELS, require, tier_dir  # noqa: E402
from app.tiers.tier1_registry import TIER1_DISEASES  # noqa: E402

RAW = tier_dir(1, "raw")
OUT = MODELS / "tier1"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42


def load(key: str) -> tuple[pd.DataFrame, pd.Series]:
    path = require(RAW / f"{key}.parquet", f"python backend/scripts/download_tier1.py {key}")
    df = pd.read_parquet(path)
    y = df.pop("__target__").astype(int)
    # drop id-like columns
    for c in list(df.columns):
        if c.lower() in {"id", "name", "patient_id"}:
            df = df.drop(columns=c)
    return df, y


def preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])


def candidates(n_rows: int, pos_rate: float) -> dict:
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    spw = (1 - pos_rate) / max(pos_rate, 1e-6)  # class imbalance weight
    small = n_rows < 1000
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5),
        "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2 if small else 5,
                                                class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
        "xgboost": XGBClassifier(n_estimators=300 if small else 600, max_depth=3 if small else 6,
                                 learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                 scale_pos_weight=spw, eval_metric="logloss", n_jobs=-1, random_state=SEED),
        "lightgbm": LGBMClassifier(n_estimators=300 if small else 600, num_leaves=7 if small else 31,
                                   learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                   min_child_samples=5 if small else 20, scale_pos_weight=spw,
                                   n_jobs=-1, random_state=SEED, verbose=-1),
    }


def train_one(key: str) -> dict:
    X, y = load(key)
    n, pos = len(X), float(y.mean())
    folds = 5 if n < 50_000 else 3
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    results = {}
    for name, est in candidates(n, pos).items():
        pipe = Pipeline([("prep", preprocessor(X)), ("clf", est)])
        t0 = time.time()
        scores = cross_validate(pipe, X, y, cv=cv, n_jobs=1,
                                scoring={"auc": "roc_auc", "brier": "neg_brier_score"})
        results[name] = {
            "auc_mean": round(float(scores["test_auc"].mean()), 4),
            "auc_std": round(float(scores["test_auc"].std()), 4),
            "brier": round(float(-scores["test_brier"].mean()), 4),
            "fit_seconds": round(time.time() - t0, 1),
        }
        print(f"    {name:<20} AUC {results[name]['auc_mean']:.3f} ± {results[name]['auc_std']:.3f}   Brier {results[name]['brier']:.3f}")
    best = sorted(results, key=lambda k: (-results[k]["auc_mean"], results[k]["brier"]))[0]

    final = Pipeline([("prep", preprocessor(X)), ("clf", candidates(n, pos)[best])]).fit(X, y)
    joblib.dump({"pipeline": final, "features": X.columns.tolist(),
                 "dtypes": {c: str(t) for c, t in X.dtypes.items()}}, OUT / f"{key}.joblib")
    return {"disease": key, "label": TIER1_DISEASES[key]["label"], "rows": n, "positive_rate": round(pos, 4),
            "cv_folds": folds, "candidates": results, "selected_model": best,
            "selected_auc": results[best]["auc_mean"], "features": X.columns.tolist()}


if __name__ == "__main__":
    keys = sys.argv[1:] or list(TIER1_DISEASES)
    bench_file = OUT / "benchmark.json"
    bench = json.loads(bench_file.read_text()) if bench_file.exists() else {}
    for k in keys:
        print(f"\n[{k}] {TIER1_DISEASES[k]['label']}")
        bench[k] = train_one(k)
        print(f"  -> selected {bench[k]['selected_model']} (AUC {bench[k]['selected_auc']:.3f})")
        bench_file.write_text(json.dumps(bench, indent=2))
    print("\nSummary")
    for k in keys:
        b = bench[k]
        print(f"  {k:<26} {b['selected_model']:<20} AUC {b['selected_auc']:.3f}  (n={b['rows']})")
