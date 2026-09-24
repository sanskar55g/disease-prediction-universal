"""
Tier 2 trainer: multi-class symptom -> acute condition classifier on DDXPlus.

Feature encoding: each DDXPlus "evidence" is a token. Binary evidences become
one token (E_55). Categorical/multi-valued evidences become token+value
(E_55_@_V_12). Plus age and sex. That gives a sparse bag-of-evidence matrix,
which is what the DDXPlus authors and follow-up papers use.

Benchmarks logistic regression vs LightGBM (multi-class) on a held-out split,
picks by macro-F1 (49 classes are imbalanced), saves the winner.

If only the --small download exists, training uses the validate split and
evaluates on the test split. Both are real DDXPlus releases. With the full
download it trains on train and evaluates on test.

Run:  python backend/scripts/train_tier2.py
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
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data_guard import MODELS, require, tier_dir  # noqa: E402

RAW = tier_dir(2, "raw")
OUT = MODELS / "tier2"
OUT.mkdir(parents=True, exist_ok=True)
HINT = "python backend/scripts/download_tier2.py --small"
SEED = 42
MAX_TRAIN = 300_000  # cap for laptop-friendly training; raise on a bigger machine


def evidence_tokens(ev: str) -> str:
    """'[\"E_55\", \"E_91_@_V_12\"]' -> 'E_55 E_91_@_V_12' (tokens kept whole)."""
    items = json.loads(ev.replace("'", '"'))
    return " ".join(items)


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_parquet(require(RAW / f"release_{name}_patients.parquet", HINT))
    df["tokens"] = df["EVIDENCES"].map(evidence_tokens)
    return df


def build_X(vec: CountVectorizer, df: pd.DataFrame, fit: bool):
    bag = vec.fit_transform(df["tokens"]) if fit else vec.transform(df["tokens"])
    age = sparse.csr_matrix((df["AGE"].to_numpy() / 100.0).reshape(-1, 1))
    sex = sparse.csr_matrix((df["SEX"].eq("M").astype(float)).to_numpy().reshape(-1, 1))
    return sparse.hstack([bag, age, sex]).tocsr()


if __name__ == "__main__":
    train_name = "train" if (RAW / "release_train_patients.parquet").exists() else "validate"
    print(f"Tier 2: training on DDXPlus '{train_name}' split, evaluating on 'test'")
    tr, te = load_split(train_name), load_split("test")
    if len(tr) > MAX_TRAIN:
        tr = tr.sample(MAX_TRAIN, random_state=SEED)
    print(f"  train={len(tr):,}  test={len(te):,}  conditions={tr['PATHOLOGY'].nunique()}")

    vec = CountVectorizer(token_pattern=r"[^\s]+", binary=True, lowercase=False)
    Xtr, Xte = build_X(vec, tr, fit=True), build_X(vec, te, fit=False)
    le = LabelEncoder().fit(tr["PATHOLOGY"])
    ytr, yte = le.transform(tr["PATHOLOGY"]), le.transform(te["PATHOLOGY"])

    from lightgbm import LGBMClassifier
    cands = {
        "logistic_regression": LogisticRegression(max_iter=300, C=1.0, n_jobs=-1),
        "lightgbm": LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.1,
                                   colsample_bytree=0.5, n_jobs=-1, random_state=SEED, verbose=-1),
    }
    results, fitted = {}, {}
    for name, est in cands.items():
        t0 = time.time()
        est.fit(Xtr, ytr)
        proba = est.predict_proba(Xte)
        pred = proba.argmax(1)
        results[name] = {
            "accuracy": round(float(accuracy_score(yte, pred)), 4),
            "macro_f1": round(float(f1_score(yte, pred, average="macro")), 4),
            "top3_accuracy": round(float(top_k_accuracy_score(yte, proba, k=3, labels=np.arange(len(le.classes_)))), 4),
            "fit_seconds": round(time.time() - t0, 1),
        }
        fitted[name] = est
        print(f"    {name:<20} acc {results[name]['accuracy']:.3f}  macroF1 {results[name]['macro_f1']:.3f}  top3 {results[name]['top3_accuracy']:.3f}  ({results[name]['fit_seconds']}s)")

    best = max(results, key=lambda k: results[k]["macro_f1"])
    conditions = json.loads(require(RAW / "release_conditions.json", HINT).read_text())
    evidences = json.loads(require(RAW / "release_evidences.json", HINT).read_text())
    joblib.dump({"model": fitted[best], "vectorizer": vec, "label_encoder": le,
                 "conditions": conditions, "evidences": evidences}, OUT / "tier2_model.joblib")
    bench = {"train_split": train_name, "train_rows": len(tr), "test_rows": len(te),
             "n_conditions": int(len(le.classes_)), "candidates": results, "selected_model": best,
             "conditions": sorted(le.classes_.tolist())}
    (OUT / "benchmark.json").write_text(json.dumps(bench, indent=2))
    print(f"  -> selected {best}; saved {OUT / 'tier2_model.joblib'}")
