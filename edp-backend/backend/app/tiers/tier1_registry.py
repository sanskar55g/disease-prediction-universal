"""
Registry of Tier 1 chronic diseases.

Adding a disease = adding one entry here. The downloader, trainer and API all
read from this dict. Every entry points at a real UCI dataset id.

to_binary: maps the raw target column to 1 (disease / event present) or 0.
"""
from __future__ import annotations

import pandas as pd


def _eq(value):
    return lambda s: (s.astype(str).str.strip().str.lower() == value).astype(int)


def _gt0(s: pd.Series):
    return (pd.to_numeric(s, errors="coerce") > 0).astype(int)


TIER1_DISEASES: dict[str, dict] = {
    "heart_disease": {
        "label": "Coronary heart disease",
        "uci_id": 45, "target_col": "num", "to_binary": _gt0,
        "target_note": "num > 0 (any vessel narrowing)",
        "domain": "cardiovascular",
    },
    "heart_failure_death": {
        "label": "Heart failure: death event",
        "uci_id": 519, "target_col": "death_event", "to_binary": _gt0,
        "target_note": "death_event == 1",
        "domain": "cardiovascular",
    },
    "diabetes_brfss": {
        "label": "Type 2 diabetes (population survey)",
        "uci_id": 891, "target_col": "Diabetes_binary", "to_binary": _gt0,
        "target_note": "Diabetes_binary == 1",
        "domain": "metabolic",
    },
    "diabetes_early": {
        "label": "Early-stage diabetes (symptom based)",
        "uci_id": 529, "target_col": "class", "to_binary": _eq("positive"),
        "target_note": "class == Positive",
        "domain": "metabolic",
    },
    "chronic_kidney": {
        "label": "Chronic kidney disease",
        "uci_id": 336, "target_col": "class", "to_binary": _eq("ckd"),
        "target_note": "class == ckd",
        "domain": "renal",
    },
    "liver_disease": {
        "label": "Liver disease (ILPD)",
        "uci_id": 225, "target_col": "Selector", "to_binary": lambda s: (pd.to_numeric(s, errors="coerce") == 1).astype(int),
        "target_note": "Selector == 1 (liver patient)",
        "domain": "hepatic",
    },
    "cirrhosis_death": {
        "label": "Cirrhosis: death during follow-up",
        "uci_id": 878, "target_col": "Status", "to_binary": _eq("d"),
        "target_note": "Status == D (death)",
        "domain": "hepatic",
    },
    "parkinsons": {
        "label": "Parkinson's disease (voice features)",
        "uci_id": 174, "target_col": "status", "to_binary": _gt0,
        "target_note": "status == 1",
        "domain": "neurological",
    },
    "breast_cancer": {
        "label": "Breast cancer malignancy (FNA imaging features)",
        "uci_id": 17, "target_col": "Diagnosis", "to_binary": _eq("m"),
        "target_note": "Diagnosis == M (malignant)",
        "domain": "oncology",
    },
    "thyroid_cancer_recurrence": {
        "label": "Differentiated thyroid cancer recurrence",
        "uci_id": 915, "target_col": "Recurred", "to_binary": _eq("yes"),
        "target_note": "Recurred == Yes",
        "domain": "endocrine",
    },
}
