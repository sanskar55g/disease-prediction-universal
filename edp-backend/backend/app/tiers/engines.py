"""
Runtime engines for the four tiers. Each exposes .run(intake) -> dict and
.status() -> dict. All load only guarded, provenance-tracked artifacts.
Every engine is lazy: nothing is loaded until first use, so the API starts
even if a tier has not been built yet and reports that tier as unavailable.
"""
from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from ..data_guard import MODELS, MissingDataError, require, tier_dir
from .ontology import TermMatcher, parse_obo
from .tier1_registry import TIER1_DISEASES


# ----------------------------------------------------------------- Tier 1 ---
class Tier1:
    name = "Tier 1: Chronic disease risk models"

    def __init__(self):
        self._models: dict[str, dict] = {}
        self.bench = json.loads((MODELS / "tier1" / "benchmark.json").read_text()) if (MODELS / "tier1" / "benchmark.json").exists() else {}

    def available(self) -> list[str]:
        return [k for k in TIER1_DISEASES if (MODELS / "tier1" / f"{k}.joblib").exists()]

    def _model(self, key: str) -> dict:
        if key not in self._models:
            self._models[key] = joblib.load(require(MODELS / "tier1" / f"{key}.joblib", f"python backend/scripts/train_tier1.py {key}"))
        return self._models[key]

    def schema(self, key: str) -> dict:
        m = self._model(key)
        return {"disease": key, "label": TIER1_DISEASES[key]["label"], "features": m["features"], "dtypes": m["dtypes"]}

    def run(self, intake: dict) -> dict:
        """intake['tier1'] = {"disease": key, "features": {...}}. Missing features
        are imputed by the pipeline; we report which ones were missing."""
        req = intake.get("tier1") or {}
        key = req.get("disease")
        if key not in TIER1_DISEASES:
            return {"error": f"unknown disease '{key}'. Available: {self.available()}"}
        m = self._model(key)
        feats = req.get("features") or {}
        row = {f: feats.get(f, np.nan) for f in m["features"]}
        X = pd.DataFrame([row])
        for c, t in m["dtypes"].items():
            if t.startswith(("int", "float")):
                X[c] = pd.to_numeric(X[c], errors="coerce")
        p = float(m["pipeline"].predict_proba(X)[0, 1])
        band = "low" if p < 0.2 else "moderate" if p < 0.5 else "high"
        contrib = self._explain(m, X)
        b = self.bench.get(key, {})
        return {"disease": key, "label": TIER1_DISEASES[key]["label"], "risk_probability": round(p, 4),
                "risk_band": band, "model": b.get("selected_model"), "cv_auc": b.get("selected_auc"),
                "missing_features": [f for f in m["features"] if f not in feats],
                "top_drivers": contrib}

    @staticmethod
    def _explain(m: dict, X: pd.DataFrame) -> list[dict]:
        """Per-prediction attribution. Tree models via SHAP TreeExplainer;
        logistic regression via coefficient*value. Fails soft to []."""
        try:
            import shap
            pipe = m["pipeline"]
            Xt = pipe.named_steps["prep"].transform(X)
            names = pipe.named_steps["prep"].get_feature_names_out()
            clf = pipe.named_steps["clf"]
            if hasattr(clf, "coef_"):
                vals = (clf.coef_[0] * Xt[0])
            else:
                sv = shap.TreeExplainer(clf).shap_values(Xt)
                vals = sv[1][0] if isinstance(sv, list) else (sv[0, :, 1] if sv.ndim == 3 else sv[0])
            order = np.argsort(-np.abs(vals))[:6]
            return [{"feature": str(names[i]).split("__", 1)[-1], "impact": round(float(vals[i]), 4)} for i in order]
        except Exception:
            return []

    def status(self) -> dict:
        return {"ready": bool(self.available()), "diseases": self.available(),
                "benchmark": {k: {"model": v["selected_model"], "auc": v["selected_auc"], "rows": v["rows"]} for k, v in self.bench.items()}}


# ----------------------------------------------------------------- Tier 2 ---
class Tier2:
    name = "Tier 2: Acute / infectious symptom classifier"
    RED_FLAGS = {"chest pain": "possible cardiac event", "shortness of breath": "respiratory distress",
                 "difficulty breathing": "respiratory distress", "stiff neck": "possible meningitis",
                 "confusion": "altered mental status", "blue lips": "hypoxia", "seizure": "seizure",
                 "coughing blood": "haemoptysis", "severe abdominal pain": "acute abdomen",
                 "fainting": "syncope", "unresponsive": "emergency"}

    def __init__(self):
        self._art = None

    def _load(self):
        if self._art is None:
            self._art = joblib.load(require(MODELS / "tier2" / "tier2_model.joblib", "python backend/scripts/train_tier2.py"))
            ev = self._art["evidences"]
            # name -> code lookup from the English questions/values in the release
            self._name2code = {}
            for code, e in ev.items():
                self._name2code[e["question_en"].lower()] = code
                for tok in e.get("abbr_en", "").lower().split():
                    pass
        return self._art

    def red_flags(self, symptoms: list[str]) -> list[dict]:
        low = [s.lower() for s in symptoms]
        return [{"symptom": s, "reason": r} for s, r in self.RED_FLAGS.items() if any(s in x for x in low)]

    def run(self, intake: dict) -> dict:
        art = self._load()
        req = intake.get("tier2") or {}
        codes: list[str] = req.get("evidence_codes") or []
        age, sex = int(intake.get("age") or 40), (intake.get("sex") or "F")[0].upper()
        from scipy import sparse
        bag = art["vectorizer"].transform([" ".join(codes)])
        X = sparse.hstack([bag, sparse.csr_matrix([[age / 100.0]]), sparse.csr_matrix([[1.0 if sex == "M" else 0.0]])]).tocsr()
        proba = art["model"].predict_proba(X)[0]
        top = np.argsort(-proba)[:5]
        conds = art["conditions"]
        out = []
        for i in top:
            name = art["label_encoder"].classes_[i]
            c = conds.get(name, {})
            out.append({"condition": name, "probability": round(float(proba[i]), 4),
                        "severity_1_to_5": c.get("severity"), "icd10": c.get("icd10-id")})
        flags = self.red_flags(intake.get("symptoms") or [])
        triage = "EMERGENCY: seek care now" if flags else (
            "urgent care today" if out and (out[0]["severity_1_to_5"] or 5) <= 2 else
            "book a primary care visit" if out and (out[0]["severity_1_to_5"] or 5) <= 3 else "self-care and monitor")
        return {"differential": out, "red_flags": flags, "triage": triage, "n_evidence_used": len(codes),
                "note": "Trained on DDXPlus release; probabilities are model confidence, not prevalence."}

    def evidence_catalog(self) -> list[dict]:
        art = self._load()
        return [{"code": c, "question": e["question_en"], "type": e["data_type"],
                 "values": e.get("value_meaning", {})} for c, e in art["evidences"].items()]

    def status(self) -> dict:
        p = MODELS / "tier2" / "benchmark.json"
        return {"ready": (MODELS / "tier2" / "tier2_model.joblib").exists(), "benchmark": json.loads(p.read_text()) if p.exists() else None}


# ----------------------------------------------------------------- Tier 3 ---
class Tier3:
    name = "Tier 3: Broad differential reasoner (knowledge graph)"

    def __init__(self):
        self._idx = None
        self._matcher = None

    def _load(self):
        if self._idx is None:
            self._idx = joblib.load(require(tier_dir(3, "processed") / "index.joblib", "python backend/scripts/build_tier3.py"))
            self._matcher = TermMatcher(parse_obo(require(tier_dir(3, "raw") / "hp.obo", "python backend/scripts/download_tier3.py")))
        return self._idx

    def run(self, intake: dict) -> dict:
        idx = self._load()
        matched = self._matcher.match_many(intake.get("symptoms") or [])
        pt = {m["hpo_id"] for m in matched if m["hpo_id"]}
        if not pt:
            return {"matched_terms": matched, "differential": [], "note": "no symptom mapped to an HPO term"}
        idf = idx["idf"]
        denom = sum(idf.get(h, 0) for h in pt) or 1.0
        scored = []
        for d, hs in idx["disease_phenos"].items():
            hit = pt & hs
            if hit:
                scored.append((sum(idf.get(h, 0) for h in hit) / denom, d, hit))
        scored.sort(reverse=True)
        out = []
        for s, d, hit in scored[:10]:
            evidence = "high" if s >= 0.75 else "medium" if s >= 0.4 else "low"
            out.append({"disease_id": d, "name": idx["names"].get(d), "score": round(s, 3), "evidence_level": evidence,
                        "matched_phenotypes": sorted(hit & set(idx["disease_direct"].get(d, []))) or sorted(hit)[:5],
                        "codes": idx["xrefs"].get(d, [])})
        return {"matched_terms": matched, "differential": out, "searched_diseases": idx["n_diseases"]}

    def status(self) -> dict:
        return {"ready": (tier_dir(3, "processed") / "index.joblib").exists()}


# ----------------------------------------------------------------- Tier 4 ---
class Tier4:
    name = "Tier 4: Rare disease phenotype matcher"

    def __init__(self):
        self._idx = None
        self._hp = None
        self._matcher = None
        self._anc: dict[str, set] = {}

    def _load(self):
        if self._idx is None:
            self._idx = joblib.load(require(tier_dir(4, "processed") / "index.joblib", "python backend/scripts/build_tier4.py"))
            self._hp = parse_obo(require(tier_dir(4, "raw") / "hp.obo", "python backend/scripts/download_tier4.py"))
            self._matcher = TermMatcher(self._hp)
        return self._idx

    def run(self, intake: dict) -> dict:
        from .ontology import ancestors
        idx = self._load()
        phrases = list(intake.get("hpo_terms") or []) + list(intake.get("symptoms") or [])
        matched = self._matcher.match_many(phrases)
        pt = {m["hpo_id"] for m in matched if m["hpo_id"]}
        if not pt:
            return {"matched_terms": matched, "candidates": [], "note": "no phenotype mapped"}
        ic = idx["ic"]
        pt_anc = {h: ancestors(h, self._hp, self._anc) for h in pt}
        max_self = sum(ic.get(h, 0) for h in pt) or 1.0
        scored = []
        for orpha, d in idx["disorders"].items():
            exp = idx["expanded"][orpha]
            total, hits = 0.0, []
            for h in pt:
                common = pt_anc[h] & exp
                if not common:
                    continue
                best = max(common, key=lambda c: ic.get(c, 0))
                w = d["phenos"].get(h, 0.6 if best != h else 0.6)
                total += ic.get(best, 0) * (1.0 if best == h else 0.7) * max(w, 0.3)
                hits.append((h, best))
            if hits:
                scored.append((total / max_self, orpha, hits))
        scored.sort(reverse=True)
        out = []
        for s, orpha, hits in scored[:10]:
            meta = idx["meta"].get(orpha, {})
            out.append({"orpha_id": orpha, "name": idx["disorders"][orpha]["name"], "score": round(min(s, 1.0), 3),
                        "matched": [{"patient_term": h, "patient_name": self._hp.get(h, {}).get("name"),
                                     "disease_term": b, "disease_name": self._hp.get(b, {}).get("name"), "exact": h == b} for h, b in hits],
                        "unmatched_patient_terms": sorted(pt - {h for h, _ in hits}),
                        "codes": [x for x in meta.get("xrefs", []) if x.split(":")[0] in {"OMIM", "ICD-10", "ICD-11", "UMLS"}][:6],
                        "referral": "Medical genetics / rare disease clinic; consider genetic counselling"})
        return {"matched_terms": matched, "candidates": out, "searched_disorders": idx["n"]}

    def status(self) -> dict:
        return {"ready": (tier_dir(4, "processed") / "index.joblib").exists()}


@lru_cache(maxsize=1)
def engines() -> dict:
    return {1: Tier1(), 2: Tier2(), 3: Tier3(), 4: Tier4()}
