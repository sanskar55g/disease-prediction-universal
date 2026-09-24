"""
Tier 3 index builder: broad differential over ~12,900 diseases.

Builds data/tier3/processed/index.joblib from the real downloads:
  disease_phenos: {disease_id: set(HPO ids, expanded to ancestors)}
  idf:            information weight per HPO id (rarer phenotypes count more)
  names, xrefs:   disease names and ICD-10/OMIM/Orphanet/SNOMED codes via MONDO

No model is trained. Ranking at query time = IDF-weighted overlap between the
patient's HPO set and each disease's annotated set (a standard, explainable
knowledge-graph scoring approach; see BUILD_PLAN.md for the Bayesian upgrade).

Run:  python backend/scripts/build_tier3.py
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data_guard import require, tier_dir  # noqa: E402
from app.tiers.ontology import ancestors, parse_obo  # noqa: E402

RAW, OUT = tier_dir(3, "raw"), tier_dir(3, "processed")
HINT = "python backend/scripts/download_tier3.py"

if __name__ == "__main__":
    hp = parse_obo(require(RAW / "hp.obo", HINT))
    print(f"  HPO terms: {len(hp):,}")

    names: dict[str, str] = {}
    direct: dict[str, set[str]] = defaultdict(set)
    with open(require(RAW / "phenotype.hpoa", HINT), encoding="utf-8") as f:
        header = None
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if header is None:
                header = parts
                continue
            row = dict(zip(header, parts))
            if row.get("qualifier") == "NOT" or row.get("aspect") != "P":
                continue
            did = row["database_id"]
            names[did] = row["disease_name"]
            direct[did].add(row["hpo_id"])
    print(f"  diseases with phenotype annotations: {len(direct):,}")

    cache: dict[str, set] = {}
    expanded = {d: set().union(*(ancestors(h, hp, cache) for h in hs)) for d, hs in direct.items()}
    n = len(expanded)
    df = defaultdict(int)
    for hs in expanded.values():
        for h in hs:
            df[h] += 1
    idf = {h: math.log(n / c) for h, c in df.items()}

    # MONDO cross-references so the report can show ICD-10 / SNOMED codes
    mondo = parse_obo(require(RAW / "mondo.obo", HINT))
    xrefs: dict[str, list[str]] = defaultdict(list)
    for mid, t in mondo.items():
        for x in t["xrefs"]:
            key = x.replace("Orphanet:", "ORPHA:")
            if key in direct:
                xrefs[key] = sorted({xx for xx in t["xrefs"] if xx.split(":")[0] in {"ICD10CM", "ICD10", "SCTID", "OMIM", "Orphanet", "MESH"}})
    joblib.dump({"disease_phenos": expanded, "disease_direct": {d: sorted(h) for d, h in direct.items()},
                 "idf": idf, "names": names, "xrefs": dict(xrefs), "n_diseases": n}, OUT / "index.joblib")
    (OUT / "provenance.json").write_text('{"index.joblib": {"source": "built by build_tier3.py from tier3/raw", "n_diseases": %d}}' % n)
    print(f"  -> {OUT / 'index.joblib'}  ({n:,} diseases, {len(xrefs):,} with MONDO cross-refs)")
