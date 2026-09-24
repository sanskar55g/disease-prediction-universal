"""
Tier 4 index builder: rare disease phenotype matcher from Orphanet.

Reads Orphanet en_product4.xml (disease -> HPO with frequency class) and
en_product1.xml (names, synonyms, cross-refs), plus Tier 4's own HPO copy.

Scoring at query time is Resnik-style: each HPO term gets an information
content IC = -log(p(term)) computed over Orphanet diseases; a patient term
matched to a disease term (or its nearest common ancestor) contributes that
IC, weighted by Orphanet frequency (obligate 1.0 ... very rare 0.2).

Run:  python backend/scripts/build_tier4.py
"""
from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data_guard import require, tier_dir  # noqa: E402
from app.tiers.ontology import ancestors, parse_obo  # noqa: E402

RAW, OUT = tier_dir(4, "raw"), tier_dir(4, "processed")
HINT = "python backend/scripts/download_tier4.py"
FREQ_W = {"Obligate (100%)": 1.0, "Very frequent (99-80%)": 0.9, "Frequent (79-30%)": 0.6,
          "Occasional (29-5%)": 0.3, "Very rare (<4-1%)": 0.15, "Excluded (0%)": 0.0}

if __name__ == "__main__":
    hp = parse_obo(require(RAW / "hp.obo", HINT))

    disorders: dict[str, dict] = {}
    root = ET.parse(require(RAW / "en_product4.xml", HINT)).getroot()
    for d in root.iter("Disorder"):
        orpha = "ORPHA:" + d.findtext("OrphaCode")
        name = d.findtext("Name")
        phenos = {}
        for a in d.iter("HPODisorderAssociation"):
            hid = a.find("HPO").findtext("HPOId")
            w = FREQ_W.get(a.findtext("HPOFrequency/Name") or "", 0.3)
            if w > 0:
                phenos[hid] = w
        if phenos:
            disorders[orpha] = {"name": name, "phenos": phenos}
    print(f"  Orphanet disorders with phenotypes: {len(disorders):,}")

    # names / synonyms / external codes from product1
    meta: dict[str, dict] = {}
    root1 = ET.parse(require(RAW / "en_product1.xml", HINT)).getroot()
    for d in root1.iter("Disorder"):
        orpha = "ORPHA:" + d.findtext("OrphaCode")
        syns = [s.text for s in d.iter("Synonym") if s.text]
        refs = [f"{r.findtext('Source')}:{r.findtext('Reference')}" for r in d.iter("ExternalReference")]
        meta[orpha] = {"synonyms": syns, "xrefs": refs, "type": d.findtext("DisorderType/Name")}

    cache: dict[str, set] = {}
    expanded = {o: set().union(*(ancestors(h, hp, cache) for h in d["phenos"])) for o, d in disorders.items()}
    n = len(expanded)
    cnt = defaultdict(int)
    for hs in expanded.values():
        for h in hs:
            cnt[h] += 1
    ic = {h: -math.log(c / n) for h, c in cnt.items()}

    joblib.dump({"disorders": disorders, "expanded": expanded, "ic": ic, "meta": meta, "n": n},
                OUT / "index.joblib")
    (OUT / "provenance.json").write_text('{"index.joblib": {"source": "built by build_tier4.py from tier4/raw", "n_disorders": %d}}' % n)
    print(f"  -> {OUT / 'index.joblib'}")
