"""
Tier 3: broad differential reasoner knowledge sources.

Free, license-free sources used now:
  - HPO ontology (hp.obo): 20k+ clinical finding terms with synonyms, used to
    map plain-language symptoms to codes.
  - phenotype.hpoa: disease -> phenotype annotations for ~12,900 diseases
    (OMIM + Orphanet + DECIPHER). This is the disease "knowledge graph" edges.
  - MONDO ontology: unified disease ids and names, cross-references to
    ICD-10, OMIM, Orphanet, SNOMED so the report can print familiar codes.

Licensed upgrades (documented in BUILD_PLAN.md, not fetched here):
  - SNOMED CT (India is an IHTSDO member; national release is free but needs
    registration), UMLS (free UTS account required). Drop RF2 / RRF files
    into data/tier3/raw/licensed/ and extend build_tier3.py.

Run:  python backend/scripts/download_tier3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data_guard import record_provenance, tier_dir  # noqa: E402

OUT = tier_dir(3, "raw")
FILES = {
    "hp.obo": ("Human Phenotype Ontology (JAX)",
               "https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo"),
    "phenotype.hpoa": ("HPO disease annotations (OMIM/Orphanet/DECIPHER)",
                       "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/phenotype.hpoa"),
    "mondo.obo": ("MONDO Disease Ontology (Monarch Initiative)",
                  "https://purl.obolibrary.org/obo/mondo.obo"),
}

if __name__ == "__main__":
    with httpx.Client(follow_redirects=True, timeout=600) as c:
        for name, (src, url) in FILES.items():
            print(f"  fetching {name} from {url}")
            r = c.get(url)
            r.raise_for_status()
            out = OUT / name
            out.write_bytes(r.content)
            rows = sum(1 for line in r.text.splitlines() if line and not line.startswith("#")) if name.endswith(".hpoa") else None
            record_provenance(out, src, url, rows=rows)
            print(f"     -> {out.name}: {out.stat().st_size/1e6:.1f} MB")
    print("done. provenance written to", OUT / "provenance.json")
