"""
Tier 4: rare and genetic disease phenotype matching sources.

Kept deliberately separate from Tier 3 even where sources overlap, so the two
tiers can be versioned, evaluated and swapped independently.

  - Orphanet en_product4.xml: rare disease -> HPO phenotype associations with
    frequency classes (obligate, very frequent, frequent, occasional, very
    rare, excluded). ~4,300 rare disorders. Free (CC BY 4.0).
  - Orphanet en_product1.xml: rare disease names, synonyms, ORPHAcodes and
    cross-refs to OMIM / ICD-10 / UMLS / MeSH.
  - hp.obo: a Tier 4 copy of the ontology so Tier 4 has its own pinned version
    for information-content and ancestor computations.

Licensed / registration upgrades (see BUILD_PLAN.md): OMIM full download
(requires academic licence), ClinVar variant summaries (free, large).

Run:  python backend/scripts/download_tier4.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data_guard import record_provenance, tier_dir  # noqa: E402

OUT = tier_dir(4, "raw")
FILES = {
    "en_product4.xml": ("Orphanet: rare disease HPO associations (Orphadata)",
                        "https://www.orphadata.com/data/xml/en_product4.xml"),
    "en_product1.xml": ("Orphanet: rare disease nomenclature and cross-references (Orphadata)",
                        "https://www.orphadata.com/data/xml/en_product1.xml"),
    "hp.obo": ("Human Phenotype Ontology (JAX), Tier 4 pinned copy",
               "https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo"),
}

if __name__ == "__main__":
    with httpx.Client(follow_redirects=True, timeout=900) as c:
        for name, (src, url) in FILES.items():
            print(f"  fetching {name} from {url}")
            r = c.get(url)
            r.raise_for_status()
            out = OUT / name
            out.write_bytes(r.content)
            record_provenance(out, src, url)
            print(f"     -> {out.name}: {out.stat().st_size/1e6:.1f} MB")
    print("done. provenance written to", OUT / "provenance.json")
