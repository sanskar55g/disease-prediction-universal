"""
Tier 2: common acute / infectious disease symptom classifier data.

Source: DDXPlus (Fansi Tchango et al., NeurIPS 2022 Datasets track), hosted on
figshare, article 22687585. 49 conditions, 223 evidence (symptom/antecedent)
codes, ~1.3M cases generated from a validated medical knowledge base and
released by the authors. It is the standard public benchmark for symptom-to-
diagnosis models. We download the authors' release files as-is.

Run:  python backend/scripts/download_tier2.py --small   # validate+test only (~40 MB), quick start
      python backend/scripts/download_tier2.py           # also the 141 MB training split
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data_guard import record_provenance, tier_dir  # noqa: E402

OUT = tier_dir(2, "raw")
ARTICLE = "https://api.figshare.com/v2/articles/22687585"
SOURCE = "DDXPlus (figshare article 22687585)"

WANT_SMALL = {"release_conditions.json", "release_evidences.json",
              "release_validate_patients.zip", "release_test_patients.zip"}
WANT_FULL = WANT_SMALL | {"release_train_patients.zip"}


def download(url: str) -> bytes:
    with httpx.Client(follow_redirects=True, timeout=600) as c:
        r = c.get(url)
        r.raise_for_status()
        return r.content


if __name__ == "__main__":
    small = "--small" in sys.argv
    want = WANT_SMALL if small else WANT_FULL
    files = httpx.get(ARTICLE, timeout=60).json()["files"]
    for f in files:
        if f["name"] not in want:
            continue
        print(f"  fetching {f['name']} ({f['size']/1e6:.1f} MB)")
        blob = download(f["download_url"])
        if f["name"].endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                csv_name = [n for n in z.namelist() if not n.endswith("/")][0]  # file inside has no extension
                df = pd.read_csv(z.open(csv_name))
            out = OUT / f["name"].replace(".zip", ".parquet")
            df.to_parquet(out, index=False)
            record_provenance(out, SOURCE, f["download_url"], rows=len(df))
            print(f"     -> {out.name}: {len(df):,} patients")
        else:
            out = OUT / f["name"]
            out.write_bytes(blob)
            record_provenance(out, SOURCE, f["download_url"])
    print("done. provenance written to", OUT / "provenance.json")
