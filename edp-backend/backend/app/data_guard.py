"""
Data provenance guard.

Rule for this repo: nothing trains, indexes, or serves predictions from data that
was not downloaded from a named public source. There is no synthetic fallback
anywhere. If a file is missing, the code stops and tells you which download
script to run.

Every download script writes a provenance.json next to the files it fetched:
    {"source": "...", "url": "...", "sha256": "...", "fetched_at": "...", "rows": N}
Loaders call require() which checks the file exists AND has a provenance entry.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODELS = ROOT / "models"

if os.environ.get("ALLOW_SYNTHETIC_DATA"):
    sys.exit("ALLOW_SYNTHETIC_DATA is set. This project does not support synthetic data. Unset it.")


class MissingDataError(RuntimeError):
    pass


def tier_dir(tier: int, kind: str = "raw") -> Path:
    p = DATA / f"tier{tier}" / kind
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record_provenance(path: Path, source: str, url: str, rows: int | None = None, note: str = "") -> None:
    prov_file = path.parent / "provenance.json"
    prov = json.loads(prov_file.read_text()) if prov_file.exists() else {}
    prov[path.name] = {
        "source": source,
        "url": url,
        "sha256": sha256_of(path),
        "bytes": path.stat().st_size,
        "rows": rows,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": note,
    }
    prov_file.write_text(json.dumps(prov, indent=2))


def require(path: Path, download_hint: str) -> Path:
    """Return path if the file exists and has provenance, otherwise raise with a clear fix."""
    prov_file = path.parent / "provenance.json"
    if not path.exists():
        raise MissingDataError(
            f"Missing real data file: {path}\n"
            f"No synthetic fallback exists by design. Run:  {download_hint}"
        )
    if not prov_file.exists() or path.name not in json.loads(prov_file.read_text()):
        raise MissingDataError(
            f"{path} exists but has no provenance record. Files placed by hand are not accepted.\n"
            f"Re-fetch it with:  {download_hint}"
        )
    return path


def provenance(tier: int, kind: str = "raw") -> dict:
    f = tier_dir(tier, kind) / "provenance.json"
    return json.loads(f.read_text()) if f.exists() else {}
