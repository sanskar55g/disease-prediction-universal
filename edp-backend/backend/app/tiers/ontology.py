"""Shared, dependency-free OBO parser and symptom -> HPO term matcher."""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path


def parse_obo(path: Path) -> dict[str, dict]:
    """Return {term_id: {name, synonyms:[...], is_a:[...], xrefs:[...]}} for non-obsolete terms."""
    terms: dict[str, dict] = {}
    cur: dict | None = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                cur = {"synonyms": [], "is_a": [], "xrefs": [], "obsolete": False}
            elif line == "" and cur is not None:
                if "id" in cur and not cur["obsolete"]:
                    terms[cur["id"]] = cur
                cur = None
            elif cur is not None and ":" in line:
                key, val = line.split(": ", 1)
                if key == "id":
                    cur["id"] = val
                elif key == "name":
                    cur["name"] = val
                elif key == "synonym":
                    m = re.match(r'"(.*?)"', val)
                    if m:
                        cur["synonyms"].append(m.group(1))
                elif key == "is_a":
                    cur["is_a"].append(val.split(" ")[0])
                elif key == "xref":
                    cur["xrefs"].append(val.split(" ")[0])
                elif key == "is_obsolete" and val == "true":
                    cur["obsolete"] = True
    return terms


def ancestors(term: str, terms: dict[str, dict], cache: dict[str, set] | None = None) -> set[str]:
    cache = cache if cache is not None else {}
    if term in cache:
        return cache[term]
    out = {term}
    for p in terms.get(term, {}).get("is_a", []):
        out |= ancestors(p, terms, cache)
    cache[term] = out
    return out


_norm_re = re.compile(r"[^a-z0-9 ]+")


def norm(s: str) -> str:
    return _norm_re.sub(" ", s.lower()).strip()


class TermMatcher:
    """Map free-text symptom phrases to HPO ids using exact name/synonym match,
    then token-overlap fallback. Deterministic, no ML."""

    def __init__(self, terms: dict[str, dict], restrict_to_prefix: str = "HP:"):
        self.exact: dict[str, str] = {}
        self.token_index: dict[str, set[str]] = defaultdict(set)
        self.terms = terms
        for tid, t in terms.items():
            if not tid.startswith(restrict_to_prefix) or "name" not in t:
                continue
            for label in [t["name"], *t["synonyms"]]:
                n = norm(label)
                self.exact.setdefault(n, tid)
                for tok in n.split():
                    if len(tok) > 2:
                        self.token_index[tok].add(tid)

    def match(self, phrase: str) -> tuple[str | None, float]:
        n = norm(phrase)
        if n in self.exact:
            return self.exact[n], 1.0
        toks = [t for t in n.split() if len(t) > 2]
        if not toks:
            return None, 0.0
        scores: dict[str, int] = defaultdict(int)
        for t in toks:
            for tid in self.token_index.get(t, ()):
                scores[tid] += 1
        if not scores:
            return None, 0.0
        best = max(scores, key=lambda k: (scores[k], -len(self.terms[k]["name"])))
        cov = scores[best] / len(toks)
        return (best, round(cov, 2)) if cov >= 0.5 else (None, 0.0)

    def match_many(self, phrases: list[str]) -> list[dict]:
        out = []
        for p in phrases:
            p = p.strip()
            if not p:
                continue
            if p.upper().startswith("HP:"):
                out.append({"input": p, "hpo_id": p.upper(), "name": self.terms.get(p.upper(), {}).get("name", "?"), "confidence": 1.0})
                continue
            tid, conf = self.match(p)
            out.append({"input": p, "hpo_id": tid, "name": self.terms[tid]["name"] if tid else None, "confidence": conf})
        return out
