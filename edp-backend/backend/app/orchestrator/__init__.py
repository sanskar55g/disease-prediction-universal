"""Orchestration layer public surface."""
from __future__ import annotations

from . import llm
from .rules import ACUTE_DAYS, CHRONIC_DAYS, RED_FLAGS, ROUTE_THRESHOLD, Decision, Orchestrator, Step

_orch = Orchestrator()


def classify(intake: dict) -> dict:
    """Single entry point used by the API and the simulator.
    Rules always decide; the LLM (if enabled) only enriches inputs and explains."""
    enriched, changed = llm.enrich_intake(intake)
    decision = _orch.route(enriched)
    d = decision.to_dict()
    if changed:
        d["mode"] = "rules+llm"
        d["llm_enrichment"] = changed
    d["llm_explanation"] = llm.explain(d)
    return d


def recent(n: int = 20) -> list[dict]:
    return _orch.recent(n)


__all__ = ["classify", "recent", "Orchestrator", "Decision", "Step",
           "ACUTE_DAYS", "CHRONIC_DAYS", "ROUTE_THRESHOLD", "RED_FLAGS"]
