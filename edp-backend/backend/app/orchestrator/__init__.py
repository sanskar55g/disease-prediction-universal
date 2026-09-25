"""Orchestration layer public surface."""
from __future__ import annotations

from . import llm
from .rules import ACUTE_DAYS, CHRONIC_DAYS, RED_FLAGS, ROUTE_THRESHOLD, Decision, Orchestrator, Step

_orch = Orchestrator()

def _rule_situation(intake: dict, decision: dict) -> dict:
    symptoms = [str(x).lower() for x in (intake.get("symptoms") or [])]
    text = (intake.get("free_text") or "").lower()
    duration = intake.get("symptom_duration_days")
    structured = intake.get("structured") or {}
    hpo = intake.get("hpo_terms") or []
    genomic = bool(intake.get("genomic_data"))
    screening = bool(intake.get("screening_requested")) or bool((intake.get("tier1") or {}).get("disease"))

    if hpo or genomic or any(x in text for x in ("genetic", "congenital", "since birth", "developmental", "syndrome")):
        return {"situation": "rare_or_genetic", "confidence": 0.9, "reason": "Genetic, congenital, developmental, phenotype, or genomic context is present."}
    if any(x in text or any(x in s for s in symptoms) for x in ("depressed", "depression", "anxiety", "panic", "suicidal", "insomnia", "sleep", "hallucination")):
        return {"situation": "mental_health", "confidence": 0.82, "reason": "The main information supplied is a mental or behavioral health presentation."}
    if screening or structured:
        return {"situation": "risk_screening", "confidence": 0.88, "reason": "The intake contains structured risk factors, measurements, labs, or an explicit screening request."}
    if symptoms or text:
        if duration is None or float(duration) <= ACUTE_DAYS:
            return {"situation": "acute_symptoms", "confidence": 0.86, "reason": "The patient has symptoms with an acute or unknown duration."}
        return {"situation": "chronic_condition", "confidence": 0.84, "reason": "The patient has symptoms that have persisted beyond the acute window."}
    return {"situation": "general_or_unclear", "confidence": 0.45, "reason": "There is not enough clinical information to classify the patient's situation."}

def classify(intake: dict) -> dict:
    """Classify the patient's situation first, then expose the model tier routing."""
    enriched, changed = llm.enrich_intake(intake)
    decision = _orch.route(enriched)
    d = decision.to_dict()

    situation = llm.classify_situation(enriched) or _rule_situation(enriched, d)
    d["patient_situation"] = situation
    d["recommended_tier"] = d.get("primary_tier")
    d["orchestration_primary"] = "patient_situation"

    if changed:
        d["mode"] = "rules+llm"
        d["llm_enrichment"] = changed
    else:
        d["mode"] = "rules"

    explanation = llm.explain(d)
    if explanation:
        d["llm_explanation"] = explanation
    return d

def recent(n: int = 20) -> list[dict]:
    return _orch.recent(n)

__all__ = ["classify", "recent", "Orchestrator", "Decision", "Step",
           "ACUTE_DAYS", "CHRONIC_DAYS", "ROUTE_THRESHOLD", "RED_FLAGS"]
