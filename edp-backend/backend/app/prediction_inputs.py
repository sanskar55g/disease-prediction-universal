"""Helpers for adapting router payloads to tier-engine input contracts."""
from __future__ import annotations


def build_prediction_engine_input(
    tier: int,
    intake: dict,
    payload: dict,
    symptoms: list[str],
    hpo_terms: list[str],
) -> dict:
    """Adapt router hand-off data to the input contract each engine expects."""
    model_input = {**intake, **payload}
    if tier == 1:
        original = intake.get("tier1") or {}
        model_input["tier1"] = {
            **original,
            "disease": payload.get("disease") or original.get("disease"),
            "features": payload.get("features") or intake.get("structured") or original.get("features") or {},
        }
    elif tier == 2:
        original = intake.get("tier2") or {}
        model_input["tier2"] = {
            **original,
            "evidence_codes": payload.get("evidence_codes") or original.get("evidence_codes") or [],
        }
    elif tier == 3:
        model_input["symptoms"] = payload.get("symptoms") or symptoms
    elif tier == 4:
        model_input["symptoms"] = payload.get("symptoms") or symptoms
        model_input["hpo_terms"] = payload.get("hpo_terms") or hpo_terms
    return model_input
