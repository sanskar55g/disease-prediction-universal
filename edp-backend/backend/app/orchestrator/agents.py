"""Sequential, auditable clinical-information agents for routing (not diagnosis)."""
from __future__ import annotations

import json
import re
from typing import Any


def _loads(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, json.JSONDecodeError):
            return {}
    return {}


def collect_saved_evidence(documents: list, records: list) -> dict:
    """Read only persisted structured analysis; uploaded originals are not retained."""
    items = []
    for doc in documents:
        analysis = _loads(getattr(doc, "analysis_json", ""))
        items.append({
            "document_id": doc.id,
            "filename": doc.original_name,
            "document_type": doc.document_type,
            "created_at": str(doc.created_at),
            "processing_status": analysis.get("processing_status", "unknown"),
            "analysis": analysis,
        })
    record_items = []
    for record in records:
        content = _loads(record.content)
        record_items.append({
            "record_id": record.id,
            "title": record.title,
            "record_type": record.record_type,
            "content": content if content else record.content[:2000],
            "source_document_id": record.source_document_id,
            "created_at": str(record.created_at),
        })
    return {"documents": items, "records": record_items}


def _facts(evidence: dict) -> dict:
    conditions, medications, labs, symptoms, dates, history = [], [], [], [], [], []
    for item in evidence.get("documents", []):
        analysis = item.get("analysis") or {}
        for key, target in (("conditions", conditions), ("medications", medications),
                            ("labs", labs), ("symptoms", symptoms), ("dates", dates),
                            ("important_history", history)):
            value = analysis.get(key, [])
            if isinstance(value, list):
                target.extend(value)
    for item in evidence.get("records", []):
        obj = item.get("content")
        if isinstance(obj, dict):
            for key, target in (("conditions", conditions), ("medications", medications),
                                ("labs", labs), ("symptoms", symptoms), ("dates", dates),
                                ("important_history", history)):
                value = obj.get(key, [])
                if isinstance(value, list):
                    target.extend(value)
    def unique(items):
        seen, result = set(), []
        for item in items:
            marker = json.dumps(item, sort_keys=True, default=str)
            if marker not in seen:
                seen.add(marker)
                result.append(item)
        return result
    return {"conditions": unique(conditions), "medications": unique(medications), "labs": unique(labs),
            "symptoms": unique(symptoms), "dates": unique(dates), "important_history": unique(history)}


def run_agent_pipeline(intake: dict, evidence: dict, decision: dict) -> tuple[dict, list[dict]]:
    """Produce one completed result per agent; every downstream agent gets prior outputs."""
    facts = _facts(evidence)
    history_findings = {
        "saved_document_count": len(evidence.get("documents", [])),
        "saved_record_count": len(evidence.get("records", [])),
        "known_conditions": facts["conditions"],
        "medications": facts["medications"],
        "labs": facts["labs"],
        "prior_symptoms": facts["symptoms"],
        "important_history": facts["important_history"],
        "note": "Facts are taken from saved structured records; they are not independently confirmed diagnoses."
    }
    symptoms = intake.get("symptoms") or []
    free_text = str(intake.get("free_text") or intake.get("message") or "").strip()
    duration = intake.get("symptom_duration_days")
    symptom_findings = {
        "reported_symptoms": symptoms,
        "patient_statement": free_text[:1500],
        "duration_days": duration,
        "onset": intake.get("onset"),
        "fever_reported": intake.get("fever"),
        "note": "These are patient-reported symptoms, not confirmed conditions.",
        "context_from_history_agent": {
            "known_conditions": facts["conditions"],
            "medications": facts["medications"],
            "prior_symptoms": facts["symptoms"],
            "important_history": facts["important_history"]
        }
    }
    evidence_findings = {
        "documents_reviewed": [
            {"document_id": x.get("document_id"), "filename": x.get("filename"),
             "type": x.get("document_type"), "status": x.get("processing_status"),
             "summary": (x.get("analysis") or {}).get("summary", "")}
            for x in evidence.get("documents", [])
        ],
        "extractable_facts": {k: facts[k] for k in ("conditions", "medications", "labs", "symptoms", "dates")},
        "processing_note": "PDF text and image OCR can extract readable text. Radiology pixels are not interpreted as imaging findings; unreadable files remain needs_review.",
        "context_from_history_agent": history_findings,
        "context_from_symptom_agent": symptom_findings
    }
    triage = {
        "emergency_flag": bool(decision.get("emergency")),
        "red_flags": decision.get("red_flags", []),
        "reasons": decision.get("reasons", {}),
        "instruction": ("Urgent clinical assessment may be needed based on the red-flag screen; do not wait for this demo."
                        if decision.get("emergency") else
                        "No configured red-flag trigger was detected. This does not rule out an emergency."),
        "context_from_history_agent": history_findings,
        "context_from_symptom_agent": symptom_findings,
        "context_from_evidence_agent": evidence_findings
    }
    tier = {
        "primary_tier": decision.get("primary_tier"),
        "recommended_tier": decision.get("recommended_tier", decision.get("primary_tier")),
        "secondary_tiers": decision.get("secondary_tiers", []),
        "scores": decision.get("scores", {}),
        "reasons": decision.get("reasons", {}),
        "decision_basis": "Deterministic routing rules; prediction/classification models were not executed.",
        "context_from_previous_agents": {
            "history": history_findings,
            "symptoms": symptom_findings,
            "evidence": evidence_findings,
            "triage": triage
        }
    }
    trace = [
        {"agent": "History Agent", "status": "completed", "message": f"Reviewed {history_findings['saved_document_count']} saved documents and {history_findings['saved_record_count']} saved records. Found {len(facts['medications'])} medication entries, {len(facts['labs'])} lab entries and {len(facts['conditions'])} condition entries.", "findings": history_findings, "consumed": []},
        {"agent": "Symptom Agent", "status": "completed", "message": f"Reviewed {len(symptoms)} structured patient-reported symptoms and the latest statement{f'; duration {duration} day(s)' if duration is not None else '; duration not specified'}.", "findings": symptom_findings, "consumed": ["History Agent"]},
        {"agent": "Evidence Agent", "status": "needs_review" if any(x.get("processing_status") in ("needs_review", "processed_with_review") for x in evidence.get("documents", [])) else "completed", "message": f"Reviewed {len(evidence.get('documents', []))} persisted document analyses and extracted structured facts. Documents requiring review: {sum(1 for x in evidence.get('documents', []) if x.get('processing_status') == 'needs_review')}.", "findings": evidence_findings, "consumed": ["History Agent", "Symptom Agent"]},
        {"agent": "Triage Agent", "status": "completed", "message": triage["instruction"], "findings": triage, "consumed": ["History Agent", "Symptom Agent", "Evidence Agent"]},
        {"agent": "Tier Router", "status": "completed" if tier["primary_tier"] is not None else "needs_more_information", "message": (f"Selected Tier {tier['primary_tier']} using deterministic routing rules." if tier["primary_tier"] is not None else "Insufficient evidence to select a tier; more information is needed."), "findings": tier, "consumed": ["History Agent", "Symptom Agent", "Evidence Agent", "Triage Agent"]},
    ]
    # Keep evidence facts available to deterministic routing, while preserving patient-provided fields.
    enriched = dict(intake)
    enriched["saved_evidence"] = facts
    merged_symptoms = list(enriched.get("symptoms") or [])
    for value in facts["symptoms"]:
        label = value.get("name") if isinstance(value, dict) else str(value)
        if label and label not in merged_symptoms:
            merged_symptoms.append(label)
    enriched["symptoms"] = merged_symptoms
    structured = dict(enriched.get("structured") or {})
    for lab in facts["labs"]:
        if isinstance(lab, dict) and lab.get("name") and lab.get("value") is not None:
            structured.setdefault(str(lab["name"]), lab["value"])
    enriched["structured"] = structured
    enriched["agent_trace"] = trace
    return enriched, trace
