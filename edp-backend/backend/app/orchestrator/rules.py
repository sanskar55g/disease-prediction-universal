"""
Orchestration layer: decides which tier(s) a patient intake goes to, and why.

Design
------
Rules first, LLM optional (see llm.py). Every decision is a list of trace steps
so the simulator window can replay it one step at a time:

    Step 1  Red-flag screen           -> emergency override?
    Step 2  Evidence inventory        -> what kinds of data did we receive?
    Step 3  Tier scoring              -> 0..1 score per tier, with reasons
    Step 4  Routing                   -> primary tier + secondary tiers
    Step 5  Hand-off                  -> the exact payload each tier receives

Tier semantics (from pipeline_architecture.md)
    Tier 1  structured chronic-risk data: vitals, labs, lifestyle, or an explicit screening request
    Tier 2  acute, short-duration symptoms (<= ACUTE_DAYS)
    Tier 3  non-specific / long-standing symptoms, or nothing else fits
    Tier 4  HPO-coded phenotypes, genomic data, or a chronic undiagnosed course

Nothing here touches models or data, so it can be unit-tested with plain dicts.
"""
from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any

ACUTE_DAYS = 14          # symptoms at or under this duration are "acute"
CHRONIC_DAYS = 90        # at or over this, "long-standing"
ROUTE_THRESHOLD = 0.35   # a tier also runs as secondary if its score clears this

RED_FLAGS: dict[str, str] = {
    "chest pain": "possible acute coronary syndrome",
    "crushing chest": "possible acute coronary syndrome",
    "shortness of breath": "respiratory distress",
    "difficulty breathing": "respiratory distress",
    "blue lips": "hypoxia",
    "stiff neck": "possible meningitis",
    "worst headache": "possible subarachnoid haemorrhage",
    "confusion": "altered mental status",
    "unresponsive": "unresponsive patient",
    "seizure": "active seizure",
    "coughing blood": "haemoptysis",
    "vomiting blood": "upper GI bleed",
    "severe abdominal pain": "acute abdomen",
    "fainting": "syncope",
    "slurred speech": "possible stroke",
    "face drooping": "possible stroke",
    "one sided weakness": "possible stroke",
    "suicidal": "psychiatric emergency",
}

TIER1_SIGNALS = {"bmi", "systolic", "diastolic", "blood_pressure", "hba1c", "glucose", "fasting_glucose",
                 "ldl", "hdl", "cholesterol", "triglycerides", "creatinine", "egfr", "bilirubin", "albumin",
                 "smoking", "alcohol", "physical_activity", "family_history", "ejection_fraction",
                 "serum_sodium", "platelets", "hemoglobin", "hypertension", "heart_rate"}


@dataclass
class Step:
    n: int
    name: str
    detail: str
    data: dict[str, Any] = field(default_factory=dict)
    ms: float = 0.0


@dataclass
class Decision:
    request_id: str
    primary_tier: int | None
    secondary_tiers: list[int]
    emergency: bool
    scores: dict[int, float]
    reasons: dict[int, list[str]]
    payloads: dict[int, dict]
    steps: list[Step]
    mode: str = "rules"           # "rules" or "rules+llm"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["scores"] = {str(k): v for k, v in self.scores.items()}
        d["reasons"] = {str(k): v for k, v in self.reasons.items()}
        d["payloads"] = {str(k): v for k, v in self.payloads.items()}
        return d


def _norm_keys(d: dict | None) -> set[str]:
    return {k.lower().replace(" ", "_") for k in (d or {})}


def _lower(xs: list[str] | None) -> list[str]:
    return [x.lower().strip() for x in (xs or []) if x and x.strip()]


class Orchestrator:
    """Stateless routing + an in-memory ring buffer of recent decisions (for the simulator)."""

    def __init__(self, history: int = 200):
        self.history: deque[Decision] = deque(maxlen=history)

    # ----------------------------------------------------------------- API --
    def route(self, intake: dict) -> Decision:
        t0 = time.perf_counter()
        steps: list[Step] = []
        rid = intake.get("request_id") or uuid.uuid4().hex[:10]

        def step(name: str, detail: str, **data) -> None:
            steps.append(Step(len(steps) + 1, name, detail, data, round((time.perf_counter() - t0) * 1000, 2)))

        symptoms = _lower(intake.get("symptoms"))
        free_text = (intake.get("free_text") or "").lower()
        duration = intake.get("symptom_duration_days")
        duration = float(duration) if duration not in (None, "") else None
        structured = intake.get("structured") or {}
        keys = _norm_keys(structured)
        hpo = [h for h in (intake.get("hpo_terms") or []) if h]
        genomic = bool(intake.get("genomic_data"))
        screening = bool(intake.get("screening_requested")) or bool(intake.get("tier1", {}).get("disease"))
        prior_tiers_failed = set(intake.get("prior_tiers_without_answer") or [])

        # -- Step 1: red flags ----------------------------------------------
        # A red-flag word only counts as an emergency if the presentation is
        # acute or of unknown duration. A lifelong seizure history is Tier 4
        # context, not an ambulance call.
        text_blob = " | ".join(symptoms + [free_text])
        chronic_context = (duration is not None and duration >= CHRONIC_DAYS) or any(
            k in free_text for k in ("since birth", "since childhood", "history of", "long-standing", "for years", "known "))
        hits = [{"trigger": k, "concern": v} for k, v in RED_FLAGS.items() if k in text_blob]
        flags = [] if chronic_context else hits
        emergency = bool(flags)
        step("Red-flag screen",
             f"{len(flags)} red flag(s) found" if flags else
             (f"{len(hits)} red-flag word(s) present but in a chronic context, not treated as emergency" if hits
              else "no red flags in symptoms or free text"),
             flags=flags, suppressed_in_chronic_context=hits if chronic_context else [], emergency_override=emergency)

        # -- Step 2: evidence inventory -------------------------------------
        t1_hits = sorted(keys & TIER1_SIGNALS)
        inventory = {
            "structured_fields": len(keys), "tier1_signal_fields": t1_hits,
            "symptom_count": len(symptoms), "free_text_chars": len(free_text),
            "duration_days": duration, "hpo_terms": len(hpo), "genomic_data": genomic,
            "screening_requested": screening, "prior_tiers_without_answer": sorted(prior_tiers_failed),
        }
        step("Evidence inventory", self._describe_inventory(inventory), **inventory)

        # -- Step 3: tier scoring -------------------------------------------
        scores: dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        reasons: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: []}

        def add(tier: int, pts: float, why: str) -> None:
            scores[tier] = min(1.0, scores[tier] + pts)
            reasons[tier].append(f"+{pts:.2f} {why}")

        # Tier 1: structured chronic-risk data or explicit screening
        if screening:
            add(1, 0.6, "screening explicitly requested")
        if t1_hits:
            add(1, min(0.6, 0.15 * len(t1_hits)), f"{len(t1_hits)} chronic-risk fields present: {', '.join(t1_hits[:5])}")
        elif keys:
            add(1, 0.15, f"{len(keys)} structured fields present (none are known chronic-risk signals)")
        if not symptoms and not free_text and keys:
            add(1, 0.2, "no acute complaint, data-only intake")

        # Tier 2: acute symptoms
        if symptoms or free_text:
            if duration is not None and duration <= ACUTE_DAYS:
                add(2, 0.6, f"symptom duration {duration:g} d is within the acute window (<= {ACUTE_DAYS} d)")
            elif duration is None:
                add(2, 0.35, "symptoms present, duration unknown (assume possibly acute)")
            add(2, min(0.3, 0.06 * len(symptoms)), f"{len(symptoms)} discrete symptom(s) to classify")
            if intake.get("onset") == "sudden":
                add(2, 0.15, "sudden onset")
            if intake.get("fever") or "fever" in text_blob:
                add(2, 0.1, "fever reported")

        # Tier 3: non-specific / long-standing / fallback
        if symptoms or free_text:
            if duration is not None and duration > ACUTE_DAYS:
                add(3, 0.4 if duration < CHRONIC_DAYS else 0.5, f"symptom duration {duration:g} d is beyond the acute window")
            if len(symptoms) >= 4:
                add(3, 0.2, "broad multi-system symptom set")
            if any(s in text_blob for s in ("fatigue", "tired", "weight loss", "malaise", "weakness", "night sweats")):
                add(3, 0.2, "non-specific constitutional symptoms")
            if 2 in prior_tiers_failed:
                add(3, 0.4, "Tier 2 previously returned no confident answer")
        if not any(scores.values()) and (symptoms or free_text):
            add(3, 0.3, "fallback: symptoms present but nothing else matched")

        # Tier 4: phenotype codes, genomics, chronic undiagnosed
        if hpo:
            add(4, min(0.7, 0.25 * len(hpo)), f"{len(hpo)} HPO-coded phenotype(s) supplied")
        if genomic:
            add(4, 0.5, "genomic / variant data supplied")
        if duration is not None and duration >= CHRONIC_DAYS and {2, 3} & prior_tiers_failed:
            add(4, 0.5, "chronic course and earlier tiers failed to explain it")
        if intake.get("family_history_rare") or "rare" in free_text or "genetic" in free_text or "syndrome" in free_text:
            add(4, 0.2, "rare / genetic context in history")
        if any(s in text_blob for s in ("developmental delay", "dysmorphic", "congenital", "since birth")):
            add(4, 0.3, "congenital / developmental features")

        scores = {k: round(v, 2) for k, v in scores.items()}
        step("Tier scoring", "; ".join(f"T{k}={v:.2f}" for k, v in scores.items()), scores=scores, reasons=reasons)

        # -- Step 4: routing ------------------------------------------------
        ranked = sorted(scores, key=lambda k: -scores[k])
        primary = ranked[0] if scores[ranked[0]] > 0 else None
        secondary = [t for t in ranked[1:] if scores[t] >= ROUTE_THRESHOLD]
        if emergency:
            detail = "EMERGENCY override: patient told to seek immediate care; tiers still run for the clinician view"
        elif primary is None:
            detail = "no usable evidence; ask the patient for symptoms, labs, or phenotype terms"
        else:
            detail = f"primary Tier {primary}" + (f", also Tier {', '.join(map(str, secondary))}" if secondary else "")
        step("Routing", detail, primary=primary, secondary=secondary, threshold=ROUTE_THRESHOLD, emergency=emergency)

        # -- Step 5: hand-off payloads --------------------------------------
        payloads: dict[int, dict] = {}
        for t in ([primary] if primary else []) + secondary:
            payloads[t] = self._payload(t, intake, symptoms, hpo, structured)
        step("Hand-off", f"prepared payloads for tier(s) {sorted(payloads)}" if payloads else "nothing to hand off",
             payloads={str(k): v for k, v in payloads.items()})

        d = Decision(rid, primary, secondary, emergency, scores, reasons, payloads, steps)
        self.history.append(d)
        return d

    def recent(self, n: int = 20) -> list[dict]:
        return [d.to_dict() for d in list(self.history)[-n:]][::-1]

    # ------------------------------------------------------------- helpers --
    @staticmethod
    def _payload(tier: int, intake: dict, symptoms: list[str], hpo: list[str], structured: dict) -> dict:
        base = {"age": intake.get("age"), "sex": intake.get("sex")}
        if tier == 1:
            return {**base, "disease": intake.get("tier1", {}).get("disease"), "features": structured}
        if tier == 2:
            return {**base, "symptoms": symptoms, "evidence_codes": intake.get("tier2", {}).get("evidence_codes", []),
                    "duration_days": intake.get("symptom_duration_days"), "onset": intake.get("onset")}
        if tier == 3:
            return {**base, "symptoms": symptoms, "free_text": intake.get("free_text"), "comorbidities": intake.get("comorbidities", [])}
        return {**base, "hpo_terms": hpo, "symptoms": symptoms, "genomic_data": intake.get("genomic_data")}

    @staticmethod
    def _describe_inventory(inv: dict) -> str:
        parts = []
        if inv["structured_fields"]:
            parts.append(f"{inv['structured_fields']} structured fields ({len(inv['tier1_signal_fields'])} chronic-risk signals)")
        if inv["symptom_count"]:
            parts.append(f"{inv['symptom_count']} symptoms" + (f" over {inv['duration_days']:g} days" if inv["duration_days"] is not None else ", duration unknown"))
        if inv["free_text_chars"]:
            parts.append(f"free text ({inv['free_text_chars']} chars)")
        if inv["hpo_terms"]:
            parts.append(f"{inv['hpo_terms']} HPO terms")
        if inv["genomic_data"]:
            parts.append("genomic data")
        if inv["screening_requested"]:
            parts.append("screening requested")
        return ", ".join(parts) if parts else "empty intake"
