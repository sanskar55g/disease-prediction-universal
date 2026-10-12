"""
Optional LLM assist for the orchestrator. OFF unless an API key is set.

The LLM is never the routing authority. It can enrich free text and classify
the patient's high-level situation for the demo. Deterministic rules remain
the fallback when no key is configured or the LLM is unavailable.

Env vars:
  ORCH_LLM_API_KEY     OpenAI-compatible key. GROQ_API_KEY is also accepted.
  ORCH_LLM_BASE_URL    default https://api.openai.com/v1
  ORCH_LLM_MODEL       default gpt-4o-mini
"""
from __future__ import annotations

import json
import os

import httpx

EXTRACT_PROMPT = """You extract the patient's CURRENT CUMULATIVE clinical intake from a conversation.
Use the previous patient messages as context and the latest message as the newest evidence.

Return ONLY a JSON object with keys:
  symptoms: list of short symptom phrases (lowercase)
  symptom_duration_days: number or null
  onset: "sudden" | "gradual" | null
  fever: true|false|null
  hpo_like_terms: list of phenotype phrases that sound congenital/genetic/developmental (may be empty)

Rules:
- Combine information across the conversation; do not forget previously stated symptoms unless the patient explicitly says they resolved.
- If the latest message gives a new explicit duration, use that latest duration.
- If the latest message does not give a duration, retain the duration supported by earlier patient messages.
- Only extract information actually stated by the patient. Do not diagnose.
- Return an empty list or null when the conversation does not provide the field.

PREVIOUS PATIENT MESSAGES:
{previous_context}

LATEST PATIENT MESSAGE:
{latest_message}
"""

SITUATION_PROMPT = """Classify the patient's HIGH-LEVEL CLINICAL SITUATION for an orchestration demo.
Do not diagnose a disease and do not give treatment advice.

Return ONLY JSON with:
{
  "situation": one of [
    "acute_symptoms",
    "chronic_condition",
    "risk_screening",
    "rare_or_genetic",
    "mental_health",
    "general_or_unclear"
  ],
  "confidence": number from 0 to 1,
  "reason": "one short sentence based only on the supplied information"
}

Use acute_symptoms for new/short-duration symptoms or urgent presentations.
Use chronic_condition for persistent/long-standing symptoms or known chronic disease context.
Use risk_screening when the input is mainly vitals, labs, lifestyle, family history, or an explicit screening request.
Use rare_or_genetic for congenital/developmental/genetic/HPO/variant context.
Use mental_health when the main presentation is mental or behavioral symptoms.
Use general_or_unclear when there is not enough information.

Patient intake:
"""

def _api_key() -> str | None:
    return os.environ.get("ORCH_LLM_API_KEY") or os.environ.get("GROQ_API_KEY")

def enabled() -> bool:
    return bool(_api_key())

def _chat(messages: list[dict], max_tokens: int = 400) -> str:
    base = os.environ.get("ORCH_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    key = _api_key()
    if not key:
        raise RuntimeError("ORCH_LLM_API_KEY or GROQ_API_KEY is not configured")
    r = httpx.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": os.environ.get("ORCH_LLM_MODEL", "gpt-4o-mini"),
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def _duration_from_text(text: str) -> float | None:
    """Small offline fallback so chat remains longitudinal without an API key."""
    import re

    value = text.lower()
    matches = []
    patterns = [
        (r"(\\d+(?:\\.\\d+)?)\\s*(?:days?|d)\\b", 1.0),
        (r"(\\d+(?:\\.\\d+)?)\\s*(?:weeks?|wks?|wk)\\b", 7.0),
        (r"(\\d+(?:\\.\\d+)?)\\s*(?:months?|mos?|mo)\\b", 30.0),
        (r"(\\d+(?:\\.\\d+)?)\\s*(?:years?|yrs?|yr)\\b", 365.0),
    ]
    for pattern, multiplier in patterns:
        for match in re.finditer(pattern, value):
            matches.append((match.start(), float(match.group(1)) * multiplier))

    if any(x in value for x in ("today", "just started", "started today")):
        matches.append((len(value), 0.0))
    if any(x in value for x in ("yesterday", "since yesterday", "started yesterday")):
        matches.append((len(value), 1.0))

    return max(matches, key=lambda x: x[0])[1] if matches else None


def _offline_extract(text: str) -> dict:
    """Extract a conservative set of common clinical signals without an LLM."""
    value = text.lower()
    symptom_terms = [
        "headache", "migraine", "fever", "cough", "cold", "sore throat",
        "vomiting", "nausea", "diarrhea", "abdominal pain", "stomach pain",
        "chest pain", "shortness of breath", "difficulty breathing",
        "dizziness", "fainting", "fatigue", "tired", "weakness", "pain",
        "back pain", "joint pain", "muscle pain", "rash", "itching",
        "swelling", "weight loss", "night sweats", "insomnia", "anxiety",
        "depression", "panic", "hallucination", "seizure", "blurred vision",
        "vision problems", "palpitations", "heart racing", "blood in stool",
        "blood in urine",
    ]
    symptoms = [term for term in symptom_terms if term in value]
    onset = None
    if any(x in value for x in ("sudden", "suddenly", "all of a sudden")):
        onset = "sudden"
    elif any(x in value for x in ("gradual", "gradually", "slowly")):
        onset = "gradual"

    return {
        "symptoms": symptoms,
        "symptom_duration_days": _duration_from_text(text),
        "onset": onset,
        "fever": True if "fever" in value else None,
        "hpo_like_terms": [
            x for x in ("since birth", "since childhood", "developmental delay", "congenital", "genetic", "syndrome")
            if x in value
        ],
    }


def enrich_intake(intake: dict) -> tuple[dict, dict]:
    latest = str(intake.get("free_text") or "")
    previous = str(intake.get("conversation_context") or "")
    if not latest and not previous:
        return intake, {}

    # The patient history is used for extraction, not treated as an assistant
    # instruction. Only patient messages are placed in conversation_context.
    prompt = EXTRACT_PROMPT.format(previous_context=previous[-12000:], latest_message=latest)

    data = {}
    changed = {}
    if enabled():
        try:
            raw = _chat([{"role": "user", "content": prompt}])
            data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        except Exception as e:
            changed["llm_error"] = str(e)[:200]

    # Always keep an offline fallback. This also means the routing pipeline
    # still evolves across chat messages when no API key is configured.
    fallback = _offline_extract((previous + "\n" + latest).strip())
    if not data:
        data = fallback
    else:
        for key, value in fallback.items():
            if key == "symptoms":
                data[key] = sorted(set(map(str.lower, data.get(key) or [])) | set(value or []))
            elif data.get(key) in (None, "", []):
                data[key] = value

    out, changes = dict(intake), dict(changed)

    existing = set(map(str.lower, out.get("symptoms") or []))
    extracted = set(map(str.lower, data.get("symptoms") or []))
    merged = sorted(existing | extracted)
    if merged != sorted(existing):
        out["symptoms"] = merged
        changes["symptoms"] = merged

    # An explicit duration in the latest message should replace an older
    # duration; otherwise retain the duration extracted from the conversation.
    latest_duration = _duration_from_text(latest)
    duration = latest_duration if latest_duration is not None else data.get("symptom_duration_days")
    if out.get("symptom_duration_days") in (None, "") and duration is not None:
        out["symptom_duration_days"] = duration
        changes["symptom_duration_days"] = duration
    elif latest_duration is not None and out.get("symptom_duration_days") != latest_duration:
        out["symptom_duration_days"] = latest_duration
        changes["symptom_duration_days"] = latest_duration

    for key in ("onset", "fever"):
        if data.get(key) is not None and out.get(key) in (None, ""):
            out[key] = data[key]
            changes[key] = data[key]

    if data.get("hpo_like_terms"):
        existing_hpo = set(out.get("hpo_terms") or [])
        merged_hpo = sorted(existing_hpo | set(data["hpo_like_terms"]))
        if merged_hpo != sorted(existing_hpo):
            out["hpo_terms"] = merged_hpo
            changes["hpo_terms"] = merged_hpo

    return out, changes

def classify_situation(intake: dict) -> dict | None:
    """LLM-assisted situation classification. Returns None when offline."""
    if not enabled():
        return None
    try:
        raw = _chat([{"role": "user", "content": SITUATION_PROMPT + json.dumps(intake, default=str)}], max_tokens=180)
        data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        allowed = {
            "acute_symptoms", "chronic_condition", "risk_screening",
            "rare_or_genetic", "mental_health", "general_or_unclear"
        }
        if data.get("situation") not in allowed:
            return None
        data["confidence"] = round(max(0.0, min(1.0, float(data.get("confidence", 0.0)))), 2)
        data["reason"] = str(data.get("reason") or "")[:300]
        return data
    except Exception:
        return None

def explain(decision_dict: dict) -> str | None:
    if not enabled():
        return None
    try:
        return _chat([{"role": "user", "content":
            "In 3 plain sentences for a nurse, explain this routing decision. Do not add medical advice.\n"
            + json.dumps({k: decision_dict[k] for k in ("primary_tier", "secondary_tiers", "emergency", "scores", "reasons")})}],
            max_tokens=180)
    except Exception:
        return None
