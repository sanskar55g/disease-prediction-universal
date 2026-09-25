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

EXTRACT_PROMPT = """You extract clinical intake fields from a patient's free text.
Return ONLY a JSON object with keys:
  symptoms: list of short symptom phrases (lowercase)
  symptom_duration_days: number or null
  onset: "sudden" | "gradual" | null
  fever: true|false|null
  hpo_like_terms: list of phenotype phrases that sound congenital/genetic/developmental (may be empty)
Do not add anything that is not stated. Text:
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

def enrich_intake(intake: dict) -> tuple[dict, dict]:
    if not enabled() or not intake.get("free_text"):
        return intake, {}
    try:
        raw = _chat([{"role": "user", "content": EXTRACT_PROMPT + intake["free_text"]}])
        data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except Exception as e:
        return intake, {"llm_error": str(e)[:200]}
    out, changed = dict(intake), {}
    if data.get("symptoms"):
        merged = sorted(set(map(str.lower, out.get("symptoms") or [])) | set(data["symptoms"]))
        if merged != sorted(map(str.lower, out.get("symptoms") or [])):
            out["symptoms"], changed["symptoms"] = merged, data["symptoms"]
    for k in ("symptom_duration_days", "onset", "fever"):
        if out.get(k) in (None, "") and data.get(k) is not None:
            out[k], changed[k] = data[k], data[k]
    if data.get("hpo_like_terms") and not out.get("hpo_terms"):
        out["hpo_terms"], changed["hpo_terms"] = data["hpo_like_terms"], data["hpo_like_terms"]
    return out, changed

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
