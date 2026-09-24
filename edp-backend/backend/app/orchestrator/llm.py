"""
Optional LLM assist for the orchestrator. OFF unless an API key is set.

What it does when enabled (and only then):
  1. Reads free_text and extracts a structured list of symptoms, duration in
     days, onset, and any phenotype-like terms, so the rules engine has better
     inputs. The rules engine still makes the routing decision.
  2. Writes a one-paragraph plain-language explanation of the rules decision.

What it never does: pick the tier on its own, invent data, or run when offline.

Env vars:
  ORCH_LLM_API_KEY     any OpenAI-compatible key (OpenAI, Groq, Together, Ollama proxy...)
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


def enabled() -> bool:
    return bool(os.environ.get("ORCH_LLM_API_KEY"))


def _chat(messages: list[dict], max_tokens: int = 400) -> str:
    base = os.environ.get("ORCH_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    r = httpx.post(f"{base}/chat/completions",
                   headers={"Authorization": f"Bearer {os.environ['ORCH_LLM_API_KEY']}"},
                   json={"model": os.environ.get("ORCH_LLM_MODEL", "gpt-4o-mini"), "messages": messages,
                         "temperature": 0, "max_tokens": max_tokens},
                   timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def enrich_intake(intake: dict) -> tuple[dict, dict]:
    """Fill missing structured fields from free_text. Returns (new_intake, what_changed)."""
    if not enabled() or not intake.get("free_text"):
        return intake, {}
    try:
        raw = _chat([{"role": "user", "content": EXTRACT_PROMPT + intake["free_text"]}])
        data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except Exception as e:  # LLM problems never block routing
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
