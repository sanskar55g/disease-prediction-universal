from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

GROQ_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def generate_guidance(prediction: dict, symptoms: str = "", relevant_history: str = "") -> dict:
    """Ask Groq for cautious, patient-friendly next-step guidance.

    The model output is informational, not a diagnosis or a treatment plan.
    The API key is read from the process environment (load .env in main.py/app startup).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured in the backend environment")

    system_prompt = """You are a cautious health-education assistant, not a doctor. Use only the supplied prediction output and patient context; do not invent test findings or claim the prediction is confirmed. Return ONLY valid JSON with this schema: {"most_likely_condition": string, "confidence_note": string, "specialists": [string], "prevention_measures": [string], "diet_suggestions": [string], "activity_suggestions": [string], "warning_signs": [string], "when_to_seek_care": string, "disclaimer": string}. Infer the appropriate specialist from the suspected condition, and explain that the clinician should confirm the diagnosis. Give general, low-risk lifestyle information only. Do not prescribe medication, dosing, or claim that a diet/exercise cures a disease. Adapt activity advice to the symptoms and advise against strenuous activity when symptoms could make it unsafe. Include urgent warning signs appropriate to the reported context. If the result is uncertain or unavailable, say so and recommend clinical assessment."""
    user_payload = {
        "prediction_result": prediction,
        "reported_symptoms": symptoms,
        "relevant_history": relevant_history,
    }
    body = json.dumps({
        "model": GROQ_MODEL,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, default=str)},
        ],
    }).encode("utf-8")
    request = Request(GROQ_URL, data=body, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }, method="POST")
    try:
        with urlopen(request, timeout=45) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Groq API returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not reach the Groq API: {exc}") from exc

    try:
        content = response_data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        content = str(content).strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]
        result = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Groq returned a response that was not valid JSON guidance") from exc

    # Ensure predictable frontend fields and avoid rendering unexpected types.
    list_fields = ("specialists", "prevention_measures", "diet_suggestions", "activity_suggestions", "warning_signs")
    for field in list_fields:
        value = result.get(field, [])
        result[field] = [str(item)[:500] for item in value] if isinstance(value, list) else []
    for field in ("most_likely_condition", "confidence_note", "when_to_seek_care", "disclaimer"):
        result[field] = str(result.get(field, "") or "")[:1500]
    result["source"] = "Groq generated guidance"
    result["is_medical_diagnosis"] = False
    return result
