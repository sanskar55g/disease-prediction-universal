from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import httpx
from pypdf import PdfReader

UPLOAD_ROOT = Path(os.getenv("UPLOAD_ROOT", "./uploads")).resolve()
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

ALLOWED = {
    "application/pdf": ".pdf",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


def save_upload(user_id: int, filename: str, mime_type: str, content: bytes) -> Path:
    if mime_type not in ALLOWED:
        raise ValueError("Only PDF, JPG, PNG and WEBP medical files are supported.")
    folder = UPLOAD_ROOT / str(user_id)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{uuid4().hex}{ALLOWED[mime_type]}"


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages).strip()


def infer_document_type(filename: str, text: str = "") -> str:
    value = f"{filename} {text[:4000]}".lower()
    if any(x in value for x in ("prescription", "rx", "tablet", "capsule", "dosage")):
        return "prescription"
    if any(x in value for x in ("blood test", "cbc", "hemoglobin", "glucose", "laboratory", "lab report")):
        return "lab_report"
    if any(x in value for x in ("x-ray", "xray", "radiology", "ct scan", "mri", "ultrasound")):
        return "scan"
    if any(x in value for x in ("discharge summary", "medical history", "clinical summary")):
        return "clinical_record"
    return "other"


def groq_enabled() -> bool:
    return bool(os.getenv("GROQ_API_KEY") or os.getenv("ORCH_LLM_API_KEY"))


def groq_chat(prompt: str, max_tokens: int = 900) -> str | None:
    key = os.getenv("GROQ_API_KEY") or os.getenv("ORCH_LLM_API_KEY")
    if not key:
        return None
    base = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
    model = os.getenv("GROQ_MODEL", os.getenv("ORCH_LLM_MODEL", "llama-3.3-70b-versatile"))
    try:
        r = httpx.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "temperature": 0, "max_tokens": max_tokens,
                  "messages":[
                      {"role":"system","content":"You are a medical-data extraction assistant for a student research application. Extract only information explicitly present. Do not diagnose or invent facts."},
                      {"role":"user","content":prompt}
                  ]},
            timeout=45,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


def extract_record(text: str, document_type: str) -> dict:
    if not text.strip():
        return {"document_type": document_type, "summary": "", "conditions": [], "medications": [], "labs": [], "symptoms": []}
    prompt = f"""Extract structured medical-history facts from this {document_type}.
Return ONLY JSON with keys: summary (string), conditions (list), medications (list),
labs (list of objects with name/value/unit/date when present), symptoms (list),
dates (list), and important_history (list). Never infer a diagnosis.
TEXT:
{text[:12000]}"""
    raw = groq_chat(prompt)
    if not raw:
        return {"document_type": document_type, "summary": text[:2000], "conditions": [], "medications": [], "labs": [], "symptoms": [], "dates": [], "important_history": []}
    try:
        clean = raw[raw.find("{"):raw.rfind("}") + 1]
        out = json.loads(clean)
        out["document_type"] = document_type
        return out
    except Exception:
        return {"document_type": document_type, "summary": raw[:4000], "conditions": [], "medications": [], "labs": [], "symptoms": [], "dates": [], "important_history": []}


def build_history_context(records: list, documents: list, messages: list, max_chars: int = 24000) -> str:
    chunks = []
    for r in records:
        chunks.append(f"[{r.record_type}] {r.title}: {r.content}")
    for d in documents:
        if d.analysis_json:
            chunks.append(f"[document:{d.document_type}] {d.original_name}: {d.analysis_json}")
        elif d.extracted_text:
            chunks.append(f"[document:{d.document_type}] {d.original_name}: {d.extracted_text[:3000]}")
    for m in messages[-30:]:
        chunks.append(f"[chat:{m.role}] {m.content}")
    return "\n".join(chunks)[-max_chars:]
