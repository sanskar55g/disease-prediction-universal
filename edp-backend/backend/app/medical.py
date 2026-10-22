from __future__ import annotations

import json
import os
import re
from pathlib import Path
from uuid import uuid4

import httpx
from pypdf import PdfReader

try:
    import fitz
except ImportError:
    fitz = None

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None

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


def _ocr_image(image) -> str:
    if pytesseract is None or Image is None:
        return ""
    try:
        return pytesseract.image_to_string(image).strip()
    except Exception:
        return ""


def extract_document_text(path: Path, mime_type: str) -> str:
    """Extract embedded PDF text first, then use local Tesseract OCR when available."""
    if mime_type == "application/pdf":
        text = extract_pdf_text(path)
        if text:
            return text
        if fitz is None:
            return ""
        from io import BytesIO
        chunks = []
        with fitz.open(path) as pdf:
            for page in pdf:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image = Image.open(BytesIO(pix.tobytes("png"))) if Image else None
                if image is not None:
                    page_text = _ocr_image(image)
                    if page_text:
                        chunks.append(page_text)
        return "\n".join(chunks).strip()

    if mime_type.startswith("image/") and Image is not None:
        with Image.open(path) as image:
            return _ocr_image(image)
    return ""


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


def _offline_extract_record(text: str, document_type: str) -> dict:
    """Conservative local fallback: capture literal text and obvious lab/value pairs without diagnosing."""
    normalized = " ".join(text.split())
    lines = [line.strip(" •-*\t") for line in text.splitlines() if line.strip()]
    medications = []
    labs = []
    # Preserve medication-like lines only when dosage/administration language is explicit.
    for line in lines:
        if re.search(r"\b(\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|iu)\b|tablet|capsule|take once|twice daily|daily dose)\b", line, re.I):
            medications.append({"text": line[:300], "source": "literal_document_text"})
        lab_match = re.search(r"([A-Za-z][A-Za-z0-9 /()-]{1,45})\s*[:=]\s*(-?\d+(?:\.\d+)?)\s*(mg/dl|mmol/l|g/dl|%|mmhg|10\^?\d*/?ul|iu/l|u/l|ng/ml)?", line, re.I)
        if lab_match:
            labs.append({"name": lab_match.group(1).strip(), "value": lab_match.group(2),
                         "unit": lab_match.group(3) or "", "source_text": line[:300]})
    return {
        "document_type": document_type,
        "summary": normalized[:2000],
        "conditions": [],
        "medications": medications[:50],
        "labs": labs[:100],
        "symptoms": [],
        "dates": list(dict.fromkeys(re.findall(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b", text)))[:50],
        "important_history": [],
        "extraction_method": "local_text_fallback",
        "processing_status": "processed_with_review" if medications or labs else "needs_review",
        "review_note": "Local fallback extracts literal lines and possible lab/value pairs only. Verify against the source; no diagnosis is inferred."
    }


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
        return _offline_extract_record(text, document_type)
    try:
        clean = raw[raw.find("{"):raw.rfind("}") + 1]
        out = json.loads(clean)
        out["document_type"] = document_type
        out["extraction_method"] = "groq_structured_extraction"
        out["processing_status"] = "processed"
        out.setdefault("conditions", [])
        out.setdefault("medications", [])
        out.setdefault("labs", [])
        out.setdefault("symptoms", [])
        out.setdefault("dates", [])
        out.setdefault("important_history", [])
        return out
    except Exception:
        return _offline_extract_record(text, document_type)


def build_history_context(records: list, documents: list, messages: list, max_chars: int = 24000) -> str:
    chunks = []
    for r in records:
        chunks.append(f"[{r.record_type}] {r.title}: {r.content}")
    for d in documents:
        if d.analysis_json:
            analysis = d.analysis_json
            if isinstance(analysis, str):
                try:
                    analysis = json.loads(analysis)
                except (TypeError, json.JSONDecodeError):
                    pass
            chunks.append(f"[document:{d.document_type}] {d.original_name}: {json.dumps(analysis, default=str)}")
        elif d.extracted_text:
            chunks.append(f"[document:{d.document_type}] {d.original_name}: {d.extracted_text[:3000]}")
    for m in messages[-30:]:
        chunks.append(f"[chat:{m.role}] {m.content}")
    return "\n".join(chunks)[-max_chars:]
