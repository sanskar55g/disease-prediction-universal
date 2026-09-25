from __future__ import annotations

import json
import os

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .database import ChatMessage, MedicalDocument, MedicalRecord, Prediction, SessionLocal, User, init_db
from .medical import build_history_context, extract_pdf_text, extract_record, infer_document_type, save_upload
from .orchestrator import classify
from .schemas import ChatRequest, LoginRequest, PredictionRequest, RegisterRequest
from .security import create_token, hash_password, user_id_from_token, verify_password
from .tiers.engines import engines

app = FastAPI(title="Universal Disease Prediction API", version="1.0.0")

origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "*").split(",")]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
init_db()

def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

def current_user(authorization: str | None = Header(default=None), session: Session = Depends(db)) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Bearer token required")
    try:
        uid = user_id_from_token(authorization.split(" ", 1)[1])
    except Exception:
        raise HTTPException(401, "Invalid or expired token")
    user = session.get(User, uid)
    if not user:
        raise HTTPException(401, "User not found")
    return user

def history_for(user: User, session: Session) -> str:
    records = session.query(MedicalRecord).filter_by(user_id=user.id).order_by(MedicalRecord.created_at.asc()).all()
    docs = session.query(MedicalDocument).filter_by(user_id=user.id).order_by(MedicalDocument.created_at.asc()).all()
    msgs = session.query(ChatMessage).filter_by(user_id=user.id).order_by(ChatMessage.created_at.asc()).all()
    return build_history_context(records, docs, msgs)

def run_prediction(user: User, intake: dict, session: Session, history: str) -> dict:
    intake = dict(intake)
    intake["medical_history"] = history
    decision = classify(intake)
    results = {}
    for tier in [decision.get("primary_tier"), *(decision.get("secondary_tiers") or [])]:
        if not tier:
            continue
        try:
            payload = decision.get("payloads", {}).get(str(tier), {})
            results[str(tier)] = engines()[tier].run({**intake, **payload})
        except Exception as exc:
            results[str(tier)] = {"available": False, "error": str(exc)[:500]}
    result = {"routing": decision, "tier_results": results, "medical_history_used": bool(history)}
    session.add(Prediction(user_id=user.id, request_json=json.dumps(intake, default=str), result_json=json.dumps(result, default=str)))
    session.commit()
    return result

@app.get("/api/health")
def health():
    return {"ok": True, "service": "universal-disease-prediction", "version": app.version}

@app.post("/api/orchestrator/classify")
def orchestrator_classify(intake: dict):
    """Demo endpoint: classify patient situation and show the downstream tier recommendation.
    No authentication and no patient data is persisted. Use only with local/demo data."""
    return classify(intake)

@app.post("/api/auth/register")
def register(req: RegisterRequest, session: Session = Depends(db)):
    email = req.email.strip().lower()
    if session.query(User).filter_by(email=email).first():
        raise HTTPException(409, "Email already registered")
    user = User(email=email, name=req.name.strip(), password_hash=hash_password(req.password))
    session.add(user)
    session.commit()
    session.refresh(user)
    return {"user": {"id": user.id, "email": user.email, "name": user.name}, "token": create_token(user.id)}

@app.post("/api/auth/login")
def login(req: LoginRequest, session: Session = Depends(db)):
    user = session.query(User).filter_by(email=req.email.strip().lower()).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(401, "Invalid email or password")
    return {"user": {"id": user.id, "email": user.email, "name": user.name}, "token": create_token(user.id)}

@app.get("/api/me")
def me(user: User = Depends(current_user)):
    return {"id": user.id, "email": user.email, "name": user.name, "created_at": user.created_at}

@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...), document_type: str = Form("auto"),
                          user: User = Depends(current_user), session: Session = Depends(db)):
    data = await file.read()
    if len(data) > int(os.getenv("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024))):
        raise HTTPException(413, "File is too large")
    try:
        path = save_upload(user.id, file.filename or "medical-file", file.content_type or "", data)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    path.write_bytes(data)
    extracted = extract_pdf_text(path) if file.content_type == "application/pdf" else ""
    dtype = document_type if document_type != "auto" else infer_document_type(file.filename or "", extracted)
    analysis = extract_record(extracted, dtype) if extracted else {
        "document_type": dtype,
        "summary": "Image stored. Connect a vision-capable model to extract image findings.",
        "conditions": [], "medications": [], "labs": [], "symptoms": []
    }
    doc = MedicalDocument(user_id=user.id, original_name=file.filename or path.name, stored_path=str(path),
                          mime_type=file.content_type or "", document_type=dtype,
                          extracted_text=extracted, analysis_json=json.dumps(analysis, default=str))
    session.add(doc)
    session.commit()
    session.refresh(doc)
    session.add(MedicalRecord(user_id=user.id, record_type=dtype, title=doc.original_name,
                              content=json.dumps(analysis, default=str), source_document_id=doc.id))
    session.commit()
    return {"id": doc.id, "filename": doc.original_name, "document_type": dtype, "analysis": analysis}

@app.get("/api/history")
def history(user: User = Depends(current_user), session: Session = Depends(db)):
    docs = session.query(MedicalDocument).filter_by(user_id=user.id).order_by(MedicalDocument.created_at.desc()).all()
    records = session.query(MedicalRecord).filter_by(user_id=user.id).order_by(MedicalRecord.created_at.desc()).all()
    predictions = session.query(Prediction).filter_by(user_id=user.id).order_by(Prediction.created_at.desc()).limit(50).all()
    return {
        "documents": [{"id": d.id, "name": d.original_name, "type": d.document_type, "created_at": d.created_at} for d in docs],
        "records": [{"id": r.id, "type": r.record_type, "title": r.title, "content": r.content, "created_at": r.created_at} for r in records],
        "predictions": [{"id": p.id, "result": json.loads(p.result_json), "created_at": p.created_at} for p in predictions]
    }

@app.post("/api/predict")
def predict(req: PredictionRequest, user: User = Depends(current_user), session: Session = Depends(db)):
    return run_prediction(user, req.model_dump(), session, history_for(user, session))

@app.post("/api/chat")
def chat(req: ChatRequest, user: User = Depends(current_user), session: Session = Depends(db)):
    session.add(ChatMessage(user_id=user.id, role="user", content=req.message))
    session.commit()
    intake = req.model_dump()
    intake["free_text"] = req.message
    if not intake["symptoms"]:
        intake["symptoms"] = [req.message]
    result = run_prediction(user, intake, session, history_for(user, session))
    reply = {"message": "Your message was added to your longitudinal history and processed by the routing pipeline. Model outputs are not a diagnosis.", "result": result}
    session.add(ChatMessage(user_id=user.id, role="assistant", content=json.dumps(reply, default=str)))
    session.commit()
    return reply

@app.get("/api/orchestrator/history")
def orchestrator_history(limit: int = 20, user: User = Depends(current_user), session: Session = Depends(db)):
    rows = session.query(Prediction).filter_by(user_id=user.id).order_by(Prediction.created_at.desc()).limit(min(limit, 100)).all()
    return [{"id": p.id, "routing": json.loads(p.result_json).get("routing"), "created_at": p.created_at} for p in rows]

@app.get("/api/status")
def status(user: User = Depends(current_user)):
    out = {}
    for tier, engine in engines().items():
        try:
            out[str(tier)] = engine.status()
        except Exception as exc:
            out[str(tier)] = {"ready": False, "error": str(exc)[:300]}
    return {"tiers": out, "groq_enabled": bool(os.getenv("GROQ_API_KEY") or os.getenv("ORCH_LLM_API_KEY"))}

@app.get("/api/tier1/diseases")
def tier1_diseases(user: User = Depends(current_user)):
    return engines()[1].status()

@app.get("/api/tier2/evidence")
def tier2_evidence(user: User = Depends(current_user)):
    try:
        return {"evidence": engines()[2].evidence_catalog()}
    except Exception as exc:
        raise HTTPException(503, str(exc))

@app.get("/api/medical-disclaimer")
def medical_disclaimer():
    return {"text": "Research/demo decision-support only. It does not diagnose disease or replace a qualified clinician."}
