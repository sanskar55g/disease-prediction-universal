from __future__ import annotations

import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .database import ChatMessage, MedicalDocument, MedicalRecord, Prediction, SessionLocal, User, init_db
from .medical import build_history_context, extract_document_text, extract_record, infer_document_type, save_upload
from .orchestrator import classify
from .orchestrator.agents import collect_saved_evidence, run_agent_pipeline
from .schemas import ChatRequest, LoginRequest, PredictionRequest, RegisterRequest
from .security import create_token, hash_password, user_id_from_token, verify_password
from .tiers.engines import engines
from .guidance import answer_prediction_question, generate_guidance
from .prediction_inputs import build_prediction_engine_input
from .prediction_input_builder import answers_to_evidence_codes

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

def history_for(user: User, session: Session, session_id: str | None = None) -> str:
    records = session.query(MedicalRecord).filter_by(user_id=user.id).order_by(MedicalRecord.created_at.asc()).all()
    docs = session.query(MedicalDocument).filter_by(user_id=user.id).order_by(MedicalDocument.created_at.asc()).all()
    message_query = session.query(ChatMessage).filter_by(user_id=user.id)
    if session_id:
        message_query = message_query.filter(ChatMessage.session_id == session_id)
    else:
        message_query = message_query.filter(ChatMessage.session_id.is_(None))
    msgs = message_query.order_by(ChatMessage.created_at.asc()).all()
    return build_history_context(records, docs, msgs)

def chat_context_for(user: User, session: Session, session_id: str | None = None, limit: int = 30) -> str:
    msgs = (
        session.query(ChatMessage)
        .filter_by(user_id=user.id, role="user")
        .filter(ChatMessage.session_id == session_id if session_id else ChatMessage.session_id.is_(None))
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    return "\n".join(f"[patient] {m.content}" for m in msgs[-limit:])

def route_with_agents(user: User, intake: dict, session: Session, session_id: str | None = None) -> dict:
    """Run all five agents to completion, using persisted evidence before tier routing."""
    docs = session.query(MedicalDocument).filter_by(user_id=user.id).order_by(MedicalDocument.created_at.asc()).all()
    records = session.query(MedicalRecord).filter_by(user_id=user.id).order_by(MedicalRecord.created_at.asc()).all()
    evidence = collect_saved_evidence(docs, records)
    enriched = dict(intake)
    enriched["medical_history"] = history_for(user, session, session_id)
    enriched, _ = run_agent_pipeline(enriched, evidence, {})
    decision = classify(enriched)
    _, trace = run_agent_pipeline(enriched, evidence, decision)
    decision["agent_trace"] = trace
    decision["medical_history_used"] = bool(evidence.get("documents") or evidence.get("records"))
    decision["evidence_summary"] = {
        "documents_reviewed": len(evidence.get("documents", [])),
        "records_reviewed": len(evidence.get("records", [])),
        "needs_review": sum(1 for d in evidence.get("documents", []) if d.get("processing_status") in ("needs_review", "processed_with_review"))
    }
    decision["message"] = "Orchestration completed through tier routing. Disease prediction models were not executed."
    return decision


def run_prediction(user: User, intake: dict, session: Session, history: str) -> dict:
    intake = dict(intake)

    # Convert structured Tier 2 questionnaire answers into DDXPlus evidence tokens.
    tier2_input = dict(intake.get("tier2") or {})
    evidence_answers = tier2_input.pop("answers", None)

    if evidence_answers is not None:
        if not isinstance(evidence_answers, dict):
            raise HTTPException(
                status_code=422,
                detail="tier2.answers must be an object of evidence answers.",
            )

        try:
            evidence_catalog = engines()[2].evidence_catalog()
            generated_codes = answers_to_evidence_codes(
                evidence_catalog,
                evidence_answers,
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Could not load the Tier 2 evidence catalog: {str(exc)[:200]}",
            ) from exc

        existing_codes = tier2_input.get("evidence_codes") or []
        if not isinstance(existing_codes, list) or not all(
            isinstance(code, str) and code.strip() for code in existing_codes
        ):
            raise HTTPException(
                status_code=422,
                detail="tier2.evidence_codes must be a list of non-empty strings.",
            )

        tier2_input["evidence_codes"] = list(
            dict.fromkeys(
                [code.strip() for code in existing_codes] + generated_codes
            )
        )

    intake["tier2"] = tier2_input
    intake["medical_history"] = history
    decision = classify(intake)
    routed_payloads = decision.get("payloads", {}) or {}

    # The router enriches free text before building its payloads. Reuse those
    # extracted symptoms for every phenotype tier, even if that tier was not
    # selected as primary/secondary by the router.
    symptoms = list(intake.get("symptoms") or [])
    hpo_terms = list(intake.get("hpo_terms") or [])
    for payload in routed_payloads.values():
        if not symptoms and payload.get("symptoms"):
            symptoms = list(payload["symptoms"])
        if not hpo_terms and payload.get("hpo_terms"):
            hpo_terms = list(payload["hpo_terms"])

    results = {}
    for tier, engine in engines().items():
        payload = routed_payloads.get(str(tier), {}) or {}
        try:
            if tier == 1 and not ((intake.get("tier1") or {}).get("disease") or payload.get("disease")):
                results[str(tier)] = {
                    "available": False,
                    "skipped": True,
                    "reason": "Tier 1 is a disease-specific risk model. Select a Tier 1 disease and provide its structured risk features before running it.",
                }
                continue
            if tier == 2 and not ((intake.get("tier2") or {}).get("evidence_codes") or payload.get("evidence_codes")):
                results[str(tier)] = {
                    "available": False,
                    "skipped": True,
                    "reason": "Tier 2 requires DDXPlus evidence codes. Submit structured questionnaire answers in tier2.answers or provide tier2.evidence_codes; free-text symptoms alone are not converted into DDXPlus features.",
                }
                continue
            model_input = build_prediction_engine_input(tier, intake, payload, symptoms, hpo_terms)
            results[str(tier)] = engine.run(model_input)
        except Exception as exc:
            results[str(tier)] = {"available": False, "error": str(exc)[:500]}
    result = {
        "routing": decision,
        "tier_results": results,
        "all_tiers_evaluated": True,
        "medical_history_used": bool(history),
    }
    session.add(Prediction(user_id=user.id, request_json=json.dumps(intake, default=str), result_json=json.dumps(result, default=str)))
    session.commit()
    return result

@app.get("/api/health")
def health():
    return {"ok": True, "service": "universal-disease-prediction", "version": app.version}

@app.post("/api/orchestrator/classify")
def orchestrator_classify(intake: dict):
    """Local/demo endpoint for routing without running prediction models."""
    return classify(intake)


@app.post("/api/orchestrator/route")
def orchestrator_route(intake: dict, user: User = Depends(current_user), session: Session = Depends(db)):
    """Run orchestration through tier selection using the authenticated patient history."""
    decision = route_with_agents(user, intake, session)
    return {
        "status": "routed" if decision.get("primary_tier") is not None else "needs_more_information",
        "patient_situation": decision.get("patient_situation"),
        "primary_tier": decision.get("primary_tier"),
        "recommended_tier": decision.get("recommended_tier"),
        "secondary_tiers": decision.get("secondary_tiers", []),
        "emergency": decision.get("emergency", False),
        "scores": decision.get("scores", {}),
        "reasons": decision.get("reasons", {}),
        "steps": decision.get("steps", []),
        "agent_trace": decision.get("agent_trace", []),
        "evidence_summary": decision.get("evidence_summary", {}),
        "request_id": decision.get("request_id"),
        "medical_history_used": decision.get("medical_history_used", False),
        "message": decision.get("message")
    }
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
    try:
        extracted = extract_document_text(path, file.content_type or "")
        dtype = document_type if document_type != "auto" else infer_document_type(file.filename or "", extracted)
        analysis = extract_record(extracted, dtype) if extracted.strip() else {
            "document_type": dtype,
            "summary": "No readable text was extracted. Check image quality or OCR setup; findings were not inferred.",
            "conditions": [], "medications": [], "labs": [], "symptoms": [],
            "dates": [], "important_history": [], "processing_status": "needs_review"
        }
        analysis.setdefault("document_type", dtype)
        analysis.setdefault("processing_status", "needs_review" if not extracted.strip() else "processed")
        # Persist structured evidence only; the uploaded source is temporary.
        doc = MedicalDocument(
            user_id=user.id,
            original_name=file.filename or path.name,
            stored_path="",
            mime_type=file.content_type or "",
            document_type=dtype,
            extracted_text="",
            analysis_json=json.dumps(analysis, default=str),
        )
        session.add(doc)
        session.flush()
        session.add(MedicalRecord(
            user_id=user.id,
            record_type=dtype,
            title=doc.original_name,
            content=json.dumps(analysis, default=str),
            source_document_id=doc.id,
        ))
        session.commit()
        session.refresh(doc)
        return {"id": doc.id, "filename": doc.original_name, "document_type": dtype, "analysis": analysis}
    except HTTPException:
        session.rollback()
        raise
    except Exception as exc:
        session.rollback()
        raise HTTPException(422, f"Could not process or save the uploaded document: {str(exc)[:250]}")
    finally:
        path.unlink(missing_ok=True)


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
    # Capture existing history before saving the current message.
    session_id = req.session_id
    previous_chat = chat_context_for(user, session, session_id)
    medical_history = history_for(user, session, session_id)

    session.add(ChatMessage(user_id=user.id, role="user", content=req.message, session_id=session_id))
    session.commit()

    intake = req.model_dump()
    intake["free_text"] = req.message
    intake["conversation_context"] = previous_chat
    intake["medical_history"] = medical_history
    intake["session_id"] = session_id

    # Do not replace a structured symptom list supplied by the client.
    # The orchestrator's enrichment layer will extract/merge symptoms from
    # both the current message and previous patient messages.
    result = route_with_agents(user, intake, session, session_id)
    # Save the routing/agent trace for longitudinal audit, but do not run tier engines here.
    session.add(Prediction(user_id=user.id, request_json=json.dumps(intake, default=str),
                           result_json=json.dumps({"routing": result, "tier_results": {}}, default=str)))
    reply = {"message": result.get("message", "Orchestration completed through tier routing."),
             "result": {"routing": result, "tier_results": {}, "agent_trace": result.get("agent_trace", []),
                        "medical_history_used": result.get("medical_history_used", False),
                        "evidence_summary": result.get("evidence_summary", {})}}
    session.add(ChatMessage(user_id=user.id, role="assistant", content=json.dumps(reply, default=str), session_id=session_id))

    session.commit()

    return reply


@app.post("/api/guidance/chat")
def guidance_chat(payload: dict, user: User = Depends(current_user)):
    prediction = payload.get("prediction")
    messages = payload.get("messages")
    if not isinstance(prediction, dict) or not prediction:
        raise HTTPException(400, "A completed prediction result is required")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(400, "At least one chat message is required")
    try:
        return answer_prediction_question(prediction, str(payload.get("symptoms", ""))[:4000], messages)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        raise HTTPException(502, f"Groq follow-up failed: {str(exc)[:250]}")

@app.post("/api/guidance")
def prevention_guidance(payload: dict, user: User = Depends(current_user)):
    """Generate patient-friendly prevention guidance from completed model output."""
    prediction = payload.get("prediction")
    if not isinstance(prediction, dict) or not prediction:
        raise HTTPException(400, "A completed prediction result is required")
    try:
        return generate_guidance(
            prediction=prediction,
            symptoms=str(payload.get("symptoms", ""))[:4000],
            relevant_history=str(payload.get("relevant_history", ""))[:4000],
        )
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        raise HTTPException(502, f"Guidance generation failed: {str(exc)[:250]}")

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

@app.get("/api/tier1/schema/{disease}")
def tier1_schema(disease: str, user: User = Depends(current_user)):
    try:
        return engines()[1].schema(disease)
    except Exception as exc:
        raise HTTPException(422, f"Could not load Tier 1 schema: {str(exc)[:250]}") from exc

@app.get("/api/tier2/evidence")
def tier2_evidence(user: User = Depends(current_user)):
    try:
        return {"evidence": engines()[2].evidence_catalog()}
    except Exception as exc:
        raise HTTPException(503, str(exc))

@app.get("/api/medical-disclaimer")
def medical_disclaimer():
    return {"text": "Research/demo decision-support only. It does not diagnose disease or replace a qualified clinician."}
