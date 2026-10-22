# Evidence Agent setup and verification

## What this PR does

- Accepts PDF, JPEG, PNG and WEBP uploads through the authenticated `POST /api/documents` endpoint.
- Extracts embedded PDF text first. For scanned PDFs it renders pages and attempts Tesseract OCR; images use Tesseract OCR.
- Produces a structured JSON analysis with summary, conditions, medications, labs, symptoms, dates, important history, extraction method and processing status.
- Persists that JSON in `medical_documents.analysis_json` and creates a linked `medical_records` row. The uploaded source file is temporary and is deleted after processing; the structured JSON remains available for future sessions.
- The History Agent reads saved records and document JSON in later chat/routing sessions. The Evidence Agent reports document processing status and exposes structured findings in the live trace.
- Chat and routing stop after the Tier Router. Tier prediction engines are not executed by `/api/chat`; explicit `POST /api/predict` remains the separate prediction endpoint.

## Required OCR runtime

Install Python dependencies from `requirements.txt`. Tesseract itself is a system executable and must also be installed:

- Windows: install the Tesseract OCR Windows build and add its install directory to `PATH`.
- Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y tesseract-ocr`
- macOS: `brew install tesseract`

Restart the backend after installation. If Tesseract or PyMuPDF is missing, readable PDFs may still work, but image-only uploads and scanned PDFs can return `needs_review`. Do not interpret that status as successful image understanding.

## Safety and limits

OCR reads text; it does **not** interpret radiology pixels or diagnose from X-ray/CT/MRI images. Such images are marked for review unless readable text is available. The offline fallback captures literal medication-like lines and possible lab/value pairs; these are candidates requiring source verification, not medical advice. A configured Groq-compatible key enables the structured extraction prompt, which is still subject to review.

## Quick verification

1. Start the backend and authenticate.
2. Upload a text PDF, a scanned PDF, and a legible prescription image using `POST /api/documents`.
3. Inspect each response's `analysis.processing_status`, `analysis.extraction_method`, and structured JSON fields.
4. Check `GET /api/history`: each upload should appear under `documents` and its structured JSON under the linked `records`.
5. Start a new chat session for the same user. The History Agent's structured findings should include persisted document facts, and the response should include exactly five entries in `result.agent_trace`.
6. Confirm `result.tier_results` is empty for chat/routing. No disease-prediction model is run until the explicit prediction endpoint is called.
