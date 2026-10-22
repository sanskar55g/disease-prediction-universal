import json
import unittest
from types import SimpleNamespace

from app.orchestrator.agents import collect_saved_evidence, run_agent_pipeline


class AgentPipelineTests(unittest.TestCase):
    def test_saved_document_json_is_read_for_history_and_evidence(self):
        analysis = {
            "processing_status": "processed",
            "summary": "Prescription text",
            "medications": [{"text": "metformin 500 mg"}],
            "labs": [{"name": "HbA1c", "value": "7.2", "unit": "%"}],
            "conditions": [],
            "symptoms": [],
        }
        doc = SimpleNamespace(
            id=7, original_name="report.pdf", document_type="prescription",
            created_at="2026-10-22", analysis_json=json.dumps(analysis)
        )
        record = SimpleNamespace(
            id=9, title="report.pdf", record_type="prescription",
            content=json.dumps(analysis), source_document_id=7, created_at="2026-10-22"
        )
        evidence = collect_saved_evidence([doc], [record])
        _, trace = run_agent_pipeline({"message": "I feel tired", "symptoms": ["fatigue"]}, evidence,
                                      {"primary_tier": 1, "secondary_tiers": [], "scores": {"1": 0.6},
                                       "reasons": {"1": ["structured data"]}, "emergency": False})
        self.assertEqual(len(trace), 5)
        self.assertEqual(trace[0]["findings"]["saved_document_count"], 1)
        self.assertEqual(len(trace[0]["findings"]["medications"]), 1)
        self.assertEqual(len(trace[0]["findings"]["labs"]), 1)
        self.assertEqual(trace[2]["findings"]["documents_reviewed"][0]["filename"], "report.pdf")
        self.assertEqual(trace[4]["findings"]["primary_tier"], 1)

    def test_unreadable_evidence_is_flagged_for_review(self):
        doc = SimpleNamespace(id=1, original_name="scan.png", document_type="scan",
                              created_at="2026-10-22",
                              analysis_json=json.dumps({"processing_status": "needs_review", "summary": "No readable text"}))
        evidence = collect_saved_evidence([doc], [])
        _, trace = run_agent_pipeline({"message": "headache"}, evidence, {"primary_tier": 2, "emergency": False})
        self.assertEqual(trace[2]["status"], "needs_review")
        self.assertIn("not interpreted", trace[2]["findings"]["processing_note"])


if __name__ == "__main__":
    unittest.main()
