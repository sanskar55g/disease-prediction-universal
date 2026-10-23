import unittest

from app.prediction_inputs import build_prediction_engine_input


class PredictionInputTests(unittest.TestCase):
    def test_tier1_fields_are_nested_for_engine(self):
        result = build_prediction_engine_input(
            1,
            {"tier1": {"disease": "heart_disease"}, "structured": {"age": 55}},
            {"disease": "heart_disease", "features": {"cholesterol": 220}},
            [],
            [],
        )
        self.assertEqual(result["tier1"]["disease"], "heart_disease")
        self.assertEqual(result["tier1"]["features"], {"cholesterol": 220})

    def test_tier2_evidence_codes_are_nested_for_engine(self):
        result = build_prediction_engine_input(
            2,
            {"tier2": {}, "age": 35},
            {"evidence_codes": ["E_55", "E_91_@_V_12"]},
            ["fever"],
            [],
        )
        self.assertEqual(
            result["tier2"]["evidence_codes"], ["E_55", "E_91_@_V_12"]
        )

    def test_tier3_receives_enriched_symptoms_even_if_not_routed(self):
        result = build_prediction_engine_input(
            3, {"symptoms": [], "free_text": "fever"}, {}, ["fever", "cough"], []
        )
        self.assertEqual(result["symptoms"], ["fever", "cough"])

    def test_tier4_receives_enriched_symptoms_and_hpo_terms(self):
        result = build_prediction_engine_input(
            4, {"symptoms": [], "hpo_terms": []}, {}, ["weakness"], ["HP:0001250"]
        )
        self.assertEqual(result["symptoms"], ["weakness"])
        self.assertEqual(result["hpo_terms"], ["HP:0001250"])


if __name__ == "__main__":
    unittest.main()
