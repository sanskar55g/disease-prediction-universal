
from __future__ import annotations

import re
from typing import Any


def answers_to_evidence_codes(
    evidence_catalog: list[dict[str, Any]],
    answers: dict[str, Any],
) -> list[str]:
    """Convert binary, categorical, and multiple-value answers to DDXPlus tokens."""

    if not isinstance(answers, dict):
        raise ValueError("answers must be an object")

    catalog = {
        item["code"]: item
        for item in evidence_catalog
        if isinstance(item, dict) and item.get("code")
    }

    tokens: list[str] = []

    for code, answer in answers.items():
        if code not in catalog:
            raise ValueError(f"Unknown evidence code: {code}")

        if answer is None or answer == "":
            continue

        item = catalog[code]
        question_type = item.get("type")
        values = item.get("values") or {}

        if question_type == "B":
            if isinstance(answer, bool):
                affirmative = answer
            elif isinstance(answer, str):
                normalized = answer.strip().lower()
                if normalized in {"yes", "true"}:
                    affirmative = True
                elif normalized in {"no", "false", "unknown", "not sure"}:
                    affirmative = False
                else:
                    raise ValueError(
                        f"{code} requires a yes/no answer"
                    )
            else:
                raise ValueError(f"{code} requires a boolean answer")

            if affirmative:
                tokens.append(code)

        elif question_type in {"C", "M"}:
            if question_type == "C":
                if not isinstance(answer, str):
                    raise ValueError(
                        f"{code} requires one value ID string"
                    )
                selected_values = [answer.strip()]

            else:
                if isinstance(answer, str):
                    selected_values = [answer.strip()]
                elif isinstance(answer, list):
                    selected_values = answer
                else:
                    raise ValueError(
                        f"{code} requires a list of selected value IDs"
                    )

            if not selected_values:
                continue

            for value_id in selected_values:
                if not isinstance(value_id, str):
                    raise ValueError(
                        f"Every selected value for {code} must be a string"
                    )

                value_id = value_id.strip()

                if not re.fullmatch(r"V_\d+", value_id):
                    raise ValueError(
                        f"Invalid value ID for {code}: {value_id}"
                    )

                # Validate against the catalog when it exposes value IDs.
                if values and value_id not in values:
                    raise ValueError(
                        f"{value_id} is not a valid value for {code}"
                    )

                tokens.append(f"{code}_@_{value_id}")

        else:
            raise ValueError(
                f"Unsupported question type {question_type!r} for {code}"
            )

    return list(dict.fromkeys(tokens))


def build_tier2_input(
    evidence_codes: list[str],
    age: int | None = None,
    sex: str | None = None,
) -> dict[str, Any]:
    """Build the input contract expected by the Tier 2 engine."""

    if not isinstance(evidence_codes, list):
        raise ValueError("evidence_codes must be a list")

    if not all(
        isinstance(code, str) and code.strip()
        for code in evidence_codes
    ):
        raise ValueError("Evidence codes must be non-empty strings")

    codes = list(dict.fromkeys(code.strip() for code in evidence_codes))

    if not codes:
        raise ValueError(
            "Answer at least one relevant evidence question before predicting."
        )

    patient: dict[str, Any] = {
        "tier2": {"evidence_codes": codes}
    }

    if age is not None:
        if (
            not isinstance(age, int)
            or isinstance(age, bool)
            or not 0 <= age <= 120
        ):
            raise ValueError("age must be an integer between 0 and 120")
        patient["age"] = age

    if sex is not None:
        normalized_sex = sex.strip().upper()
        if normalized_sex not in {"M", "F"}:
            raise ValueError("sex must be M or F")
        patient["sex"] = normalized_sex

    return patient
