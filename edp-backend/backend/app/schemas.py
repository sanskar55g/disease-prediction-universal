from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    email: str
    password: str = Field(min_length=8)
    name: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    structured: dict[str, Any] = {}
    symptoms: list[str] = []
    symptom_duration_days: float | None = None
    onset: str | None = None
    fever: bool | None = None
    age: int | None = None
    sex: str | None = None
    tier1: dict[str, Any] = {}
    tier2: dict[str, Any] = {}
    hpo_terms: list[str] = []
    genomic_data: Any = None
    screening_requested: bool = False


class PredictionRequest(BaseModel):
    symptoms: list[str] = []
    free_text: str = ""
    structured: dict[str, Any] = {}
    symptom_duration_days: float | None = None
    onset: str | None = None
    fever: bool | None = None
    age: int | None = None
    sex: str | None = None
    tier1: dict[str, Any] = {}
    tier2: dict[str, Any] = {}
    hpo_terms: list[str] = []
    genomic_data: Any = None
    screening_requested: bool = False
