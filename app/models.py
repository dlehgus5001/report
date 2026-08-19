from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ChangeType(str, Enum):
    NEW = "new"
    MISSING = "missing"
    MOVED = "moved"
    MODIFIED = "modified"


class Box(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)


class Detection(BaseModel):
    id: str
    image: Literal["before", "after"]
    label: str
    confidence: float = Field(ge=0, le=1)
    box: Box


class Change(BaseModel):
    id: str
    type: ChangeType
    label: str
    confidence: float = Field(ge=0, le=1)
    description: str
    before_box: Box | None = None
    after_box: Box | None = None


class Registration(BaseModel):
    method: str
    score: float = Field(ge=0, le=1)
    transform: list[list[float]]


class AnalysisResult(BaseModel):
    analysis_id: str
    status: Literal["completed"] = "completed"
    created_at: str
    inputs: dict[str, str]
    registration: Registration
    detections: list[Detection]
    changes: list[Change]
    summary: dict[str, int]
    report: str
    pipeline_mode: Literal["demo", "model"] = "demo"

