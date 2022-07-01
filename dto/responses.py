from pydantic import BaseModel


class TumorDetectionResponse(BaseModel):
    prediction: bool
    confidence: float


class AlzheimerDetectionResponse(BaseModel):
    prediction: int
