from pydantic import BaseModel
from typing import List, Optional

# --- Image Schemas ---

class ImageBase(BaseModel):
    original_filename: str
    annotated_image_url: Optional[str] = None
    analysis_workflow: Optional[str] = None
    scores: Optional[List[float]] = []
    boxes_xywhn: Optional[List[List[float]]] = []
    summary: Optional[str] = None

class ImageCreate(ImageBase):
    pass

class Image(ImageBase):
    id: int
    patient_id: int

    class Config:
        from_attributes = True # orm_mode = True for pydantic v1


# --- Patient Schemas ---

class PatientBase(BaseModel):
    name: str

class PatientCreate(PatientBase):
    pass

class Patient(PatientBase):
    id: int
    images: List[Image] = []

    class Config:
        from_attributes = True # orm_mode = True for pydantic v1
