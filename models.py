from sqlalchemy import Column, Integer, String, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship

from .database import Base


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, unique=True)

    images = relationship("Image", back_populates="patient")


class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    original_filename = Column(String)
    annotated_image_url = Column(String, nullable=True)
    analysis_workflow = Column(String, nullable=True)
    scores = Column(JSON, nullable=True)
    boxes_xywhn = Column(JSON, nullable=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))

    patient = relationship("Patient", back_populates="images")