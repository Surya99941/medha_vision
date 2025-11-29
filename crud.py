from sqlalchemy.orm import Session, selectinload
from sqlalchemy.future import select
from . import models, schemas

# --- Patient CRUD ---

def get_patient(db: Session, patient_id: int):
    """
    Fetches a single patient and eagerly loads their images.
    """
    return db.query(models.Patient).options(selectinload(models.Patient.images)).filter(models.Patient.id == patient_id).first()

def get_patient_by_name(db: Session, name: str):
    return db.query(models.Patient).filter(models.Patient.name == name).first()

def get_patients(db: Session, skip: int = 0, limit: int = 100):
    """
    Fetches a list of patients and eagerly loads their images.
    """
    return db.query(models.Patient).options(selectinload(models.Patient.images)).offset(skip).limit(limit).all()

def create_patient(db: Session, patient: schemas.PatientCreate):
    db_patient = models.Patient(name=patient.name)
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient

# --- Image CRUD ---

def create_patient_image(db: Session, image: schemas.ImageCreate, patient_id: int):
    db_image = models.Image(**image.model_dump(), patient_id=patient_id)
    db.add(db_image)
    db.commit()
    db.refresh(db_image)
    return db_image

def get_image(db: Session, image_id: int):
    return db.query(models.Image).filter(models.Image.id == image_id).first()

def update_image_summary(db: Session, image_id: int, summary: str):
    db_image = get_image(db, image_id)
    if db_image:
        db_image.summary = summary
        db.commit()
        db.refresh(db_image)
    return db_image

