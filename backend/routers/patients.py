from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from db.database import get_db
from db.models import Patient

router = APIRouter(prefix="/api/patients", tags=["patients"])


class PatientCreate(BaseModel):
    name: str
    bed_number: str
    gestational_age_weeks: int | None = None
    birth_weight_grams: int | None = None
    notes: str | None = None


class PatientResponse(BaseModel):
    id: int
    name: str
    bed_number: str
    gestational_age_weeks: int | None
    birth_weight_grams: int | None
    admitted_at: datetime
    is_active: bool
    notes: str | None

    class Config:
        from_attributes = True


@router.get("/", response_model=list[PatientResponse])
async def list_patients(active_only: bool = True, db: AsyncSession = Depends(get_db)):
    query = select(Patient)
    if active_only:
        query = query.where(Patient.is_active == True)
    result = await db.execute(query)
    return result.scalars().all()


@router.post("/", response_model=PatientResponse, status_code=201)
async def create_patient(patient: PatientCreate, db: AsyncSession = Depends(get_db)):
    db_patient = Patient(**patient.model_dump())
    db.add(db_patient)
    await db.commit()
    await db.refresh(db_patient)
    return db_patient


@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(patient_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Patient).where(Patient.id == patient_id))
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.put("/{patient_id}", response_model=PatientResponse)
async def update_patient(patient_id: int, data: PatientCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Patient).where(Patient.id == patient_id))
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    for key, value in data.model_dump().items():
        setattr(patient, key, value)
    await db.commit()
    await db.refresh(patient)
    return patient


@router.delete("/{patient_id}", status_code=204)
async def delete_patient(patient_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Patient).where(Patient.id == patient_id))
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    patient.is_active = False
    await db.commit()
