from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from db.database import Base


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    bed_number = Column(String, nullable=False)
    gestational_age_weeks = Column(Integer, nullable=True)
    birth_weight_grams = Column(Integer, nullable=True)
    admitted_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    notes = Column(String, nullable=True)

    pain_scores = relationship("PainScore", back_populates="patient", cascade="all, delete-orphan")


class PainScore(Base):
    __tablename__ = "pain_scores"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Individual scores
    facial_score = Column(Float, nullable=True)
    audio_score = Column(Float, nullable=True)
    composite_score = Column(Float, nullable=False)

    # Feature details
    brow_furrow = Column(Float, nullable=True)
    eye_squeeze = Column(Float, nullable=True)
    nasolabial_furrow = Column(Float, nullable=True)
    mouth_stretch = Column(Float, nullable=True)
    cry_detected = Column(Boolean, default=False)
    cry_type = Column(String, nullable=True)  # "pain", "non-pain", "no_cry"

    # Alert info
    alert_level = Column(String, nullable=True)  # "none", "moderate", "severe"

    patient = relationship("Patient", back_populates="pain_scores")
