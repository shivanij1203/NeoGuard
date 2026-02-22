from datetime import datetime
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from db.database import get_db
from db.models import PainScore

router = APIRouter(prefix="/api/scores", tags=["scores"])


class PainScoreResponse(BaseModel):
    id: int
    patient_id: int
    timestamp: datetime
    facial_score: float | None
    audio_score: float | None
    composite_score: float
    brow_furrow: float | None
    eye_squeeze: float | None
    nasolabial_furrow: float | None
    mouth_stretch: float | None
    cry_detected: bool
    cry_type: str | None
    alert_level: str | None

    class Config:
        from_attributes = True


@router.get("/{patient_id}", response_model=list[PainScoreResponse])
async def get_scores(
    patient_id: int,
    limit: int = Query(default=100, le=1000),
    since: datetime | None = None,
    db: AsyncSession = Depends(get_db),
):
    query = (
        select(PainScore)
        .where(PainScore.patient_id == patient_id)
        .order_by(desc(PainScore.timestamp))
        .limit(limit)
    )
    if since:
        query = query.where(PainScore.timestamp >= since)
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/{patient_id}/latest", response_model=PainScoreResponse | None)
async def get_latest_score(patient_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(PainScore)
        .where(PainScore.patient_id == patient_id)
        .order_by(desc(PainScore.timestamp))
        .limit(1)
    )
    return result.scalar_one_or_none()
