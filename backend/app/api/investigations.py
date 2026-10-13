import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models.user import User
from app.models.investigation import Investigation
from app.models.workflow import WorkflowStatusLog
from app.schemas.investigation import InvestigationCreate, InvestigationResponse


router = APIRouter()


@router.post("/", response_model=InvestigationResponse)
def create_investigation(
    payload: InvestigationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    investigation = Investigation(
        id=str(uuid.uuid4()),
        organisation_id=current_user.organisation_id,
        user_id=current_user.id,
        name=payload.name,
        notes=payload.notes,
        status="UPLOADED"
    )

    workflow = WorkflowStatusLog(
        id=str(uuid.uuid4()),
        investigation_id=investigation.id,
        organisation_id=current_user.organisation_id,
        current_state="UPLOADED",
        stages=[]
    )

    db.add(investigation)
    db.add(workflow)
    db.commit()
    db.refresh(investigation)

    return investigation


@router.get("/", response_model=list[InvestigationResponse])
def list_investigations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return (
        db.query(Investigation)
        .filter(Investigation.organisation_id == current_user.organisation_id)
        .all()
    )


@router.get("/{investigation_id}", response_model=InvestigationResponse)
def get_investigation(
    investigation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    investigation = (
        db.query(Investigation)
        .filter(
            Investigation.id == investigation_id,
            Investigation.organisation_id == current_user.organisation_id
        )
        .first()
    )

    if not investigation:
        raise HTTPException(status_code=404, detail="Investigation not found")

    return investigation