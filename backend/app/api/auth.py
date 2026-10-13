import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import hash_password, verify_password, create_access_token
from app.models.user import User
from app.models.organisation import Organisation
from app.schemas.auth import RegisterRequest, LoginRequest


router = APIRouter()


@router.post("/register")
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == payload.email).first()

    if existing_user:
        raise HTTPException(status_code=409, detail="Email already registered")

    organisation = Organisation(
        id=str(uuid.uuid4()),
        name=f"{payload.email}'s Organisation",
        subscription_tier="Starter"
    )

    user = User(
        id=str(uuid.uuid4()),
        organisation_id=organisation.id,
        email=payload.email,
        password_hash=hash_password(payload.password),
        role="Administrator",
        email_verified=True
    )

    db.add(organisation)
    db.add(user)
    db.commit()

    return {
        "message": "User registered successfully",
        "user_id": user.id,
        "organisation_id": organisation.id
    }


@router.post("/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()

    if not user:
        raise HTTPException(status_code=401, detail="Authentication failed")

    if not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Authentication failed")

    token = create_access_token(
        data={
            "sub": user.id,
            "organisation_id": user.organisation_id,
            "role": user.role
        }
    )

    return {
        "access_token": token,
        "token_type": "bearer"
    }