"""
DentaScan — Servicio de Autenticación.
Maneja registro, login y consulta de usuario.
"""

from __future__ import annotations

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.user import User
from app.schemas.user import UserCreate, UserLogin, Token, UserResponse
from app.utils.security import hash_password, verify_password, create_access_token
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AuthService:
    def register(self, db: Session, data: UserCreate) -> Token:
        if db.query(User).filter(User.email == data.email).first():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"El email '{data.email}' ya está registrado.",
            )

        user = User(
            email=data.email,
            full_name=data.full_name,
            hashed_password=hash_password(data.password),
            role=data.role,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info("Nuevo usuario registrado: %s (rol=%s)", user.email, user.role)

        token = create_access_token(user.id)
        return Token(access_token=token, user=UserResponse.model_validate(user))

    def login(self, db: Session, data: UserLogin) -> Token:
        user = db.query(User).filter(User.email == data.email).first()
        if not user or not verify_password(data.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Credenciales incorrectas.",
            )
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cuenta desactivada.",
            )

        logger.info("Login exitoso: %s", user.email)
        token = create_access_token(user.id)
        return Token(access_token=token, user=UserResponse.model_validate(user))

    def get_user(self, db: Session, user_id: int) -> User:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado.")
        return user


auth_service = AuthService()
