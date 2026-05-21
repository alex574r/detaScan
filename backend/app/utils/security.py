"""
DentaScan — Utilidades de seguridad.
Manejo de JWT, hashing de contraseñas y dependencias de autenticación.
"""

from datetime import datetime, timedelta, timezone
from typing import Annotated

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models.user import User
from app.schemas.user import TokenData

settings = get_settings()
security = HTTPBearer()


# ─── Contraseñas ──────────────────────────────────────────────────────────────

def hash_password(plain_password: str) -> str:
    """Genera un hash bcrypt de la contraseña."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Compara una contraseña en texto plano con su hash bcrypt."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )


# ─── JWT ──────────────────────────────────────────────────────────────────────

def create_access_token(user_id: int) -> str:
    """Genera un JWT firmado con el ID de usuario como sujeto."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": str(user_id),
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_access_token(token: str) -> TokenData:
    """Decodifica un JWT y retorna el payload. Lanza HTTPException si es inválido."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id = int(payload.get("sub", 0))
        if not user_id:
            raise JWTError("sub vacío")
        return TokenData(user_id=user_id)
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token inválido o expirado: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


# ─── Dependencias FastAPI ─────────────────────────────────────────────────────

def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    """Dependencia que extrae y valida el usuario desde el JWT del header."""
    token_data = decode_access_token(credentials.credentials)
    user = db.query(User).filter(User.id == token_data.user_id).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Usuario no encontrado")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Usuario desactivado")
    return user


def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    return current_user
