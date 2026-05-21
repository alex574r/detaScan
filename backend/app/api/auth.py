"""
DentaScan — Router de Autenticación.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.user import UserCreate, UserLogin, Token, UserResponse
from app.services.auth_service import auth_service
from app.models.user import User
from app.utils.security import get_current_active_user, create_access_token

router = APIRouter(prefix="/auth", tags=["Autenticación"])


@router.post("/register", response_model=Token, status_code=201)
def register(data: UserCreate, db: Session = Depends(get_db)):
    """Registra un nuevo usuario y retorna token JWT."""
    return auth_service.register(db, data)


@router.post("/login", response_model=Token)
def login(data: UserLogin, db: Session = Depends(get_db)):
    """Autentica con email y contraseña. Retorna token JWT."""
    return auth_service.login(db, data)


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_active_user)):
    """Retorna el perfil del usuario autenticado."""
    return current_user


@router.post("/refresh", response_model=Token)
def refresh_token(current_user: User = Depends(get_current_active_user)):
    """
    Renueva el access token. Requiere un token válido (aunque cercano a expirar).
    Devuelve un token nuevo con TTL completo y los datos actualizados del usuario.
    """
    new_token = create_access_token(current_user.id)
    return Token(access_token=new_token, user=current_user)
