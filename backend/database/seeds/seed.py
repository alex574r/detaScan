"""
DentaScan — Script de seed de base de datos.
Crea usuarios de prueba para desarrollo y demostración.
Ejecutar: python database/seeds/seed.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.database import SessionLocal, init_db
from app.models.user import User, UserRole
from app.utils.security import hash_password


SEED_USERS = [
    {
        "email": "admin@dentascan.mx",
        "full_name": "Administrador DentaScan",
        "password": "Admin1234!",
        "role": UserRole.ADMIN,
    },
    {
        "email": "odonto@dentascan.mx",
        "full_name": "Dra. Rocio Pulido",
        "password": "Dentista1234!",
        "role": UserRole.ODONTOLOGIST,
    },
    {
        "email": "estudiante@dentascan.mx",
        "full_name": "Francisco Martínez",
        "password": "Estudiante1234!",
        "role": UserRole.STUDENT,
    },
]


def run_seed():
    print("Inicializando base de datos...")
    init_db()

    db = SessionLocal()
    try:
        for user_data in SEED_USERS:
            existing = db.query(User).filter(User.email == user_data["email"]).first()
            if existing:
                print(f"  [SKIP] {user_data['email']} ya existe.")
                continue

            user = User(
                email=user_data["email"],
                full_name=user_data["full_name"],
                hashed_password=hash_password(user_data["password"]),
                role=user_data["role"],
            )
            db.add(user)
            print(f"  [OK]   {user_data['email']} ({user_data['role'].value}) creado.")

        db.commit()
        print("\nSeed completado.")
        print("\nCredenciales de prueba:")
        for u in SEED_USERS:
            print(f"  Email: {u['email']}  |  Password: {u['password']}")

    finally:
        db.close()


if __name__ == "__main__":
    run_seed()
