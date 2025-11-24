from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext
from settings import settings
import bcrypt
from fastapi import HTTPException, status

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# SALT_ROUNDS = 12


def get_password_hash(password: str) -> str:
    try:
        pwd_bytes = password.encode("utf-8")[:72]
        hashed = bcrypt.hashpw(pwd_bytes, bcrypt.gensalt())
        return hashed.decode()  # <-- ОБОВ’ЯЗКОВО
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))



def verify_password(password: str, password_hash: str) -> bool:
    """Перевіряє, чи відповідає наданий пароль збереженому хешу."""
    pwd = password.encode("utf-8")[:72]
    return bcrypt.checkpw(pwd, password_hash.encode())


def create_access_token(data: dict, expires_minutes: int = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

def decode_token(token: str):
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")