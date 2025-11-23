from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from settings import settings  # Переконайтесь, що у вас є SECRET_KEY у налаштуваннях
import bcrypt

# Налаштування хешування паролів
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SALT_ROUNDS = 12
# Конфігурація JWT
SECRET_KEY = settings.SECRET_KEY  # Наприклад: "supersecretkey123" (винесіть в .env!)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 години


def get_password_hash(password: str) -> str:
    """Створює BCrypt хеш пароля, обрізаючи його до 72 байтів."""

    # 1. Обробка обмеження BCrypt (72 байти)
    password_bytes = password.encode('utf-8')
    # Обрізаємо пароль до 72 байтів, щоб запобігти ValueError
    safe_password_bytes = password_bytes[:72]

    # 2. Хешування
    hashed_bytes = bcrypt.hashpw(
        password=safe_password_bytes,
        salt=bcrypt.gensalt(rounds=SALT_ROUNDS)
    )

    # 3. Декодування та повернення
    return hashed_bytes.decode('utf-8')


def verify_password(password: str, password_hash: str) -> bool:
    """Перевіряє, чи відповідає наданий пароль збереженому хешу."""
    try:
        # 1. Обробка обмеження BCrypt (72 байти)
        password_bytes = password.encode('utf-8')
        safe_password_bytes = password_bytes[:72]

        # 2. Перевірка
        return bcrypt.checkpw(
            password=safe_password_bytes,  # Використовуємо обрізаний пароль для перевірки
            hashed_password=password_hash.encode('utf-8')
        )
    except Exception:
        # Запобігаємо помилкам, якщо хеш у БД пошкоджений або має неправильний формат
        return False


# =========================================================================
# 2. ЛОГІКА JWT ТОКЕНІВ
# =========================================================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Створює JWT токен доступу."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=30)  # Використовуємо значення за замовчуванням

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt