from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from jose import jwt, JWTError
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import timedelta
from bson import ObjectId
from security import get_password_hash, ACCESS_TOKEN_EXPIRE_MINUTES, create_access_token, verify_password
from dependencies import connect_to_mongo, close_mongo_connection, get_database
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr, Field
from settings import settings
from datetime import datetime
from minio import Minio, error as minio_errors
import uvicorn
from io import BytesIO

# --- ІНІЦІАЛІЗАЦІЯ ---

# Виправлено: tokenUrl має відповідати роуту входу
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/jwt/create/")
app = FastAPI(title="AgriScan Backend", on_startup=[connect_to_mongo], on_shutdown=[close_mongo_connection])

# CORS-requests
origins = ["http://localhost:5173", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# Клієнт MinIO
MINIO_CLIENT = Minio(
    settings.AWS_S3_ENDPOINT_URL,
    access_key=settings.AWS_ACCESS_KEY_ID,
    secret_key=settings.AWS_SECRET_ACCESS_KEY,
    secure=False
)
MINIO_BUCKET = settings.AWS_STORAGE_BUCKET_NAME

# --- MODELS ---
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    re_password: str = Field(..., min_length=6)
    full_name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_name: str


# --- АВТЕНТИФІКАЦІЯ ---

@app.post("/api/v1/auth/users/", status_code=status.HTTP_201_CREATED, tags=["Auth"])
async def register_user(user_data: UserCreate, db: AsyncIOMotorClient = Depends(get_database)):
    """
    Реєстрація нового користувача.
    """
    # 1. Перевірка на існування користувача
    existing_user = await db["users"].find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # 2. Хешування пароля
    hashed_pw = get_password_hash(user_data.password)

    # 3. Підготовка документу
    new_user = {
        "email": user_data.email,
        "hashed_password": hashed_pw,
        "created_at": datetime.utcnow(),
        "type": "Viewer", # Встановлюємо значення за замовчуванням з моделі
        "name": user_data.full_name, # Просте ім'я за замовчуванням
    }

    # 4. Збереження
    result = await db["users"].insert_one(new_user) # <-- ДОДАНО await

    return {"message": "User created successfully", "id": str(result.inserted_id)}


@app.post("/auth/jwt/create/", tags=["Auth"])
async def login_for_access_token(form_data: UserLogin, db: AsyncIOMotorClient = Depends(get_database)):
    """
    Вхід в систему.
    Приймає JSON {email, password}.
    Повертає JWT Access Token.
    """
    # Пошук користувача
    # user = await db["users"].find_one({"email": form_data.email})
    user = await db.users.find_one({"email": form_data.email},{"hashed_password": 1, "name": 1, "_id": 1, "email": 1})

    hashed_password = user.get("hashed_password") if user else None
    # Перевірка пароля
    if not user or not verify_password(form_data.password, hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Генерація токена
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"], "user_id": str(user["_id"])},
        expires_delta=access_token_expires
    )

    # У вас тут дві різні структури, повертаємо повну:
    return {
        "access_token": access_token, # Повертаємо згенерований токен
        "token_type": "bearer",
        "user_name": user.get("name", ""),
        "user_id": str(user["_id"])
    }


# --- ЗАХИЩЕНИЙ РОУТ (приклад) ---
async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncIOMotorClient = Depends(get_database)):
    """Допоміжна функція для отримання юзера з токена"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = await db["users"].find_one({"email": email}) # <-- ДОДАНО await
    if user is None:
        raise credentials_exception
    return user


@app.get("/auth/users/me/", tags=["Users"])
async def read_users_me(current_user: dict = Depends(get_current_user)):
    """Повертає дані поточного залогіненого користувача"""
    # Конвертуємо ObjectId в стрічку для JSON
    current_user["_id"] = str(current_user["_id"])
    # Видаляємо пароль з відповіді
    current_user.pop("hashed_password", None)
    return current_user


# --- ОТРИМАННЯ ФАЙЛУ (UPLOAD) ---
@app.post("/api/fields/upload", tags=["Fields"])
async def upload_field_image(
        field_id: str = Form(...),
        file: UploadFile = File(...),
        current_user: dict = Depends(get_current_user),
        db: AsyncIOMotorClient = Depends(get_database)
):
    """
    Завантаження зображення поля в MinIO.
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/tiff"]:
        raise HTTPException(400, detail="Invalid file type. Only JPEG, PNG, TIFF allowed.")

    file_content = await file.read()
    file_size = len(file_content)

    # Визначення шляху (об'єктного ключа) у MinIO
    object_name = f"fields/{field_id}/{file.filename}"

    try:
        MINIO_CLIENT.put_object(
            bucket_name=MINIO_BUCKET,
            object_name=object_name,
            data=BytesIO(file_content),
            length=file_size,
            content_type=file.content_type
        )
    except minio_errors.S3Error as e:
        print(f"MinIO Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload file to storage."
        )

    # 4. Оновлення запису в БД (зберігаємо шлях до об'єкта)
    await db["fields"].update_one( # <-- ДОДАНО await
        {"_id": ObjectId(field_id)},
        {"$set": {"image_key": object_name, "updated_at": datetime.utcnow()}}
    )

    return {
        "message": "File uploaded successfully to MinIO",
        "object_key": object_name,
        "filename": file.filename
    }

@app.get("/health", tags=["Health"])
async def health_check():
    # Просто повертаємо статус без перепідключення, оскільки підключення керується on_startup/on_shutdown
    return {"status": "ok"}

if __name__ == "__main__":
    # ВИДАЛЕНО: Синхронний код MongoEngine тут не працюватиме коректно з Motor/FastAPI.
    # Uvicorn сам ініціалізує додаток.
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
# --- ЗАПУСК ---
# uvicorn main:app --reload --host 0.0.0.0 --port 8000