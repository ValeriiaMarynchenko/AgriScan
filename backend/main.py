from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from jose import jwt, JWTError
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timedelta
from bson import ObjectId
from security import get_password_hash, ACCESS_TOKEN_EXPIRE_MINUTES, create_access_token, verify_password
from dependencies import connect_to_mongo, close_mongo_connection, get_database
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr, Field
from settings import settings

import uvicorn
import settings


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")
app = FastAPI(title="AgriScan Backend", on_startup=[connect_to_mongo], on_shutdown=[close_mongo_connection])

# CORS-requests
origins = ["http://localhost:5173", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# --- MODELS ---
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_name: str
# --- АВТЕНТИФІКАЦІЯ ---

@app.post("/api/auth/register", status_code=status.HTTP_201_CREATED, tags=["Auth"])
async def register_user(user_data: UserCreate, db: AsyncIOMotorClient = Depends(get_database)):
    """
    Реєстрація нового користувача.
    1. Перевіряє, чи існує email.
    2. Хешує пароль.
    3. Зберігає в БД.
    """
    # Перевірка на існування користувача
    existing_user = await db["users"].find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Хешування пароля
    hashed_pw = get_password_hash(user_data.password)

    # Підготовка документу
    new_user = {
        "email": user_data.email,
        "hashed_password": hashed_pw,
        "full_name": user_data.full_name,
        "created_at": datetime.utcnow()
    }

    # Збереження
    result = await db["users"].insert_one(new_user)

    return {"message": "User created successfully", "id": str(result.inserted_id)}


@app.post("/auth/jwt/create/", response_model=Token, tags=["Auth"])
async def login_for_access_token(form_data: UserLogin, db: AsyncIOMotorClient = Depends(get_database)):
    """
    Вхід в систему.
    Приймає JSON {email, password}.
    Повертає JWT Access Token.
    """
    # Пошук користувача
    user = await db["users"].find_one({"email": form_data.email})

    # Перевірка пароля
    if not user or not verify_password(form_data.password, user["hashed_password"]):
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

    return {
        "access": "...",
        "refresh": "..." ,
        "token_type": "bearer",
        "user_id": str(user["_id"]),
        "full_name": user.get("full_name", "")
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

    user = await db["users"].find_one({"email": email})
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
        field_id: str = Form(...),  # Отримуємо ID поля як текст з форми
        file: UploadFile = File(...),  # Отримуємо файл
        current_user: dict = Depends(get_current_user),  # Тільки для авторизованих!
        db: AsyncIOMotorClient = Depends(get_database)
):
    """
    Завантаження зображення поля.
    Використовує multipart/form-data.
    """
    # 1. Валідація типу файлу
    if file.content_type not in ["image/jpeg", "image/png", "image/tiff"]:
        raise HTTPException(400, detail="Invalid file type. Only JPEG, PNG, TIFF allowed.")

    # 2. Валідація розміру (опціонально, треба читати чанками для великих файлів)
    # file_content = await file.read()

    # 3. Збереження файлу
    # В реальному проекті: завантажити на AWS S3 / Google Cloud Storage
    # Для MVP: зберегти в папку 'static/uploads'

    import os
    upload_dir = "static/uploads"
    os.makedirs(upload_dir, exist_ok=True)

    file_location = f"{upload_dir}/{field_id}_{file.filename}"

    with open(file_location, "wb+") as file_object:
        file_object.write(await file.read())

    # 4. Оновлення запису в БД (додаємо шлях до файлу)
    await db["fields"].update_one(
        {"_id": ObjectId(field_id)},
        {"$set": {"image_path": file_location, "updated_at": datetime.utcnow()}}
    )

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
        "path": file_location
    }


@app.get("/health", tags=["Health"])
async def health_check():
    await connect_to_mongo()
    await close_mongo_connection()
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)

# --- ЗАПУСК ---
# Зазвичай запускається через Uvicorn, наприклад:
# uvicorn main:app --reload --host 0.0.0.0 --port 8000