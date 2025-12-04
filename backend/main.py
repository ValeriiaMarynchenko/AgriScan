from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from jose import jwt, JWTError
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import timedelta
from bson import ObjectId
from security import get_password_hash, create_access_token, verify_password
from dependencies import connect_to_mongo, close_mongo_connection, get_database
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr, Field
from settings import settings
from datetime import datetime
from minio import Minio, error as minio_errors
import uvicorn
from io import BytesIO
from tasks import send_welcome_email
from ai_service import (
    initialize_ai_services,
    CORN_MODEL, POTATO_MODEL, WEED_MODEL,
    predict_corn, predict_potato, predict_weed_segmentation
)
# Імпортуємо класи, щоб вони були доступні для uvicorn
from ai_service.corn.predict_corn_disease import ClassificationCNN as CornCNN
from ai_service.potato.predict_potato_disease import ClassificationCNN as PotatoCNN

# --- ІНІЦІАЛІЗАЦІЯ ---

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/jwt/create/")
app = FastAPI(title="AgriScan_FastAPI_Backend", on_startup=[connect_to_mongo], on_shutdown=[close_mongo_connection])

# CORS-requests
origins = ["http://localhost:5173", "http://localhost:3000", "http://localhost:80", "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
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
    result = await db["users"].insert_one(new_user)
    send_welcome_email.delay(user_data.email)
    return {"message": "User created successfully", "id": str(result.inserted_id)}


@app.post("/auth/jwt/create/", tags=["Auth"])
async def login_for_access_token(form_data: UserLogin, db: AsyncIOMotorClient = Depends(get_database)):
    """
    Вхід в систему.
    Приймає JSON {email, password}.
    Повертає JWT Access Token.
    """
    # user = await db["users"].find_one({"email": form_data.email})
    user = await db.users.find_one({"email": form_data.email},{"hashed_password": 1, "name": 1, "_id": 1, "email": 1})

    hashed_password = user.get("hashed_password") if user else None

    if not user or not verify_password(form_data.password, hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": user["email"], "user_id": str(user["_id"])}, expires_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_name": user.get("name", ""),
        "user_id": str(user["_id"])
    }


# --- ЗАХИЩЕНИЙ РОУТ ---
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
# --- ДОПОМІЖНІ ФУНКЦІЇ ---

def upload_file_to_minio(file_content: bytes, object_name: str, content_type: str):
    """
    Завантажує вміст файлу в MinIO.
    """
    try:
        MINIO_CLIENT.put_object(
            bucket_name=MINIO_BUCKET,
            object_name=object_name,
            data=BytesIO(file_content),
            length=len(file_content),
            content_type=content_type
        )
        return object_name
    except minio_errors.S3Error as e:
        print(f"MinIO Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload file to storage (MinIO)."
        )


# --- НОВІ РОУТИ ДЛЯ AI АНАЛІЗУ ---

@app.post("/api/fields/upload/corn", tags=["AI Prediction"])
async def analyze_corn_disease(
        file: UploadFile = File(..., description="Зображення листя кукурудзи"),
        current_user: dict = Depends(get_current_user),
        db: AsyncIOMotorClient = Depends(get_database)
):
    """
    Класифікація хвороб кукурудзи. Приймає 1 файл зображення.
    """
    if not CORN_MODEL:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Corn model is not initialized.")
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid file type. Only JPEG and PNG allowed.")

    file_content = await file.read()

    # 1. АНАЛІЗ
    try:
        predicted_class, confidence = predict_corn(CORN_MODEL, file_content)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        print(f"Corn Prediction Error: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to process image with Corn model: {e}")

    # 2. ЗБЕРЕЖЕННЯ В MINIO
    upload_time = datetime.utcnow()
    original_object_name = f"analysis/{current_user['_id']}/corn/{upload_time.isoformat()}_{file.filename}"
    upload_file_to_minio(file_content, original_object_name, file.content_type)

    # 3. ЛОГУВАННЯ В MONGODB
    analysis_record = {
        "user_id": str(current_user["_id"]),
        "model": "corn_classification",
        "uploaded_at": upload_time,
        "original_file_key": original_object_name,
        "prediction": {
            "class": predicted_class,
            "confidence": confidence
        }
    }
    await db["analysis_logs"].insert_one(analysis_record)

    return JSONResponse(content={
        "message": "Corn disease classified successfully.",
        "prediction": predicted_class,
        "confidence": f"{confidence * 100:.2f}%",
        "file_key": original_object_name
    })


@app.post("/api/fields/upload/potato", tags=["AI Prediction"])
async def analyze_potato_disease(
        file: UploadFile = File(..., description="Зображення листя картоплі"),
        current_user: dict = Depends(get_current_user),
        db: AsyncIOMotorClient = Depends(get_database)
):
    """
    Класифікація хвороб картоплі. Приймає 1 файл зображення.
    """
    if not POTATO_MODEL:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Potato model is not initialized.")
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid file type. Only JPEG and PNG allowed.")

    file_content = await file.read()

    # 1. АНАЛІЗ
    try:
        predicted_class, confidence = predict_potato(POTATO_MODEL, file_content)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        print(f"Potato Prediction Error: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to process image with Potato model: {e}")

    # 2. ЗБЕРЕЖЕННЯ В MINIO
    upload_time = datetime.utcnow()
    original_object_name = f"analysis/{current_user['_id']}/potato/{upload_time.isoformat()}_{file.filename}"
    upload_file_to_minio(file_content, original_object_name, file.content_type)

    # 3. ЛОГУВАННЯ В MONGODB
    analysis_record = {
        "user_id": str(current_user["_id"]),
        "model": "potato_classification",
        "uploaded_at": upload_time,
        "original_file_key": original_object_name,
        "prediction": {
            "class": predicted_class,
            "confidence": confidence
        }
    }
    await db["analysis_logs"].insert_one(analysis_record)

    return JSONResponse(content={
        "message": "Potato disease classified successfully.",
        "prediction": predicted_class,
        "confidence": f"{confidence * 100:.2f}%",
        "file_key": original_object_name
    })


@app.post("/api/fields/upload/weed", tags=["AI Prediction"])
async def analyze_weed_segmentation(
        rgb_file: UploadFile = File(..., alias="rgb_image", description="RGB зображення поля"),
        aux_file: UploadFile = File(None, alias="aux_image",
                                    description="Додаткове (наприклад, NIR) зображення (необов'язково)"),
        current_user: dict = Depends(get_current_user),
        db: AsyncIOMotorClient = Depends(get_database)
):
    """
    Сегментація бур'янів (UNET). Приймає 1 (RGB) або 2 (RGB + Aux) файли.
    Повертає посилання на згенерований файл-маску.
    """
    if not WEED_MODEL:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Weed model is not initialized.")
    if rgb_file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid file type for RGB. Only JPEG and PNG allowed.")

    rgb_content = await rgb_file.read()

    # 1. АНАЛІЗ (СЕГМЕНТАЦІЯ)
    try:
        # predict_weed_segmentation повертає байтову послідовність PNG маски
        # WEED_MODEL - заглушка/реальна модель UNET
        mask_bytes = predict_weed_segmentation(WEED_MODEL, rgb_content)
    except Exception as e:
        print(f"Weed Prediction Error: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to process image with Weed model: {e}")

    upload_time = datetime.utcnow()

    # 2. ЗБЕРЕЖЕННЯ ОРИГІНАЛЬНИХ ФАЙЛІВ У MINIO
    original_rgb_object_name = f"analysis/{current_user['_id']}/weed/{upload_time.isoformat()}_{rgb_file.filename}"
    upload_file_to_minio(rgb_content, original_rgb_object_name, rgb_file.content_type)

    aux_object_name = None
    if aux_file:
        aux_content = await aux_file.read()
        aux_object_name = f"analysis/{current_user['_id']}/weed/{upload_time.isoformat()}_aux_{aux_file.filename}"
        upload_file_to_minio(aux_content, aux_object_name, aux_file.content_type)

    # 3. ЗБЕРЕЖЕННЯ РЕЗУЛЬТУЮЧОЇ МАСКИ У MINIO
    mask_object_name = f"analysis/{current_user['_id']}/weed/{upload_time.isoformat()}_mask.png"
    upload_file_to_minio(mask_bytes, mask_object_name, "image/png")

    # 4. ЛОГУВАННЯ В MONGODB
    analysis_record = {
        "user_id": str(current_user["_id"]),
        "model": "weed_segmentation",
        "uploaded_at": upload_time,
        "original_file_key": original_rgb_object_name,
        "aux_file_key": aux_object_name,
        "result_mask_key": mask_object_name,
        "prediction": {"status": "Mask generated"}
    }
    await db["analysis_logs"].insert_one(analysis_record)

    return JSONResponse(content={
        "message": "Weed segmentation completed successfully.",
        "original_rgb_file_key": original_rgb_object_name,
        "aux_file_key": aux_object_name,
        "result_mask_key": mask_object_name
    })


@app.get("/health", tags=["Health"])
async def health_check():
    if connect_to_mongo():
        return {"status": "ok"}
    else:
        return {"status": "dead"}


if __name__ == "__main__":
    # Додаємо CornCNN та PotatoCNN до globals для uvicorn (потрібно для reload=True)
    globals()['CornCNN'] = CornCNN
    globals()['PotatoCNN'] = PotatoCNN

    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
# --- ЗАПУСК ---
# uvicorn main:app --reload --host 0.0.0.0 --port 8000