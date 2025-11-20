from mongoengine import Document, EmbeddedDocument, fields, connect
from datetime import datetime
import auth_service
from ..dependencies import connect_to_mongo

# =========================================================================
#                           "type" for user                               #
# =========================================================================
USER_ROLES = (
    ("Admin", "Адміністратор (повний доступ)"),
    ("Owner", "Власник (управління даними та користувачами)"),
    ("Viewer", "Спостерігач (лише перегляд даних)")
)

# =========================================================================
#                       main class Document for User                      #
# =========================================================================

class User(Document):
    """
    Модель користувача для MongoDB, що використовує MongoEngine.
    Визначає поля, валідацію та індекси для сутності користувача.
    """
    email = fields.EmailField(
        required=True,
        unique=True,
        max_length=100,
        verbose_name="Електронна пошта"
    )

    password_hash = fields.StringField(
        required=True,
        min_length=60,
        verbose_name="Хеш пароля"
    )

    name = fields.StringField(
        required=True,
        max_length=150,
        verbose_name="Ім'я користувача"
    )

    type = fields.StringField(
        required=True,
        choices=[choice[0] for choice in USER_ROLES],
        default="Viewer",
        verbose_name="Роль користувача"
    )

    plan_id = fields.StringField(
        max_length=50,
        null=True, # Дозволяємо бути NULL, якщо план не обрано
        verbose_name="ID тарифного плану"
    )

    created_at = fields.DateTimeField(
        required=True,
        default=datetime.utcnow,
        verbose_name="Дата створення"
    )

    # Дата останнього входу
    last_logon = fields.DateTimeField(
        null=True, # Може бути NULL для нового користувача
        verbose_name="Дата останнього входу"
    )

    meta = {
        'collection': 'users', # Назва колекції в MongoDB
        'indexes': [
            'email', # Створюємо індекс для швидкого пошуку за email
            'plan_id'
        ]
    }

    def set_password(self, password):
        """Хешує пароль та зберігає хеш у полі password_hash."""
        self.password_hash = auth_service.AuthService.hash_password(password)
        # pass

    def check_password(self, password):
        """Перевіряє наданий пароль проти збереженого хешу."""
        return auth_service.AuthService.verify_password(password, self.password_hash)
        # return False # Заглушка

    def __str__(self):
        return f"Користувач: {self.name} ({self.email})"

# =========================================================================
# 3. Приклад використання (не виконується в MongoEngine без ініціалізації)
# =========================================================================

connect_to_mongo()
# Створення нового користувачами
new_user = User(
    email="admin@example.com",
    password_hash="...ваш_хеш...",
    name="Головний Адмін",
    type="Admin",
    plan_id="premium_yearly"
)
new_user.save()
print(f"Створено: {new_user}")