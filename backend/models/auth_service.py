import bcrypt
from datetime import datetime
from user import User


SALT_ROUNDS = 12

class AuthService:
    """
    Сервіс, що містить бізнес-логіку для автентифікації користувачів:
    реєстрація (створення хешу) та вхід (перевірка хешу).
    """

    @staticmethod
    def hash_password(password: str) -> str:
        """
        Створює BCrypt хеш пароля. Ця функція викликається ПРИ РЕЄСТРАЦІЇ.

        :param password: Пароль у вигляді простого тексту.
        :return: Захешований пароль (рядок).
        """
        # from str to bytes
        password_bytes = password.encode('utf-8')

        # new SALT
        hashed_bytes = bcrypt.hashpw(
            password=password_bytes,
            salt=bcrypt.gensalt(rounds=SALT_ROUNDS)
        )

        # from bytes to str
        return hashed_bytes.decode('utf-8')

    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """
        Перевіряє, чи відповідає наданий пароль збереженому хешу.
        Ця функція викликається ПРИ ВХОДІ.

        :param password: Пароль, введений користувачем.
        :param password_hash: Збережений хеш з бази даних.
        :return: True, якщо паролі збігаються, інакше False.
        """
        return bcrypt.checkpw(
            password=password.encode('utf-8'),
            hashed_password=password_hash.encode('utf-8')
        )

    @classmethod
    def register_user(cls, email: str, password: str, name: str) -> User:
        """
        Функція реєстрації з хешуванням.
        """
        hashed_pw = cls.hash_password(password)

        new_user = User(
            email=email,
            password_hash=hashed_pw,
            name=name,
            type="Owner",  # TODO default for now, could be changed
            created_at=datetime.utcnow()
        )

        new_user.save()
        return new_user

    @classmethod
    def login_user(cls, email: str, password: str) -> bool:
        """
        Функція входу.
        """
        # find user
        try:
            user = User.objects.get(email=email)
            # Цей рядок буде працювати після налаштування підключення до MongoEngine

            # Приклад: припустимо, ми знайшли користувача і знаємо його хеш
            hashed_from_db = user.password_hash
            # hashed_from_db = "$2b$12$EXAMPLEHASHFROMDB.Qc2YJ/g8E6pA8mX/3R"  # Заглушка

        except Exception:  # User.DoesNotExist:
            print("Користувача не знайдено.")
            return False

        # check password
        if cls.verify_password(password, hashed_from_db):
            print(f"Вхід успішний для {email}!")
            user.last_logon = datetime.utcnow()
            user.save()
            return True
        else:
            print("Неправильний пароль.")
            return False