from django.db import models
from django.contrib.auth.models import AbstractUser, BaseUserManager # Import BaseUserManager
from django.utils.translation import gettext_lazy as _ # Optional: for better internationalization in verbose names

# --- 1. Custom Manager (CRUCIAL FIX) ---
class CustomUserManager(BaseUserManager):
    """
    Custom user manager where email is the unique identifier
    for authentication instead of usernames.
    """
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError(_('The Email field must be set'))
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('is_active', True)

        if not extra_fields.get('is_staff'):
            raise ValueError(_('Superuser must have is_staff=True.'))
        if not extra_fields.get('is_superuser'):
            raise ValueError(_('Superuser must have is_superuser=True.'))

        return self.create_user(email, password, **extra_fields)


# --- 2. User Role Constants ---
class UserRole(models.TextChoices):
    FARMER = 'FARMER', 'Фермер'
    ADMIN = 'ADMIN', 'Адміністратор'
    VIEWER = 'VIEWER', 'Спостерігач'


# --- 3. Custom User Model ---
class CustomUser(AbstractUser):

    id = models.BigAutoField(primary_key=True)
    email = models.EmailField(unique=True, verbose_name='Електронна пошта')

    role = models.CharField(
        max_length=10,
        choices=UserRole.choices,
        default=UserRole.FARMER,
        verbose_name='Роль користувача'
    )

    organization_name = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        verbose_name='Назва організації'
    )

    # Видаляємо поле username, щоб вхід був лише через email
    username = None # MUST be set to None when using email as USERNAME_FIELD

    # Призначаємо CustomUserManager
    objects = CustomUserManager()

    # Вказуємо, що поле 'email' використовується для входу
    USERNAME_FIELD = 'email'

    # Поля, які будуть запитані при створенні суперкористувача
    # Примітка: 'password' завжди запитується.
    REQUIRED_FIELDS = ['first_name', 'last_name'] #

    class Meta:
        verbose_name = 'Користувач'
        verbose_name_plural = 'Користувачі'

    def __str__(self):
        return self.email