from djoser.serializers import UserCreateSerializer as DjoserUserCreateSerializer
from djoser.serializers import UserSerializer as DjoserUserSerializer
from .models import CustomUser

# Серіалізатор для Реєстрації (POST /api/v1/auth/users/)
# Дозволяє створювати користувача, виключаючи sensitive поля
class UserCreateSerializer(DjoserUserCreateSerializer):
    class Meta(DjoserUserCreateSerializer.Meta):
        model = CustomUser
        fields = ('id', 'email', 'first_name', 'last_name', 'password', 'organization_name')
        # Роль автоматично встановлюється як FARMER у моделі

# Серіалізатор для Читання/Оновлення (GET/PUT /api/auth/users/me/)
class CustomUserSerializer(DjoserUserSerializer):
    class Meta(DjoserUserSerializer.Meta):
        model = CustomUser
        fields = ('id', 'email', 'first_name', 'last_name', 'organization_name', 'role', 'is_active')
        read_only_fields = ('email', 'role',) # Email та роль змінюються лише через адмінку/спец. запити
        # extra_kwargs = {'password': {'write_only': True}}


