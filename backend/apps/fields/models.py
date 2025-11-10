from django.db import models

from django.contrib.gis.db import models
from apps.users.models import CustomUser  # Припустимо, у вас є CustomUser


class Field(models.Model):
    owner = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    # Зберігає полігон (геометрію) поля
    geometry = models.PolygonField()

    area_ha = models.FloatField(editable=False)  # Площа в гектарах

    def save(self, *args, **kwargs):
        # Автоматично розраховуємо площу перед збереженням
        if self.geometry:
            # Розрахунок площі з використанням сфероїда (для точності)
            self.area_ha = self.geometry.area / 10000
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name