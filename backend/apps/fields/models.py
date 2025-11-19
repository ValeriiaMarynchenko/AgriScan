from backend.apps.users.models import CustomUser  # Припустимо, у вас є CustomUser
from django.db import models


class Field(models.Model):
    owner = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)

    # Store coordinates as a string (e.g., GeoJSON or WKT)
    # Note: You lose the .area calculation and database indexing/querying
    geometry_data = models.TextField()

    area_ha = models.FloatField(editable=False, blank=True, null=True)

    def save(self, *args, **kwargs):
        # The automatic area calculation self.geometry.area will no longer work!
        # You'll have to use a third-party Python library (like Shapely)
        # to parse geometry_data and manually calculate the area.
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name