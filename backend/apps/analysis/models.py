from django.db import models

# models.py в analysis/
from django.db import models
from ..fields.models import Field
from ..users.models import CustomUser

class AnalysisJob(models.Model):
    field = models.ForeignKey(Field, on_delete=models.CASCADE, related_name='analysis_jobs')
    image_url = models.URLField(verbose_name="URL аерознімка") # Посилання на зображення в S3/GCS

    STATUS_CHOICES = [
        ('PENDING', 'Очікує'), ('PROCESSING', 'Обробляється'),
        ('COMPLETED', 'Завершено'), ('FAILED', 'Помилка')
    ]

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING') # PENDING, PROCESSING, COMPLETED, FAILED
    result_url = models.URLField(null=True, blank=True, verbose_name="URL результату") # Посилання на результат (маска/зонування)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    # updated_at = models.DateTimeField(auto_now=True)
    #
    # class Meta:
    #     abstract = True
    #

    def __str__(self):
        return f"Аналіз {self.field.name} - {self.status}"