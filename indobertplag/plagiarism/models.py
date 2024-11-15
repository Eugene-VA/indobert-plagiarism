from django.db import models
from django.contrib.postgres.fields import ArrayField

# Create your models here.
class Reference(models.Model):
    title = models.CharField(max_length=255, db_index=True)
    authors = models.TextField(default="Anonim", db_index=True)
    content = models.TextField()
    embeddings = models.JSONField()

    def __str__(self):
        return self.title