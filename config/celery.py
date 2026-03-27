"""
config/celery.py – Celery application instance for the project.
"""

import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("portfolio_engine")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
