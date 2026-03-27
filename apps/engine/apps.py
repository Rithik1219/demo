"""
apps/engine/apps.py – AppConfig for the engine application.
"""

from django.apps import AppConfig


class EngineConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.engine"
    verbose_name = "Portfolio Predictive Engine"
