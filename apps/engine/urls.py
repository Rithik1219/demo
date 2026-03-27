"""
apps/engine/urls.py – URL patterns for the engine app.
"""

from django.urls import path

from apps.engine import views

app_name = "engine"

urlpatterns = [
    path("dashboard/", views.dashboard, name="dashboard"),
    path("api/predictions/", views.api_predictions, name="api_predictions"),
    path("api/trigger-pipeline/", views.api_trigger_pipeline, name="api_trigger_pipeline"),
]
