"""
config/urls.py – Root URL configuration.
"""

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("engine/", include("apps.engine.urls")),
]
