"""
apps/engine/admin.py – Django admin registrations for the engine models.
"""

from django.contrib import admin

from apps.engine.models import NewsSentiment, Prediction, StockHistory, TechnicalFeatures


@admin.register(StockHistory)
class StockHistoryAdmin(admin.ModelAdmin):
    list_display = ("symbol", "exchange", "interval", "timestamp", "close_price", "volume")
    list_filter  = ("symbol", "exchange", "interval")
    search_fields = ("symbol",)
    ordering = ("-timestamp",)
    date_hierarchy = "timestamp"


@admin.register(TechnicalFeatures)
class TechnicalFeaturesAdmin(admin.ModelAdmin):
    list_display = ("stock", "rsi_14", "macd", "ema_50", "updated_at")
    search_fields = ("stock__symbol",)
    raw_id_fields = ("stock",)


@admin.register(NewsSentiment)
class NewsSentimentAdmin(admin.ModelAdmin):
    list_display  = ("symbol", "sentiment_label", "positive_score", "negative_score", "published_at")
    list_filter   = ("symbol", "sentiment_label")
    search_fields = ("symbol", "headline")
    ordering      = ("-published_at",)
    date_hierarchy = "published_at"


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display  = ("symbol", "signal", "buy_probability", "hold_probability",
                     "sell_probability", "predicted_at", "model_version")
    list_filter   = ("symbol", "signal", "model_version")
    search_fields = ("symbol",)
    ordering      = ("-predicted_at",)
    date_hierarchy = "predicted_at"
