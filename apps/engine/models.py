"""
apps/engine/models.py – Database schema for the Portfolio Predictive Engine.

Tables
------
- StockHistory      : raw OHLCV bars from Angel One / yfinance
- TechnicalFeatures : engineered indicators (RSI, MACD, EMA, …)
- NewsSentiment     : per-headline FinBERT sentiment scores
- Prediction        : final Buy / Hold / Sell signals from the meta-model
"""

from django.db import models


class StockHistory(models.Model):
    """Raw OHLCV price data for a single ticker at a given interval."""

    INTERVAL_CHOICES = [
        ("1m", "1 Minute"),
        ("5m", "5 Minutes"),
        ("15m", "15 Minutes"),
        ("30m", "30 Minutes"),
        ("1h", "1 Hour"),
        ("1d", "1 Day"),
        ("1w", "1 Week"),
    ]

    symbol = models.CharField(max_length=30, db_index=True)
    exchange = models.CharField(max_length=10, default="NSE")
    interval = models.CharField(max_length=5, choices=INTERVAL_CHOICES, default="1d")
    timestamp = models.DateTimeField(db_index=True)

    open_price = models.DecimalField(max_digits=14, decimal_places=4)
    high_price = models.DecimalField(max_digits=14, decimal_places=4)
    low_price = models.DecimalField(max_digits=14, decimal_places=4)
    close_price = models.DecimalField(max_digits=14, decimal_places=4)
    volume = models.BigIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = "engine"
        unique_together = ("symbol", "exchange", "interval", "timestamp")
        ordering = ["symbol", "timestamp"]
        indexes = [
            models.Index(fields=["symbol", "interval", "timestamp"]),
        ]

    def __str__(self) -> str:
        return f"{self.symbol}@{self.exchange} [{self.interval}] {self.timestamp:%Y-%m-%d %H:%M}"


class TechnicalFeatures(models.Model):
    """
    Engineered technical indicators derived from StockHistory.

    All values are computed chronologically (no look-ahead) and stored
    here so the ML pipeline can load them without re-computing each run.
    """

    stock = models.OneToOneField(
        StockHistory,
        on_delete=models.CASCADE,
        related_name="technical_features",
        primary_key=True,
    )

    # Trend indicators
    ema_9 = models.FloatField(null=True, blank=True)
    ema_21 = models.FloatField(null=True, blank=True)
    ema_50 = models.FloatField(null=True, blank=True)
    ema_200 = models.FloatField(null=True, blank=True)

    # Momentum indicators
    rsi_14 = models.FloatField(null=True, blank=True)

    # MACD
    macd = models.FloatField(null=True, blank=True)
    macd_signal = models.FloatField(null=True, blank=True)
    macd_histogram = models.FloatField(null=True, blank=True)

    # Bollinger Bands
    bb_upper = models.FloatField(null=True, blank=True)
    bb_middle = models.FloatField(null=True, blank=True)
    bb_lower = models.FloatField(null=True, blank=True)
    bb_percent = models.FloatField(null=True, blank=True)  # %B

    # Average True Range (volatility)
    atr_14 = models.FloatField(null=True, blank=True)

    # On-Balance Volume
    obv = models.FloatField(null=True, blank=True)

    # Stochastic Oscillator
    stoch_k = models.FloatField(null=True, blank=True)
    stoch_d = models.FloatField(null=True, blank=True)

    # Williams %R
    williams_r = models.FloatField(null=True, blank=True)

    # Commodity Channel Index
    cci_20 = models.FloatField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = "engine"

    def __str__(self) -> str:
        return f"Features for {self.stock}"


class NewsSentiment(models.Model):
    """Per-headline FinBERT sentiment score linked to a ticker."""

    LABEL_CHOICES = [
        ("positive", "Positive"),
        ("neutral", "Neutral"),
        ("negative", "Negative"),
    ]

    symbol = models.CharField(max_length=30, db_index=True)
    headline = models.TextField()
    source_url = models.URLField(max_length=1000, blank=True, default="")
    published_at = models.DateTimeField(db_index=True)

    # FinBERT outputs
    sentiment_label = models.CharField(max_length=10, choices=LABEL_CHOICES)
    positive_score = models.FloatField()
    neutral_score = models.FloatField()
    negative_score = models.FloatField()

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "engine"
        unique_together = ("symbol", "headline", "published_at")
        ordering = ["-published_at"]
        indexes = [
            models.Index(fields=["symbol", "published_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.symbol} | {self.sentiment_label} | {self.published_at:%Y-%m-%d}"


class Prediction(models.Model):
    """Final Buy / Hold / Sell meta-model prediction for a ticker + timestamp."""

    SIGNAL_CHOICES = [
        ("BUY", "Buy"),
        ("HOLD", "Hold"),
        ("SELL", "Sell"),
    ]

    symbol = models.CharField(max_length=30, db_index=True)
    predicted_at = models.DateTimeField(db_index=True)

    # Component model probabilities (0–1)
    lstm_confidence = models.FloatField(help_text="LSTM directional confidence (up probability)")
    xgb_confidence = models.FloatField(help_text="XGBoost BUY class probability")
    sentiment_score = models.FloatField(
        help_text="Aggregate FinBERT score: positive − negative"
    )

    # Meta-model output
    signal = models.CharField(max_length=4, choices=SIGNAL_CHOICES)
    buy_probability = models.FloatField()
    hold_probability = models.FloatField()
    sell_probability = models.FloatField()

    # Metadata
    model_version = models.CharField(max_length=40, default="v1")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "engine"
        ordering = ["-predicted_at"]
        indexes = [
            models.Index(fields=["symbol", "predicted_at"]),
        ]

    def __str__(self) -> str:
        return (
            f"{self.symbol} → {self.signal} "
            f"(buy={self.buy_probability:.2f}, "
            f"sell={self.sell_probability:.2f}) "
            f"@ {self.predicted_at:%Y-%m-%d %H:%M}"
        )
