"""
apps/engine/tasks.py – Celery tasks for the Portfolio Predictive Engine.

Tasks
-----
- calculate_technical_features : compute RSI / MACD / EMA / BB / ATR / OBV / Stoch
                                  for all symbols that have new StockHistory rows.
- run_sentiment_analysis       : scrape Yahoo Finance headlines for each symbol,
                                  run FinBERT, and persist NewsSentiment rows.
- run_ingestion_task           : scheduled data ingestion wrapper.
- run_ml_pipeline_task         : trigger the full ML pipeline and store Prediction rows.

All tasks are idempotent and safe to retry.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from celery import shared_task
from django.conf import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task: calculate technical features
# ---------------------------------------------------------------------------

@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=120,
    max_retries=3,
    name="engine.calculate_technical_features",
)
def calculate_technical_features(self, symbols: Optional[List[str]] = None) -> dict:
    """
    Compute technical indicators for *symbols* using pandas-ta and persist
    them in the TechnicalFeatures table.

    If *symbols* is None, all symbols present in StockHistory are processed.

    Anti-leakage guarantee: indicators are always computed on the full
    chronological series; no future data is used.  Callers must ensure
    they split on index (not shuffle) before passing to any ML model.

    Returns a dict mapping symbol → number of rows updated.
    """
    import numpy as np
    import pandas as pd
    import pandas_ta as ta
    from django.db import transaction

    from apps.engine.models import StockHistory, TechnicalFeatures

    # Resolve symbol list
    if not symbols:
        symbols = list(
            StockHistory.objects.values_list("symbol", flat=True).distinct()
        )

    results = {}

    for symbol in symbols:
        logger.info("Calculating technical features for %s", symbol)

        qs = (
            StockHistory.objects.filter(symbol=symbol, interval="1d")
            .order_by("timestamp")
            .values(
                "id", "timestamp",
                "open_price", "high_price", "low_price",
                "close_price", "volume",
            )
        )

        rows = list(qs)
        if len(rows) < 200:
            logger.warning(
                "%s: only %d daily bars available – skipping (need ≥200 for EMA-200)",
                symbol, len(rows),
            )
            continue

        df = pd.DataFrame(rows)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Cast to float64 – pandas-ta requires numeric series
        for col in ["open_price", "high_price", "low_price", "close_price", "volume"]:
            df[col] = df[col].astype(float)

        # ------------------------------------------------------------------
        # Compute indicators (ALL computed forward in time – no look-ahead)
        # ------------------------------------------------------------------

        # EMA
        df["ema_9"]   = ta.ema(df["close_price"], length=9)
        df["ema_21"]  = ta.ema(df["close_price"], length=21)
        df["ema_50"]  = ta.ema(df["close_price"], length=50)
        df["ema_200"] = ta.ema(df["close_price"], length=200)

        # RSI
        df["rsi_14"] = ta.rsi(df["close_price"], length=14)

        # MACD (12, 26, 9)
        macd_df = ta.macd(df["close_price"], fast=12, slow=26, signal=9)
        df["macd"]           = macd_df["MACD_12_26_9"]
        df["macd_signal"]    = macd_df["MACDs_12_26_9"]
        df["macd_histogram"] = macd_df["MACDh_12_26_9"]

        # Bollinger Bands (20, 2)
        bb_df = ta.bbands(df["close_price"], length=20, std=2)
        df["bb_upper"]   = bb_df["BBU_20_2.0"]
        df["bb_middle"]  = bb_df["BBM_20_2.0"]
        df["bb_lower"]   = bb_df["BBL_20_2.0"]
        df["bb_percent"] = bb_df["BBP_20_2.0"]

        # ATR
        df["atr_14"] = ta.atr(
            df["high_price"], df["low_price"], df["close_price"], length=14
        )

        # OBV
        df["obv"] = ta.obv(df["close_price"], df["volume"])

        # Stochastic Oscillator
        stoch_df = ta.stoch(
            df["high_price"], df["low_price"], df["close_price"], k=14, d=3, smooth_k=3
        )
        df["stoch_k"] = stoch_df["STOCHk_14_3_3"]
        df["stoch_d"] = stoch_df["STOCHd_14_3_3"]

        # Williams %R
        df["williams_r"] = ta.willr(
            df["high_price"], df["low_price"], df["close_price"], length=14
        )

        # CCI
        df["cci_20"] = ta.cci(
            df["high_price"], df["low_price"], df["close_price"], length=20
        )

        # ------------------------------------------------------------------
        # Persist – replace NaN with None (SQL NULL)
        # ------------------------------------------------------------------
        feature_cols = [
            "ema_9", "ema_21", "ema_50", "ema_200",
            "rsi_14",
            "macd", "macd_signal", "macd_histogram",
            "bb_upper", "bb_middle", "bb_lower", "bb_percent",
            "atr_14", "obv",
            "stoch_k", "stoch_d",
            "williams_r", "cci_20",
        ]
        df[feature_cols] = df[feature_cols].where(df[feature_cols].notna(), other=None)

        updated = 0
        with transaction.atomic():
            for _, row in df.iterrows():
                stock_id = int(row["id"])
                feature_data = {col: row[col] for col in feature_cols}
                # Replace numpy nan with None
                feature_data = {
                    k: (None if (v is None or (isinstance(v, float) and np.isnan(v))) else v)
                    for k, v in feature_data.items()
                }
                TechnicalFeatures.objects.update_or_create(
                    stock_id=stock_id,
                    defaults=feature_data,
                )
                updated += 1

        results[symbol] = updated
        logger.info("Updated %d TechnicalFeatures rows for %s", updated, symbol)

    return results


# ---------------------------------------------------------------------------
# Task: sentiment analysis
# ---------------------------------------------------------------------------

@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=120,
    max_retries=3,
    name="engine.run_sentiment_analysis",
)
def run_sentiment_analysis(self, symbols: Optional[List[str]] = None) -> dict:
    """
    Scrape Yahoo Finance RSS headlines for each symbol, run ProsusAI/finbert,
    and persist the results in NewsSentiment.

    Returns a dict mapping symbol → number of headlines analysed.
    """
    import xml.etree.ElementTree as ET
    from datetime import datetime, timezone

    import requests
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from apps.engine.models import NewsSentiment, StockHistory

    # Resolve symbols
    if not symbols:
        symbols = list(
            StockHistory.objects.values_list("symbol", flat=True).distinct()
        )

    # ------------------------------------------------------------------
    # Load FinBERT once per task execution
    # ------------------------------------------------------------------
    model_name = "ProsusAI/finbert"
    cache_dir = getattr(settings, "HF_MODEL_CACHE", None)

    logger.info("Loading FinBERT tokenizer and model…")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, cache_dir=cache_dir
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("FinBERT loaded on %s", device)

    label_map = {0: "positive", 1: "negative", 2: "neutral"}

    def _analyse_batch(texts: List[str]) -> List[dict]:
        """Run FinBERT on a batch and return per-text result dicts."""
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        results_batch = []
        for prob in probs:
            label_idx = int(prob.argmax())
            results_batch.append(
                {
                    "sentiment_label": label_map[label_idx],
                    "positive_score": float(prob[0]),
                    "negative_score": float(prob[1]),
                    "neutral_score": float(prob[2]),
                }
            )
        return results_batch

    # ------------------------------------------------------------------
    # Fetch headlines from Yahoo Finance RSS
    # ------------------------------------------------------------------
    results: dict = {}
    BATCH_SIZE = 16

    for symbol in symbols:
        yf_ticker = f"{symbol}.NS"
        rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={yf_ticker}&region=IN&lang=en-IN"

        try:
            resp = requests.get(rss_url, timeout=15)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except (requests.RequestException, ET.ParseError) as exc:
            logger.warning("Failed to fetch RSS for %s: %s", symbol, exc)
            results[symbol] = 0
            continue

        items = root.findall(".//item")
        headlines, urls, pub_dates = [], [], []

        for item in items:
            title_el = item.find("title")
            link_el = item.find("link")
            pub_el = item.find("pubDate")

            if title_el is None or not title_el.text:
                continue

            headline = title_el.text.strip()
            url = (link_el.text or "").strip() if link_el is not None else ""
            pub_date_str = pub_el.text.strip() if pub_el is not None and pub_el.text else ""

            try:
                pub_dt = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %z")
            except ValueError:
                pub_dt = datetime.now(tz=timezone.utc)

            headlines.append(headline)
            urls.append(url)
            pub_dates.append(pub_dt)

        if not headlines:
            results[symbol] = 0
            continue

        # Batch inference
        all_sentiments = []
        for i in range(0, len(headlines), BATCH_SIZE):
            batch_texts = headlines[i: i + BATCH_SIZE]
            all_sentiments.extend(_analyse_batch(batch_texts))

        # Persist
        saved = 0
        for headline, url, pub_dt, sentiment in zip(headlines, urls, pub_dates, all_sentiments):
            obj, created = NewsSentiment.objects.update_or_create(
                symbol=symbol,
                headline=headline,
                published_at=pub_dt,
                defaults={
                    "source_url": url,
                    **sentiment,
                },
            )
            if created:
                saved += 1

        results[symbol] = saved
        logger.info("Saved %d new sentiment rows for %s", saved, symbol)

    return results


# ---------------------------------------------------------------------------
# Task: data ingestion wrapper
# ---------------------------------------------------------------------------

@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=120,
    max_retries=3,
    name="engine.run_ingestion_task",
)
def run_ingestion_task(
    self,
    symbols: Optional[List[str]] = None,
    interval: str = "1d",
    days: int = 365,
) -> dict:
    """
    Celery-wrapped entry point for ``apps.engine.ingestion.run_ingestion``.

    Parameters
    ----------
    symbols  : list of NSE tickers; if None, a default watchlist is used
    interval : OHLCV bar interval (e.g. "1d")
    days     : lookback window in calendar days
    """
    from apps.engine.ingestion import run_ingestion

    watchlist = symbols or [
        "RELIANCE", "INFY", "TCS", "HDFCBANK", "ICICIBANK",
        "SBIN", "BHARTIARTL", "WIPRO", "HINDUNILVR", "KOTAKBANK",
    ]
    return run_ingestion(symbols=watchlist, interval=interval, days=days)


# ---------------------------------------------------------------------------
# Task: ML pipeline trigger
# ---------------------------------------------------------------------------

@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=120,
    max_retries=3,
    name="engine.run_ml_pipeline_task",
    time_limit=3600,
    soft_time_limit=3300,
)
def run_ml_pipeline_task(
    self,
    symbols: Optional[List[str]] = None,
) -> dict:
    """
    Trigger the full ML pipeline (LSTM + XGBoost + FinBERT → Logistic Regression
    meta-model) and persist Prediction rows.
    """
    from apps.engine.ml_pipeline import run_pipeline

    watchlist = symbols or [
        "RELIANCE", "INFY", "TCS", "HDFCBANK", "ICICIBANK",
        "SBIN", "BHARTIARTL", "WIPRO", "HINDUNILVR", "KOTAKBANK",
    ]
    return run_pipeline(symbols=watchlist)
