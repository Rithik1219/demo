"""
apps/engine/views.py – Django views for the Portfolio Predictive Engine dashboard.

Endpoints
---------
GET /engine/dashboard/
    Renders the main dashboard showing the latest Buy/Hold/Sell predictions
    for all tracked symbols, sorted by buy probability descending.

GET /engine/api/predictions/
    JSON endpoint returning the same data for programmatic consumption.

GET /engine/api/trigger-pipeline/   (POST recommended in production)
    Queues background tasks: ingestion → features → sentiment → ml_pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_http_methods

from apps.engine.models import Prediction, StockHistory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest_predictions() -> List[Dict[str, Any]]:
    """
    Return one prediction per symbol (the most recent), sorted by
    buy_probability descending so the strongest BUY signals appear first.
    """
    # Subquery: latest predicted_at per symbol
    from django.db.models import Max, Subquery, OuterRef

    latest_subq = (
        Prediction.objects.filter(symbol=OuterRef("symbol"))
        .order_by("-predicted_at")
        .values("id")[:1]
    )
    qs = (
        Prediction.objects.filter(id__in=Subquery(latest_subq))
        .order_by("-buy_probability")
        .values(
            "symbol",
            "signal",
            "buy_probability",
            "hold_probability",
            "sell_probability",
            "lstm_confidence",
            "xgb_confidence",
            "sentiment_score",
            "predicted_at",
            "model_version",
        )
    )
    return list(qs)


def _symbol_stats() -> Dict[str, int]:
    """Return a dict mapping symbol → number of daily bars in StockHistory."""
    from django.db.models import Count

    qs = (
        StockHistory.objects.filter(interval="1d")
        .values("symbol")
        .annotate(bar_count=Count("id"))
    )
    return {row["symbol"]: row["bar_count"] for row in qs}


# ---------------------------------------------------------------------------
# Dashboard view
# ---------------------------------------------------------------------------

@require_GET
def dashboard(request: HttpRequest) -> HttpResponse:
    """Main dashboard – renders predictions as a Tailwind CSS table."""
    predictions = _latest_predictions()
    symbol_stats = _symbol_stats()

    # Enrich each prediction row
    for pred in predictions:
        pred["bar_count"] = symbol_stats.get(pred["symbol"], 0)
        pred["predicted_at_fmt"] = pred["predicted_at"].strftime("%Y-%m-%d %H:%M UTC")
        # Convert 0-1 probabilities to 0-100 percentage strings for CSS widths
        pred["buy_pct"]  = f"{pred['buy_probability']  * 100:.1f}%"
        pred["hold_pct"] = f"{pred['hold_probability'] * 100:.1f}%"
        pred["sell_pct"] = f"{pred['sell_probability'] * 100:.1f}%"
        # Signal badge colour
        pred["signal_class"] = {
            "BUY": "bg-green-100 text-green-800",
            "HOLD": "bg-yellow-100 text-yellow-800",
            "SELL": "bg-red-100 text-red-800",
        }.get(pred["signal"], "bg-gray-100 text-gray-800")

    context = {
        "predictions": predictions,
        "total_symbols": len(predictions),
        "buy_count": sum(1 for p in predictions if p["signal"] == "BUY"),
        "hold_count": sum(1 for p in predictions if p["signal"] == "HOLD"),
        "sell_count": sum(1 for p in predictions if p["signal"] == "SELL"),
    }
    return render(request, "engine/dashboard.html", context)


# ---------------------------------------------------------------------------
# JSON API endpoints
# ---------------------------------------------------------------------------

@require_GET
def api_predictions(request: HttpRequest) -> JsonResponse:
    """Return latest predictions as JSON."""
    predictions = _latest_predictions()
    data = []
    for pred in predictions:
        row = dict(pred)
        row["predicted_at"] = pred["predicted_at"].isoformat()
        data.append(row)
    return JsonResponse({"status": "ok", "count": len(data), "predictions": data})


@require_http_methods(["GET", "POST"])
def api_trigger_pipeline(request: HttpRequest) -> JsonResponse:
    """
    Queue all pipeline tasks asynchronously.

    In production, protect this endpoint with authentication and use POST.
    For development convenience, GET is also accepted.
    """
    try:
        from apps.engine.tasks import (
            calculate_technical_features,
            run_ingestion_task,
            run_ml_pipeline_task,
            run_sentiment_analysis,
        )

        # Chain: ingest → features → sentiment → ml pipeline
        job_ids = {}

        ingest_task = run_ingestion_task.apply_async()
        job_ids["ingestion"] = str(ingest_task.id)
        logger.info("Queued ingestion task %s", ingest_task.id)

        features_task = calculate_technical_features.apply_async()
        job_ids["features"] = str(features_task.id)
        logger.info("Queued features task %s", features_task.id)

        sentiment_task = run_sentiment_analysis.apply_async()
        job_ids["sentiment"] = str(sentiment_task.id)
        logger.info("Queued sentiment task %s", sentiment_task.id)

        pipeline_task = run_ml_pipeline_task.apply_async()
        job_ids["ml_pipeline"] = str(pipeline_task.id)
        logger.info("Queued ML pipeline task %s", pipeline_task.id)

        return JsonResponse({"status": "queued", "task_ids": job_ids})
    except (ImportError, RuntimeError) as exc:  # pylint: disable=broad-except
        logger.exception("Failed to queue pipeline tasks: %s", exc)
        return JsonResponse({"status": "error", "message": str(exc)}, status=500)
