"""
apps/engine/ingestion.py – Angel One SmartAPI data ingestion with TOTP auth.

Responsibilities
----------------
1. Authenticate via SmartAPI using TOTP (pyotp).
2. Fetch historical OHLCV bars for a list of symbols.
3. UPSERT data into the StockHistory table (PostgreSQL ON CONFLICT … DO UPDATE).
4. Exponential back-off on transient API / network errors.

Usage (standalone or called from Celery tasks)::

    from apps.engine.ingestion import run_ingestion
    run_ingestion(symbols=["RELIANCE", "INFY"], interval="ONE_DAY", days=365)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pyotp
import requests
from django.conf import settings
from django.db import connection, transaction

from apps.engine.models import StockHistory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Map our internal interval strings to Angel One API interval strings
INTERVAL_MAP: Dict[str, str] = {
    "1m": "ONE_MINUTE",
    "5m": "FIVE_MINUTE",
    "15m": "FIFTEEN_MINUTE",
    "30m": "THIRTY_MINUTE",
    "1h": "ONE_HOUR",
    "1d": "ONE_DAY",
    "1w": "ONE_WEEK",
}

# SmartAPI base URL (REST)
SMART_API_BASE = "https://apiconnect.angelbroking.com"

# Exponential back-off defaults
INITIAL_BACKOFF: float = 2.0   # seconds
MAX_BACKOFF: float = 64.0      # seconds
MAX_RETRIES: int = 5


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

class SmartAPISession:
    """Thin wrapper around the Angel One SmartAPI REST endpoints."""

    def __init__(self) -> None:
        self.api_key: str = settings.ANGEL_ONE_API_KEY
        self.client_id: str = settings.ANGEL_ONE_CLIENT_ID
        self.password: str = settings.ANGEL_ONE_PASSWORD
        self.totp_secret: str = settings.ANGEL_ONE_TOTP_SECRET
        self.jwt_token: Optional[str] = None
        self.feed_token: Optional[str] = None
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-UserType": "USER",
                "X-SourceID": "WEB",
                "X-ClientLocalIP": "127.0.0.1",
                "X-ClientPublicIP": "127.0.0.1",
                "X-MACAddress": "fe80::1",
                "X-PrivateKey": self.api_key,
            }
        )

    def _generate_totp(self) -> str:
        """Generate a time-based one-time password."""
        return pyotp.TOTP(self.totp_secret).now()

    def login(self) -> None:
        """Authenticate and store the JWT for subsequent calls."""
        totp = self._generate_totp()
        payload = {
            "clientcode": self.client_id,
            "password": self.password,
            "totp": totp,
        }
        resp = self._post_with_backoff(
            f"{SMART_API_BASE}/rest/auth/angelbroking/user/v1/loginByPassword",
            json=payload,
        )
        data = resp.json()
        if not data.get("status"):
            raise RuntimeError(f"SmartAPI login failed: {data.get('message', data)}")
        self.jwt_token = data["data"]["jwtToken"]
        self.feed_token = data["data"]["feedToken"]
        self._session.headers["Authorization"] = f"Bearer {self.jwt_token}"
        logger.info("SmartAPI login successful for client %s", self.client_id)

    def get_historical_data(
        self,
        exchange: str,
        symbol_token: str,
        interval: str,
        from_date: str,
        to_date: str,
    ) -> List[Dict]:
        """
        Fetch OHLCV candles from the SmartAPI historical data endpoint.

        Parameters
        ----------
        exchange       : e.g. "NSE"
        symbol_token   : Angel One instrument token (e.g. "3045" for SBIN)
        interval       : one of the INTERVAL_MAP values, e.g. "ONE_DAY"
        from_date      : "YYYY-MM-DD HH:MM"
        to_date        : "YYYY-MM-DD HH:MM"

        Returns
        -------
        List of candle dicts: {timestamp, open, high, low, close, volume}
        """
        payload = {
            "exchange": exchange,
            "symboltoken": symbol_token,
            "interval": interval,
            "fromdate": from_date,
            "todate": to_date,
        }
        resp = self._post_with_backoff(
            f"{SMART_API_BASE}/rest/secure/angelbroking/historical/v1/getCandleData",
            json=payload,
        )
        data = resp.json()
        if not data.get("status"):
            raise RuntimeError(
                f"Historical data fetch failed for {symbol_token}: {data.get('message', data)}"
            )
        candles = data.get("data", [])
        return candles  # each candle: [timestamp, open, high, low, close, volume]

    def logout(self) -> None:
        """Invalidate the session on the server side."""
        try:
            self._post_with_backoff(
                f"{SMART_API_BASE}/rest/secure/angelbroking/user/v1/logout",
                json={"clientcode": self.client_id},
            )
            logger.info("SmartAPI logout successful")
        except requests.RequestException as exc:
            logger.warning("SmartAPI logout error (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post_with_backoff(self, url: str, **kwargs) -> requests.Response:
        """POST with exponential back-off on transient errors (5xx / network)."""
        backoff = INITIAL_BACKOFF
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self._session.post(url, timeout=30, **kwargs)
                if resp.status_code < 500:
                    return resp
                logger.warning(
                    "SmartAPI HTTP %s on attempt %d/%d – backing off %.1fs",
                    resp.status_code, attempt, MAX_RETRIES, backoff,
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                logger.warning(
                    "Network error on attempt %d/%d – %s – backing off %.1fs",
                    attempt, MAX_RETRIES, exc, backoff,
                )
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"SmartAPI call to {url} failed after {MAX_RETRIES} retries"
                )
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)


# ---------------------------------------------------------------------------
# Symbol → token resolution (static lookup + yfinance fallback)
# ---------------------------------------------------------------------------

# A minimal static map for common NSE instruments.
# In production, load the full instrument list from Angel One's scrip master.
_NSE_TOKEN_MAP: Dict[str, str] = {
    "RELIANCE": "2885",
    "INFY": "1594",
    "TCS": "11536",
    "HDFCBANK": "1333",
    "ICICIBANK": "4963",
    "SBIN": "3045",
    "BHARTIARTL": "10604",
    "WIPRO": "3787",
    "HINDUNILVR": "1394",
    "KOTAKBANK": "1922",
    "AXISBANK": "5900",
    "LTIM": "17818",
    "SUNPHARMA": "3351",
    "TITAN": "3506",
    "BAJFINANCE": "317",
}


def resolve_token(symbol: str, exchange: str = "NSE") -> str:
    """Return the Angel One instrument token for *symbol*."""
    if exchange == "NSE":
        token = _NSE_TOKEN_MAP.get(symbol.upper())
        if token:
            return token
    raise ValueError(
        f"Instrument token for {symbol} ({exchange}) not found in static map. "
        "Please update _NSE_TOKEN_MAP or implement dynamic lookup."
    )


# ---------------------------------------------------------------------------
# UPSERT helper
# ---------------------------------------------------------------------------

_UPSERT_SQL = """
INSERT INTO engine_stockhistory
    (symbol, exchange, interval, timestamp,
     open_price, high_price, low_price, close_price, volume,
     created_at, updated_at)
VALUES
    (%(symbol)s, %(exchange)s, %(interval)s, %(timestamp)s,
     %(open_price)s, %(high_price)s, %(low_price)s, %(close_price)s, %(volume)s,
     NOW(), NOW())
ON CONFLICT (symbol, exchange, interval, timestamp)
DO UPDATE SET
    open_price  = EXCLUDED.open_price,
    high_price  = EXCLUDED.high_price,
    low_price   = EXCLUDED.low_price,
    close_price = EXCLUDED.close_price,
    volume      = EXCLUDED.volume,
    updated_at  = NOW();
"""


def _upsert_candles(
    symbol: str,
    exchange: str,
    interval: str,
    candles: List[List],
) -> int:
    """
    Bulk-UPSERT candle rows into engine_stockhistory.

    Each candle from the SmartAPI is: [timestamp_str, open, high, low, close, volume]
    Timestamps come in IST; we store them as UTC-aware datetimes.

    Returns the number of rows processed.
    """
    rows = []
    for candle in candles:
        try:
            ts_str, open_p, high_p, low_p, close_p, vol = candle
            # SmartAPI returns timestamps as "YYYY-MM-DDTHH:MM+05:30" (ISO-8601)
            ts = datetime.fromisoformat(str(ts_str))
            if ts.tzinfo is None:
                # Assume IST (UTC+5:30) if no tz present
                ts = ts.replace(tzinfo=timezone(timedelta(hours=5, minutes=30)))
            ts_utc = ts.astimezone(timezone.utc)
            rows.append(
                {
                    "symbol": symbol,
                    "exchange": exchange,
                    "interval": interval,
                    "timestamp": ts_utc,
                    "open_price": float(open_p),
                    "high_price": float(high_p),
                    "low_price": float(low_p),
                    "close_price": float(close_p),
                    "volume": int(vol),
                }
            )
        except (ValueError, TypeError) as exc:
            logger.warning("Skipping malformed candle %s – %s", candle, exc)

    if not rows:
        return 0

    with transaction.atomic():
        with connection.cursor() as cursor:
            cursor.executemany(_UPSERT_SQL, rows)

    logger.info("Upserted %d candles for %s [%s]", len(rows), symbol, interval)
    return len(rows)


# ---------------------------------------------------------------------------
# yfinance fallback fetcher
# ---------------------------------------------------------------------------

def _fetch_yfinance(
    symbol: str,
    interval: str,
    days: int,
) -> int:
    """
    Fallback: fetch OHLCV from yfinance and UPSERT into the DB.
    The yfinance ticker format for NSE is "<SYMBOL>.NS".
    """
    import yfinance as yf  # lazy import – optional dependency

    yf_interval_map = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "1d": "1d", "1w": "1wk",
    }
    yf_interval = yf_interval_map.get(interval, "1d")
    yf_period = f"{days}d"
    ticker = f"{symbol.upper()}.NS"

    logger.info("yfinance fallback: downloading %s interval=%s period=%s", ticker, yf_interval, yf_period)
    df = yf.download(ticker, period=yf_period, interval=yf_interval, progress=False, auto_adjust=True)

    if df.empty:
        logger.warning("yfinance returned no data for %s", ticker)
        return 0

    df = df.reset_index()
    candles = []
    ts_col = "Datetime" if "Datetime" in df.columns else "Date"
    for _, row in df.iterrows():
        ts = row[ts_col]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        if hasattr(ts, "tzinfo") and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        candles.append([ts, row["Open"], row["High"], row["Low"], row["Close"], row.get("Volume", 0)])

    return _upsert_candles(symbol, "NSE", interval, candles)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_ingestion(
    symbols: List[str],
    interval: str = "1d",
    days: int = 365,
    use_smartapi: bool = True,
    exchange: str = "NSE",
) -> Dict[str, int]:
    """
    Fetch OHLCV data for *symbols* and upsert into StockHistory.

    Parameters
    ----------
    symbols       : list of NSE ticker symbols, e.g. ["RELIANCE", "INFY"]
    interval      : one of "1m","5m","15m","30m","1h","1d","1w"
    days          : lookback window in calendar days
    use_smartapi  : if True and credentials present, use Angel One SmartAPI;
                    otherwise fall back to yfinance
    exchange      : exchange code ("NSE" by default)

    Returns
    -------
    Dict mapping symbol → number of rows upserted
    """
    results: Dict[str, int] = {}
    has_credentials = all(
        [
            settings.ANGEL_ONE_API_KEY,
            settings.ANGEL_ONE_CLIENT_ID,
            settings.ANGEL_ONE_PASSWORD,
            settings.ANGEL_ONE_TOTP_SECRET,
        ]
    )

    smart_interval = INTERVAL_MAP.get(interval)
    if not smart_interval:
        raise ValueError(f"Unsupported interval: {interval}. Choose from {list(INTERVAL_MAP)}")

    session: Optional[SmartAPISession] = None

    if use_smartapi and has_credentials:
        session = SmartAPISession()
        session.login()

    try:
        to_dt = datetime.now(tz=timezone.utc)
        from_dt = to_dt - timedelta(days=days)
        from_str = from_dt.strftime("%Y-%m-%d %H:%M")
        to_str = to_dt.strftime("%Y-%m-%d %H:%M")

        for symbol in symbols:
            if session is not None:
                try:
                    token = resolve_token(symbol, exchange)
                    candles = session.get_historical_data(
                        exchange=exchange,
                        symbol_token=token,
                        interval=smart_interval,
                        from_date=from_str,
                        to_date=to_str,
                    )
                    count = _upsert_candles(symbol, exchange, interval, candles)
                    results[symbol] = count
                    continue
                except (ValueError, RuntimeError) as exc:
                    logger.warning(
                        "SmartAPI fetch failed for %s (%s) – falling back to yfinance",
                        symbol, exc,
                    )

            # yfinance fallback
            count = _fetch_yfinance(symbol, interval, days)
            results[symbol] = count

    finally:
        if session is not None:
            session.logout()

    return results
