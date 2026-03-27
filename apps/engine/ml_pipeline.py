"""
apps/engine/ml_pipeline.py – Core ML pipeline for the Portfolio Predictive Engine.

Architecture
------------
1. **PyTorch LSTM** – learns temporal price patterns from OHLCV + technical features.
2. **XGBoost**      – gradient-boosted trees trained on the same tabular features.
3. **FinBERT**      – pre-trained sentiment scores aggregated per symbol per day.
4. **Meta-Model**   – Logistic Regression fuses the three component outputs into
                      a Buy / Hold / Sell signal.

Anti-leakage rules (strictly enforced)
---------------------------------------
- Scalers are **fitted only on the training split** and applied to the validation
  and test splits using ``transform`` (never ``fit_transform``).
- Walk-forward cross-validation respects temporal order (no shuffling).
- Feature engineering (EMA, RSI, …) was already computed chronologically in
  ``tasks.calculate_technical_features`` and stored in the DB.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEQUENCE_LEN = 60          # number of past bars fed to the LSTM
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 50
LSTM_BATCH = 64
LSTM_LR = 1e-3

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "mlogloss",
    "use_label_encoder": False,
    "random_state": 42,
    "n_jobs": -1,
}

META_MODEL_PARAMS = {
    "max_iter": 1000,
    "solver": "lbfgs",
    "multi_class": "multinomial",
    "C": 1.0,
    "random_state": 42,
}

LABEL_BUY  = 2
LABEL_HOLD = 1
LABEL_SELL = 0
LABEL_NAMES = {LABEL_BUY: "BUY", LABEL_HOLD: "HOLD", LABEL_SELL: "SELL"}

FEATURE_COLS = [
    "open_price", "high_price", "low_price", "close_price", "volume",
    "ema_9", "ema_21", "ema_50", "ema_200",
    "rsi_14",
    "macd", "macd_signal", "macd_histogram",
    "bb_upper", "bb_middle", "bb_lower", "bb_percent",
    "atr_14", "obv",
    "stoch_k", "stoch_d",
    "williams_r", "cci_20",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_symbol_data(symbol: str) -> Optional[pd.DataFrame]:
    """
    Load StockHistory + TechnicalFeatures from the DB for *symbol* (daily bars).
    Returns a DataFrame sorted by timestamp with all FEATURE_COLS present,
    or None if there is insufficient data.
    """
    from apps.engine.models import StockHistory

    qs = (
        StockHistory.objects.filter(symbol=symbol, interval="1d")
        .select_related("technical_features")
        .order_by("timestamp")
        .values(
            "timestamp",
            "open_price", "high_price", "low_price", "close_price", "volume",
            "technical_features__ema_9",   "technical_features__ema_21",
            "technical_features__ema_50",  "technical_features__ema_200",
            "technical_features__rsi_14",
            "technical_features__macd",    "technical_features__macd_signal",
            "technical_features__macd_histogram",
            "technical_features__bb_upper","technical_features__bb_middle",
            "technical_features__bb_lower","technical_features__bb_percent",
            "technical_features__atr_14",  "technical_features__obv",
            "technical_features__stoch_k", "technical_features__stoch_d",
            "technical_features__williams_r", "technical_features__cci_20",
        )
    )

    rows = list(qs)
    if len(rows) < SEQUENCE_LEN + 50:
        logger.warning("%s: not enough data (%d rows)", symbol, len(rows))
        return None

    df = pd.DataFrame(rows)
    # Flatten prefixed column names
    df.columns = [c.replace("technical_features__", "") for c in df.columns]
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return df


def _load_sentiment(symbol: str, df_prices: pd.DataFrame) -> pd.Series:
    """
    For each date in *df_prices*, compute the aggregate sentiment score
    (positive_score − negative_score) averaged over headlines published
    on or before that date (trailing 7-day window).

    Returns a Series aligned to df_prices.index.
    """
    from apps.engine.models import NewsSentiment

    qs = NewsSentiment.objects.filter(symbol=symbol).values(
        "published_at", "positive_score", "negative_score"
    )
    sent_rows = list(qs)
    if not sent_rows:
        return pd.Series(0.0, index=df_prices.index)

    sent_df = pd.DataFrame(sent_rows)
    sent_df["date"] = pd.to_datetime(sent_df["published_at"]).dt.tz_localize(None).dt.normalize()
    sent_df["score"] = sent_df["positive_score"] - sent_df["negative_score"]
    daily_sent = sent_df.groupby("date")["score"].mean()

    price_dates = pd.to_datetime(df_prices["timestamp"]).dt.tz_localize(None).dt.normalize()
    scores = []
    for date in price_dates:
        window = daily_sent.loc[
            (daily_sent.index >= date - pd.Timedelta(days=7)) &
            (daily_sent.index <= date)
        ]
        scores.append(float(window.mean()) if not window.empty else 0.0)
    return pd.Series(scores, index=df_prices.index)


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def _make_labels(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.02) -> pd.Series:
    """
    Forward-looking return over *horizon* bars.
    - return > +threshold  → BUY  (2)
    - return < -threshold  → SELL (0)
    - otherwise            → HOLD (1)

    The last *horizon* rows get label HOLD (not used in training).
    """
    close = df["close_price"].astype(float)
    future_ret = close.shift(-horizon) / close - 1.0
    labels = np.where(future_ret > threshold, LABEL_BUY,
                      np.where(future_ret < -threshold, LABEL_SELL, LABEL_HOLD))
    return pd.Series(labels.astype(int), index=df.index)


# ---------------------------------------------------------------------------
# PyTorch LSTM
# ---------------------------------------------------------------------------

def _build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slide a window of length *seq_len* over X/y to create 3-D sequences."""
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i: i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int64)


class _LSTMClassifier:
    """Thin scikit-learn-style wrapper around a PyTorch LSTM."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = LSTM_HIDDEN,
        num_layers: int = LSTM_LAYERS,
        dropout: float = LSTM_DROPOUT,
        num_classes: int = 3,
        epochs: int = LSTM_EPOCHS,
        batch_size: int = LSTM_BATCH,
        lr: float = LSTM_LR,
    ) -> None:
        import torch
        import torch.nn as nn

        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True,
                )
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_size, num_classes)

            def forward(self, x):  # type: ignore[override]
                out, _ = self.lstm(x)
                out = self.dropout(out[:, -1, :])
                return self.fc(out)

        self.net = _Net().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LSTMClassifier":
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=False,
        )
        self.net.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.net(xb), yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.debug("LSTM epoch %d/%d loss=%.4f", epoch + 1, self.epochs, total_loss / len(loader))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        self.net.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.net(X_t)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs  # shape (N, 3)


# ---------------------------------------------------------------------------
# Walk-forward cross-validation helper
# ---------------------------------------------------------------------------

def _walk_forward_splits(
    n: int,
    n_splits: int = 5,
    min_train: float = 0.5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Produce *n_splits* expanding-window (train, test) index pairs.

    The first fold trains on the earliest ``min_train`` fraction and tests
    on the next chunk.  Each subsequent fold expands the training window.
    """
    indices = np.arange(n)
    min_train_size = int(n * min_train)
    step = (n - min_train_size) // (n_splits + 1)
    splits = []
    for k in range(n_splits):
        train_end = min_train_size + k * step
        test_end = train_end + step
        if test_end > n:
            break
        splits.append((indices[:train_end], indices[train_end:test_end]))
    return splits


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def _train_symbol(symbol: str) -> Optional[dict]:
    """
    Train all three component models plus the meta-model for *symbol*.

    Returns a dict with the latest prediction data, or None on failure.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    import xgboost as xgb

    # 1. Load data ---------------------------------------------------------
    df = _load_symbol_data(symbol)
    if df is None:
        return None

    sentiment_scores = _load_sentiment(symbol, df)
    labels = _make_labels(df)

    # Drop last rows where label is unreliable (no future data)
    horizon = 5
    df = df.iloc[:-horizon].copy()
    labels = labels.iloc[:-horizon].copy()
    sentiment_scores = sentiment_scores.iloc[:-horizon].copy()

    X_raw = df[FEATURE_COLS].values.astype(np.float32)
    y = labels.values.astype(int)
    n = len(X_raw)

    if n < SEQUENCE_LEN + 50:
        logger.warning("%s: insufficient rows after cleaning (%d)", symbol, n)
        return None

    # 2. Walk-forward training ---------------------------------------------
    splits = _walk_forward_splits(n, n_splits=5, min_train=0.6)
    if not splits:
        logger.warning("%s: no valid walk-forward splits", symbol)
        return None

    # Use the last split as the final evaluation set
    train_idx, test_idx = splits[-1]

    # 3. Scale – fit ONLY on train split -----------------------------------
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_raw[train_idx])
    X_test_scaled = scaler.transform(X_raw[test_idx])
    X_all_scaled = scaler.transform(X_raw)

    y_train = y[train_idx]
    y_test = y[test_idx]

    # 4. LSTM --------------------------------------------------------------
    logger.info("%s: training LSTM…", symbol)
    X_seq_train, y_seq_train = _build_sequences(X_train_scaled, y_train, SEQUENCE_LEN)
    if len(X_seq_train) == 0:
        logger.warning("%s: LSTM sequence generation produced 0 samples", symbol)
        return None

    lstm = _LSTMClassifier(input_size=X_raw.shape[1])
    lstm.fit(X_seq_train, y_seq_train)

    # LSTM probabilities on entire dataset (needed for meta-model features)
    X_seq_all, y_seq_all = _build_sequences(X_all_scaled, y, SEQUENCE_LEN)
    lstm_proba_all = lstm.predict_proba(X_seq_all)  # shape (M, 3)
    # Align with main dataframe (LSTM "skips" first SEQUENCE_LEN rows)
    lstm_offset = SEQUENCE_LEN

    # 5. XGBoost -----------------------------------------------------------
    logger.info("%s: training XGBoost…", symbol)
    xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False,
    )
    xgb_proba_all = xgb_model.predict_proba(X_all_scaled)  # shape (N, 3)

    # 6. Assemble meta-model features  -------------------------------------
    # Align LSTM probabilities with XGBoost (LSTM is shorter by lstm_offset)
    meta_start = lstm_offset
    meta_n = len(X_seq_all)

    lstm_proba_aligned = lstm_proba_all  # (meta_n, 3)
    xgb_proba_aligned  = xgb_proba_all[meta_start: meta_start + meta_n]  # (meta_n, 3)
    sent_aligned       = sentiment_scores.values[meta_start: meta_start + meta_n]
    y_aligned          = y[meta_start: meta_start + meta_n]

    # Meta features: [lstm_buy, lstm_hold, lstm_sell,
    #                 xgb_buy,  xgb_hold,  xgb_sell,
    #                 sentiment_score]
    meta_X = np.column_stack([
        lstm_proba_aligned,
        xgb_proba_aligned,
        sent_aligned.reshape(-1, 1),
    ])

    # Walk-forward split indices for meta-model (relative to meta_X)
    meta_train_end = int(len(meta_X) * 0.8)
    meta_X_train = meta_X[:meta_train_end]
    meta_y_train = y_aligned[:meta_train_end]
    meta_X_test  = meta_X[meta_train_end:]
    meta_y_test  = y_aligned[meta_train_end:]

    # 7. Logistic Regression meta-model ------------------------------------
    logger.info("%s: training Logistic Regression meta-model…", symbol)
    meta_model = LogisticRegression(**META_MODEL_PARAMS)
    meta_model.fit(meta_X_train, meta_y_train)

    meta_preds = meta_model.predict(meta_X_test)
    logger.info(
        "%s: meta-model test classification report:\n%s",
        symbol,
        classification_report(
            meta_y_test, meta_preds,
            target_names=["SELL", "HOLD", "BUY"],
            zero_division=0,
        ),
    )

    # 8. Latest prediction -------------------------------------------------
    # Use the very last available bar
    last_lstm_proba  = lstm_proba_all[-1]   # shape (3,)
    last_xgb_proba   = xgb_proba_all[-1]    # shape (3,)
    last_sentiment   = float(sentiment_scores.values[-1])

    last_meta_X = np.array([[
        *last_lstm_proba,
        *last_xgb_proba,
        last_sentiment,
    ]])
    last_proba  = meta_model.predict_proba(last_meta_X)[0]   # shape (3,)
    last_signal_idx = int(np.argmax(last_proba))
    # meta_model classes_ order may differ – use label encoder
    class_order = list(meta_model.classes_)  # e.g. [0, 1, 2]
    buy_prob  = float(last_proba[class_order.index(LABEL_BUY)])
    hold_prob = float(last_proba[class_order.index(LABEL_HOLD)])
    sell_prob = float(last_proba[class_order.index(LABEL_SELL)])
    signal    = LABEL_NAMES[class_order[last_signal_idx]]

    return {
        "symbol": symbol,
        "predicted_at": datetime.now(tz=timezone.utc),
        "lstm_confidence": float(last_lstm_proba[LABEL_BUY]),
        "xgb_confidence": float(last_xgb_proba[LABEL_BUY]),
        "sentiment_score": last_sentiment,
        "signal": signal,
        "buy_probability": buy_prob,
        "hold_probability": hold_prob,
        "sell_probability": sell_prob,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_pipeline(symbols: List[str]) -> Dict[str, str]:
    """
    Run the full ML pipeline for each symbol in *symbols*, persist Prediction
    rows, and return a summary dict mapping symbol → signal.

    Designed to be called from ``tasks.run_ml_pipeline_task``.
    """
    from apps.engine.models import Prediction

    summary: Dict[str, str] = {}

    for symbol in symbols:
        logger.info("=" * 60)
        logger.info("Running ML pipeline for %s", symbol)
        try:
            result = _train_symbol(symbol)
            if result is None:
                summary[symbol] = "SKIPPED"
                continue

            Prediction.objects.create(**result)
            summary[symbol] = result["signal"]
            logger.info("%s → %s (buy=%.2f sell=%.2f)", symbol, result["signal"],
                        result["buy_probability"], result["sell_probability"])
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Pipeline failed for %s: %s", symbol, exc)
            summary[symbol] = "ERROR"

    return summary
