"""Production-grade ASX backtesting Streamlit application with self-tests."""
from __future__ import annotations

import io
import math
import zipfile
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from typing import Callable, Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover - allow running tests without Altair installed
    import altair as alt
except ModuleNotFoundError:  # pragma: no cover
    from types import SimpleNamespace

    class _FallbackChart:
        """Minimal Altair-compatible chart used when Altair is unavailable."""

        def __init__(self, data=None):
            self._spec: Dict[str, object] = {"encoding": {}}
            self._spec["data"] = {"values": self._to_records(data)} if data is not None else None

        @staticmethod
        def _to_records(data):
            if hasattr(data, "to_dict"):
                try:
                    return data.to_dict(orient="records")  # type: ignore[arg-type]
                except TypeError:
                    return data.to_dict()
            return data

        def mark_line(self, **kwargs):
            self._spec["mark"] = {"type": "line", **kwargs}
            return self

        def mark_area(self, **kwargs):
            self._spec["mark"] = {"type": "area", **kwargs}
            return self

        def mark_point(self, **kwargs):
            self._spec["mark"] = {"type": "point", **kwargs}
            return self

        def mark_bar(self, **kwargs):
            self._spec["mark"] = {"type": "bar", **kwargs}
            return self

        def encode(self, **kwargs):
            self._spec.setdefault("encoding", {}).update(kwargs)
            return self

        def interactive(self):
            self._spec["interactive"] = True
            return self

        def resolve_scale(self, **kwargs):
            resolve = self._spec.setdefault("resolve", {})
            scale = resolve.setdefault("scale", {})
            scale.update(kwargs)
            return self

        def transform_fold(self, fields, as_):
            self._spec.setdefault("transform", []).append({"fold": fields, "as": list(as_)})
            return self

        def to_dict(self):
            return dict(self._spec)

    def _fallback_chart(data=None):
        return _FallbackChart(data)

    def _fallback_layer(*charts):
        chart = _FallbackChart()
        chart._spec = {"layer": [c.to_dict() for c in charts]}
        return chart

    alt = SimpleNamespace(  # type: ignore[assignment]
        Chart=_fallback_chart,
        layer=_fallback_layer,
        Tooltip=lambda field, **kwargs: {"field": field, **kwargs},
        Y=lambda field, **kwargs: {"field": field, **kwargs},
        Color=lambda field, **kwargs: {"field": field, **kwargs},
        Axis=lambda **kwargs: kwargs,
        Scale=lambda **kwargs: kwargs,
        Legend=lambda **kwargs: kwargs,
    )

import numpy as np
import pandas as pd
import streamlit as st
import ta
import yfinance as yf


st.set_page_config(page_title="ASX Backtester", layout="wide")

COMMON_TICKERS: List[str] = [
    "BHP.AX",
    "CBA.AX",
    "NAB.AX",
    "WBC.AX",
    "ANZ.AX",
    "CSL.AX",
    "WES.AX",
    "WOW.AX",
    "FMG.AX",
    "TLS.AX",
]

DEFAULT_BENCHMARKS: List[str] = ["XJO.AX", "XAO.AX", "STW.AX"]

STRATEGY_OPTIONS: List[str] = [
    "Buy and Hold",
    "Moving Average Crossover (13/55)",
    "RSI Strategy",
    "MACD Strategy",
    "Bollinger Bands",
    "Momentum",
    "Mean Reversion",
    "Golden/Death Cross",
    "Breakout",
]


@dataclass
class StrategyParameters:
    """Container for user adjustable strategy parameters."""

    short_window: int = 13
    long_window: int = 55
    rsi_lower: float = 30.0
    rsi_upper: float = 70.0
    bollinger_window: int = 20
    bollinger_width: float = 2.0
    momentum_window: int = 12
    mean_reversion_window: int = 20
    breakout_lookback: int = 20

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _sanitize_parameters(params: Optional[StrategyParameters]) -> StrategyParameters:
    """Ensure provided parameters fall within safe, sensible ranges."""

    if params is None:
        params = StrategyParameters()

    short_window = max(1, int(params.short_window))
    long_window = max(short_window + 1, int(params.long_window))

    rsi_lower = max(0.0, float(params.rsi_lower))
    rsi_upper = min(100.0, float(params.rsi_upper))
    if rsi_lower >= rsi_upper:
        midpoint = max(1.0, (rsi_lower + rsi_upper) / 2.0)
        rsi_lower = max(0.0, midpoint - 10.0)
        rsi_upper = min(100.0, midpoint + 10.0)

    bollinger_window = max(1, int(params.bollinger_window))
    bollinger_width = max(0.1, float(params.bollinger_width))

    momentum_window = max(1, int(params.momentum_window))
    mean_reversion_window = max(1, int(params.mean_reversion_window))
    breakout_lookback = max(1, int(params.breakout_lookback))

    return StrategyParameters(
        short_window=short_window,
        long_window=long_window,
        rsi_lower=rsi_lower,
        rsi_upper=rsi_upper,
        bollinger_window=bollinger_window,
        bollinger_width=bollinger_width,
        momentum_window=momentum_window,
        mean_reversion_window=mean_reversion_window,
        breakout_lookback=breakout_lookback,
    )


def _ensure_price_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Ensure essential OHLCV columns exist and contain numeric data."""

    if data is None or data.empty:
        raise ValueError("Ticker returned no price data.")

    df = data.copy()
    df.columns = [str(column) for column in df.columns]

    close_col = "Close"
    adj_close_col = "Adj Close"
    if adj_close_col not in df.columns:
        if close_col not in df.columns:
            raise ValueError("Ticker returned no price data.")
        df[adj_close_col] = df[close_col]
    if close_col not in df.columns:
        df[close_col] = df[adj_close_col]

    if "High" not in df.columns:
        df["High"] = df[[adj_close_col, close_col]].max(axis=1)
    if "Low" not in df.columns:
        df["Low"] = df[[adj_close_col, close_col]].min(axis=1)
    if "Open" not in df.columns:
        df["Open"] = df[close_col]
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    numeric_columns = ["Open", "High", "Low", close_col, adj_close_col, "Volume"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    price_columns = ["Open", "High", "Low", close_col, adj_close_col]
    df[price_columns] = df[price_columns].ffill().bfill()
    df["Volume"] = df["Volume"].fillna(0.0)

    if df[price_columns].isna().any().any():
        raise ValueError("Ticker returned no price data.")

    return df


def _force_series(values, index) -> pd.Series:
    """Coerce any input into a 1D float Series."""

    if isinstance(values, pd.Series):
        return values.astype(float)

    arr = np.asarray(values).ravel()
    return pd.Series(arr, index=index, dtype=float)


def _clean_numeric_series(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    if not isinstance(series, pd.Series):
        if hasattr(series, "index"):
            series = _force_series(series, series.index)
        else:
            raise TypeError("Input must be a pandas Series or provide an index attribute.")

    cleaned = series.replace([np.inf, -np.inf], np.nan)
    if fill_value is not None:
        cleaned = cleaned.fillna(fill_value)
    return cleaned.astype(float)


def _compute_equity_curve(returns: pd.Series) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)

    clean_returns = _clean_numeric_series(returns, fill_value=0.0).copy()
    if not clean_returns.empty:
        clean_returns.iloc[0] = 0.0

    factors = 1.0 + clean_returns
    factors = factors.clip(lower=1e-9)
    equity = factors.cumprod()
    equity = equity.replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
    return _force_series(equity, clean_returns.index)


def _format_percentage(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def _format_ratio(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.2f}"


def _safe_value(value: Optional[float]) -> float:
    if value is None or not np.isfinite(value):
        return float("nan")
    return float(value)


def get_selected_ticker(default_choice: str, custom_input: str) -> str:
    custom_input = custom_input.strip().upper()
    if custom_input:
        return custom_input
    return default_choice.strip().upper()


def download_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    if not ticker:
        raise ValueError("A ticker symbol must be provided.")
    if start >= end:
        raise ValueError("Start date must be before end date.")

    raw = yf.download(
        ticker,
        start=start,
        end=end + timedelta(days=1),
        progress=False,
        auto_adjust=False,
        group_by="column",
    )
    if raw.empty:
        raise ValueError("Ticker returned no price data.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [str(col[0]) for col in raw.columns]

    raw.columns = [str(column) for column in raw.columns]

    if "Adj Close" not in raw.columns and "Close" in raw.columns:
        raw["Adj Close"] = raw["Close"]

    raw = raw.dropna(how="all")
    if raw.empty:
        raise ValueError("Ticker returned no price data.")

    raw.sort_index(inplace=True)
    raw.index = pd.to_datetime(raw.index)

    try:
        return _ensure_price_columns(raw)
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError("Ticker returned no price data.") from exc


StrategyHandler = Callable[[pd.DataFrame, StrategyParameters], Tuple[pd.Series, Dict[str, pd.Series]]]


def _strategy_buy_and_hold(df: pd.DataFrame, _: StrategyParameters) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    signals = pd.Series(1.0, index=df.index, dtype=float)
    return signals, {}


def _strategy_moving_average(df: pd.DataFrame, params: StrategyParameters) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    short = _force_series(
        df["Adj Close"].rolling(window=params.short_window, min_periods=1).mean(),
        df.index,
    )
    long = _force_series(
        df["Adj Close"].rolling(window=params.long_window, min_periods=1).mean(),
        df.index,
    )
    signals = _force_series(np.where(short > long, 1.0, 0.0), df.index)
    return signals, {"SMA_Short": short, "SMA_Long": long}


def _strategy_rsi(df: pd.DataFrame, params: StrategyParameters) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    rsi_indicator = ta.momentum.RSIIndicator(close=df["Adj Close"], window=14, fillna=True)
    rsi = _force_series(rsi_indicator.rsi().clip(0, 100), df.index)
    raw_signal = np.where(
        rsi < params.rsi_lower,
        1.0,
        np.where(rsi > params.rsi_upper, 0.0, np.nan),
    )
    signals = _force_series(raw_signal, df.index).ffill().fillna(0.0)
    return signals, {"RSI": rsi}


def _strategy_macd(df: pd.DataFrame, _: StrategyParameters) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    macd_indicator = ta.trend.MACD(close=df["Adj Close"], window_slow=26, window_fast=12, window_sign=9, fillna=True)
    macd = _force_series(macd_indicator.macd(), df.index)
    macd_signal = _force_series(macd_indicator.macd_signal(), df.index)
    macd_hist = _force_series(macd_indicator.macd_diff(), df.index)
    signals = _force_series(np.where(macd_hist > 0, 1.0, 0.0), df.index)
    return signals, {"MACD": macd, "MACD_Signal": macd_signal, "MACD_Hist": macd_hist}


def _strategy_bollinger(df: pd.DataFrame, params: StrategyParameters) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    bb_indicator = ta.volatility.BollingerBands(
        close=df["Adj Close"],
        window=params.bollinger_window,
        window_dev=params.bollinger_width,
        fillna=True,
    )
    middle = _force_series(bb_indicator.bollinger_mavg(), df.index)
    upper = _force_series(bb_indicator.bollinger_hband(), df.index)
    lower = _force_series(bb_indicator.bollinger_lband(), df.index)
    long_signal = df["Adj Close"] < lower
    flat_signal = df["Adj Close"] > upper
    raw_signal = np.where(long_signal, 1.0, np.where(flat_signal, 0.0, np.nan))
    signals = _force_series(raw_signal, df.index).ffill().fillna(0.0)
    return signals, {"BB_Middle": middle, "BB_Upper": upper, "BB_Lower": lower}


def _strategy_momentum(df: pd.DataFrame, params: StrategyParameters) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    roc_indicator = ta.momentum.ROCIndicator(close=df["Adj Close"], window=params.momentum_window, fillna=True)
    roc = _force_series(roc_indicator.roc(), df.index)
    signals = _force_series(np.where(roc > 0, 1.0, 0.0), df.index)
    return signals, {"ROC": roc}


def _strategy_mean_reversion(df: pd.DataFrame, params: StrategyParameters) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    sma = _force_series(
        df["Adj Close"].rolling(window=params.mean_reversion_window, min_periods=1).mean(),
        df.index,
    )
    signals = _force_series(np.where(df["Adj Close"] < sma, 1.0, 0.0), df.index)
    return signals, {"MeanReversionSMA": sma}


def _strategy_golden_cross(df: pd.DataFrame, _: StrategyParameters) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    sma_50 = _force_series(df["Adj Close"].rolling(window=50, min_periods=1).mean(), df.index)
    sma_200 = _force_series(df["Adj Close"].rolling(window=200, min_periods=1).mean(), df.index)
    signals = _force_series(np.where(sma_50 > sma_200, 1.0, 0.0), df.index)
    return signals, {"SMA_50": sma_50, "SMA_200": sma_200}


def _strategy_breakout(df: pd.DataFrame, params: StrategyParameters) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    high = _force_series(
        df["High"].rolling(window=params.breakout_lookback, min_periods=1).max(),
        df.index,
    )
    low = _force_series(
        df["Low"].rolling(window=params.breakout_lookback, min_periods=1).min(),
        df.index,
    )
    raw_signal = np.where(
        df["Close"] > high.shift(1),
        1.0,
        np.where(df["Close"] < low.shift(1), 0.0, np.nan),
    )
    signals = _force_series(raw_signal, df.index).ffill().fillna(0.0)
    return signals, {"Breakout_High": high, "Breakout_Low": low}


STRATEGY_REGISTRY: Dict[str, StrategyHandler] = {
    "Buy and Hold": _strategy_buy_and_hold,
    "Moving Average Crossover (13/55)": _strategy_moving_average,
    "RSI Strategy": _strategy_rsi,
    "MACD Strategy": _strategy_macd,
    "Bollinger Bands": _strategy_bollinger,
    "Momentum": _strategy_momentum,
    "Mean Reversion": _strategy_mean_reversion,
    "Golden/Death Cross": _strategy_golden_cross,
    "Breakout": _strategy_breakout,
}


def apply_strategy(
    data: pd.DataFrame, strategy: str, params: Optional[StrategyParameters] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    if data is None or data.empty:
        raise ValueError("Input price data must not be empty.")

    df = _ensure_price_columns(data)
    df = df.sort_index().copy()
    df["Adj Close"] = pd.to_numeric(df["Adj Close"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["High"] = pd.to_numeric(df["High"], errors="coerce")
    df["Low"] = pd.to_numeric(df["Low"], errors="coerce")
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)

    params = _sanitize_parameters(params)

    df["Return"] = _force_series(_clean_numeric_series(df["Adj Close"].pct_change(), fill_value=0.0), df.index)

    handler = STRATEGY_REGISTRY.get(strategy.strip())
    if handler is None:
        raise ValueError(f"Unsupported strategy: {strategy}")

    signals, extra_columns = handler(df, params)
    signals = _force_series(signals, df.index)
    signals = signals.astype(float).clip(lower=0.0, upper=1.0)
    signals = signals.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    positions = _force_series(signals.shift(1).fillna(0.0), df.index)
    strategy_returns = _force_series(
        _clean_numeric_series(positions * df["Return"], fill_value=0.0),
        df.index,
    )

    equity_curve = _compute_equity_curve(strategy_returns)
    buy_hold_curve = _compute_equity_curve(df["Return"])

    results = df.copy()
    results["Adj Close"] = _force_series(results["Adj Close"], df.index)
    results["Close"] = _force_series(results["Close"], df.index)
    results["High"] = _force_series(results["High"], df.index)
    results["Low"] = _force_series(results["Low"], df.index)
    results["Open"] = _force_series(results["Open"], df.index)
    results["Volume"] = _force_series(results["Volume"], df.index)
    results["Return"] = _force_series(results["Return"], df.index)
    for name, values in extra_columns.items():
        results[name] = _force_series(values, df.index)
    results["Signal"] = _force_series(signals, df.index)
    results["Position"] = _force_series(positions, df.index)
    results["Strategy Return"] = _force_series(strategy_returns, df.index)
    results["Equity Curve"] = _force_series(equity_curve, df.index)
    results["Buy & Hold Equity"] = _force_series(buy_hold_curve, df.index)
    results.attrs["strategy_parameters"] = params

    return results, equity_curve, buy_hold_curve


def compute_metrics(
    equity_curve: pd.Series,
    buy_hold_curve: pd.Series,
    strategy_returns: pd.Series,
    benchmark_curve: Optional[pd.Series] = None,
) -> pd.DataFrame:
    if equity_curve.empty or buy_hold_curve.empty:
        raise ValueError("Equity curves are empty; cannot compute performance metrics.")

    equity_clean = equity_curve.replace([np.inf, -np.inf], np.nan).ffill().copy()
    buy_hold_clean = buy_hold_curve.replace([np.inf, -np.inf], np.nan).ffill().copy()

    if equity_clean.isna().all() or buy_hold_clean.isna().all():
        raise ValueError("Equity curves contain no valid data points.")

    final_equity = equity_clean.dropna().iloc[-1]
    final_buy_hold = buy_hold_clean.dropna().iloc[-1]

    total_return = final_equity - 1.0 if np.isfinite(final_equity) else np.nan
    buy_hold_return = final_buy_hold - 1.0 if np.isfinite(final_buy_hold) else np.nan

    trading_days = float(len(equity_clean.dropna()))
    years = trading_days / 252.0 if trading_days > 0 else np.nan
    if years is None or not np.isfinite(years) or years <= 0 or final_equity <= 0:
        cagr = np.nan
    else:
        cagr = final_equity ** (1.0 / years) - 1.0

    running_max = equity_clean.cummax().replace(0, np.nan)
    drawdown = (equity_clean / running_max) - 1.0
    drawdown = drawdown.replace([np.inf, -np.inf], np.nan)
    max_drawdown = drawdown.min(skipna=True)

    cleaned_returns = _clean_numeric_series(strategy_returns, fill_value=0.0)
    active_returns = cleaned_returns[np.abs(cleaned_returns) > 1e-12]
    win_rate = (active_returns > 0).mean() if not active_returns.empty else np.nan

    daily_std = cleaned_returns.std(ddof=0)
    volatility = daily_std * math.sqrt(252) if np.isfinite(daily_std) else np.nan
    if np.isfinite(daily_std) and daily_std > 0:
        sharpe = (cleaned_returns.mean() / daily_std) * math.sqrt(252)
    else:
        sharpe = np.nan

    metrics_data = [
        {
            "Metric": "Strategy Total Return",
            "RawValue": _safe_value(total_return),
            "Display": _format_percentage(total_return),
        },
        {
            "Metric": "Buy & Hold Total Return",
            "RawValue": _safe_value(buy_hold_return),
            "Display": _format_percentage(buy_hold_return),
        },
        {
            "Metric": "CAGR",
            "RawValue": _safe_value(cagr),
            "Display": _format_percentage(cagr),
        },
        {
            "Metric": "Max Drawdown",
            "RawValue": _safe_value(max_drawdown),
            "Display": _format_percentage(max_drawdown),
        },
        {
            "Metric": "Win Rate",
            "RawValue": _safe_value(win_rate),
            "Display": _format_percentage(win_rate),
        },
        {
            "Metric": "Annualized Volatility",
            "RawValue": _safe_value(volatility),
            "Display": _format_percentage(volatility),
        },
        {
            "Metric": "Sharpe Ratio",
            "RawValue": _safe_value(sharpe),
            "Display": _format_ratio(sharpe),
        },
    ]

    if benchmark_curve is not None and not benchmark_curve.empty:
        benchmark_clean = benchmark_curve.replace([np.inf, -np.inf], np.nan).ffill().dropna()
        if not benchmark_clean.empty:
            benchmark_return = benchmark_clean.iloc[-1] - 1.0
            metrics_data.append(
                {
                    "Metric": "Benchmark Total Return",
                    "RawValue": _safe_value(benchmark_return),
                    "Display": _format_percentage(benchmark_return),
                }
            )

    return pd.DataFrame(metrics_data)


def _add_indicator_layers(
    price_df: pd.DataFrame, strategy: str, params: StrategyParameters
) -> Tuple[List[alt.Chart], bool]:
    layers: List[alt.Chart] = []
    requires_secondary_axis = False

    if {"SMA_Short", "SMA_Long"}.issubset(price_df.columns):
        layers.append(
            alt.Chart(price_df)
            .mark_line(color="#ff7f0e", strokeDash=[4, 2])
            .encode(x="Date:T", y="SMA_Short:Q", tooltip=["Date:T", alt.Tooltip("SMA_Short:Q", format=".2f", title="SMA Short")])
        )
        layers.append(
            alt.Chart(price_df)
            .mark_line(color="#2ca02c", strokeDash=[2, 2])
            .encode(x="Date:T", y="SMA_Long:Q", tooltip=["Date:T", alt.Tooltip("SMA_Long:Q", format=".2f", title="SMA Long")])
        )

    if {"BB_Upper", "BB_Lower"}.issubset(price_df.columns):
        layers.append(
            alt.Chart(price_df)
            .mark_area(color="#c6dbef", opacity=0.3)
            .encode(x="Date:T", y="BB_Lower:Q", y2="BB_Upper:Q")
        )
        layers.append(
            alt.Chart(price_df)
            .mark_line(color="#9467bd")
            .encode(x="Date:T", y="BB_Middle:Q", tooltip=["Date:T", alt.Tooltip("BB_Middle:Q", format=".2f", title="BB Middle")])
        )

    if "ROC" in price_df.columns:
        requires_secondary_axis = True
        layers.append(
            alt.Chart(price_df)
            .mark_line(color="#17becf")
            .encode(
                x="Date:T",
                y=alt.Y(
                    "ROC:Q",
                    axis=alt.Axis(title="ROC", titleColor="#17becf"),
                    scale=alt.Scale(zero=False),
                ),
                tooltip=["Date:T", alt.Tooltip("ROC:Q", format=".2f")],
            )
        )

    if "MeanReversionSMA" in price_df.columns:
        layers.append(
            alt.Chart(price_df)
            .mark_line(color="#bcbd22", strokeDash=[5, 3])
            .encode(
                x="Date:T",
                y="MeanReversionSMA:Q",
                tooltip=["Date:T", alt.Tooltip("MeanReversionSMA:Q", format=".2f", title="Mean Reversion SMA")],
            )
        )

    if {"SMA_50", "SMA_200"}.issubset(price_df.columns):
        layers.append(
            alt.Chart(price_df)
            .mark_line(color="#ff9896", strokeDash=[4, 4])
            .encode(x="Date:T", y="SMA_50:Q", tooltip=["Date:T", alt.Tooltip("SMA_50:Q", format=".2f", title="50D SMA")])
        )
        layers.append(
            alt.Chart(price_df)
            .mark_line(color="#8c564b", strokeDash=[6, 3])
            .encode(x="Date:T", y="SMA_200:Q", tooltip=["Date:T", alt.Tooltip("SMA_200:Q", format=".2f", title="200D SMA")])
        )

    if {"Breakout_High", "Breakout_Low"}.issubset(price_df.columns):
        layers.append(
            alt.Chart(price_df)
            .mark_line(color="#1f77b4", strokeDash=[2, 1])
            .encode(x="Date:T", y="Breakout_High:Q", tooltip=["Date:T", alt.Tooltip("Breakout_High:Q", format=".2f", title="High Lookback")])
        )
        layers.append(
            alt.Chart(price_df)
            .mark_line(color="#d62728", strokeDash=[2, 1])
            .encode(x="Date:T", y="Breakout_Low:Q", tooltip=["Date:T", alt.Tooltip("Breakout_Low:Q", format=".2f", title="Low Lookback")])
        )

    if "RSI" in price_df.columns:
        requires_secondary_axis = True
        layers.append(
            alt.Chart(price_df)
            .mark_line(color="#d62728")
            .encode(
                x="Date:T",
                y=alt.Y(
                    "RSI:Q",
                    axis=alt.Axis(title="RSI", titleColor="#d62728"),
                    scale=alt.Scale(domain=[0, 100]),
                ),
                tooltip=["Date:T", alt.Tooltip("RSI:Q", format=".2f")],
            )
        )
        rsi_thresholds = pd.DataFrame({"RSI_Level": [params.rsi_lower, params.rsi_upper]})
        layers.append(
            alt.Chart(rsi_thresholds)
            .mark_rule(color="#d62728", strokeDash=[4, 4])
            .encode(y="RSI_Level:Q")
        )

    if {"MACD", "MACD_Signal"}.issubset(price_df.columns):
        requires_secondary_axis = True
        layers.append(
            alt.Chart(price_df)
            .mark_line(color="#9467bd")
            .encode(
                x="Date:T",
                y=alt.Y("MACD:Q", axis=alt.Axis(title="MACD", titleColor="#9467bd"), scale=alt.Scale(zero=False)),
                tooltip=["Date:T", alt.Tooltip("MACD:Q", format=".2f")],
            )
        )
        layers.append(
            alt.Chart(price_df)
            .mark_line(color="#8c564b", strokeDash=[4, 2])
            .encode(
                x="Date:T",
                y=alt.Y(
                    "MACD_Signal:Q",
                    axis=alt.Axis(title="Signal", titleColor="#8c564b"),
                    scale=alt.Scale(zero=False),
                ),
                tooltip=["Date:T", alt.Tooltip("MACD_Signal:Q", format=".2f")],
            )
        )
        if "MACD_Hist" in price_df.columns:
            layers.append(
                alt.Chart(price_df)
                .mark_bar(color="#c5b0d5", opacity=0.5)
                .encode(
                    x="Date:T",
                    y=alt.Y("MACD_Hist:Q", axis=alt.Axis(title="Histogram"), scale=alt.Scale(zero=False)),
                    tooltip=["Date:T", alt.Tooltip("MACD_Hist:Q", format=".2f", title="MACD Hist")],
                )
            )

    return layers, requires_secondary_axis


def build_price_chart(
    results: pd.DataFrame, strategy: str, params: StrategyParameters
) -> alt.Chart:
    if results.empty:
        raise ValueError("Results dataframe cannot be empty when building price chart.")

    params = _sanitize_parameters(params)

    if "Adj Close" not in results.columns:
        raise ValueError("Results missing 'Adj Close' column required for charting.")

    safe_results = results.copy()
    safe_results.index = pd.to_datetime(safe_results.index)

    drop_columns: List[str] = []
    for column_name in list(safe_results.columns):
        column = safe_results[column_name]
        try:
            coerced = _force_series(column, safe_results.index).replace([np.inf, -np.inf], np.nan)
        except Exception:
            if column_name == "Adj Close":
                raise ValueError("Results missing numeric 'Adj Close' data for charting.")
            drop_columns.append(column_name)
            continue
        if coerced.isna().all():
            if column_name == "Adj Close":
                raise ValueError("Results missing numeric 'Adj Close' data for charting.")
            drop_columns.append(column_name)
            continue
        safe_results[column_name] = coerced

    if drop_columns:
        safe_results = safe_results.drop(columns=drop_columns)

    if "Adj Close" not in safe_results.columns:
        raise ValueError("Results missing 'Adj Close' column required for charting.")

    price_df = safe_results.reset_index().rename(columns={"index": "Date"})
    if "Date" not in price_df.columns:
        price_df.rename(columns={price_df.columns[0]: "Date"}, inplace=True)
    price_df["Date"] = pd.to_datetime(price_df["Date"])
    price_df = price_df.replace([np.inf, -np.inf], np.nan)
    price_df = price_df.dropna(subset=["Adj Close"]).copy()
    price_df = price_df.dropna(axis=1, how="all")
    if "Position" in price_df.columns:
        price_df["Position"] = price_df["Position"].fillna(0.0)

    base_price = (
        alt.Chart(price_df)
        .mark_line(color="#1f77b4")
        .encode(
            x="Date:T",
            y=alt.Y("Adj Close:Q", title="Adjusted Close"),
            tooltip=["Date:T", alt.Tooltip("Adj Close:Q", format=".2f")],
        )
    )

    layers: List[alt.Chart] = [base_price]
    indicator_layers, requires_secondary_axis = _add_indicator_layers(price_df, strategy, params)
    layers.extend(indicator_layers)

    position_diff = price_df["Position"].diff().fillna(price_df["Position"])
    buy_points = price_df.loc[position_diff > 0]
    sell_points = price_df.loc[position_diff < 0]

    if not buy_points.empty:
        layers.append(
            alt.Chart(buy_points)
            .mark_point(color="#2ca02c", size=80, shape="triangle-up")
            .encode(
                x="Date:T",
                y="Adj Close:Q",
                tooltip=[
                    "Date:T",
                    alt.Tooltip("Adj Close:Q", format=".2f"),
                    alt.Tooltip("Position:Q", title="Position"),
                ],
            )
        )

    if not sell_points.empty:
        layers.append(
            alt.Chart(sell_points)
            .mark_point(color="#d62728", size=80, shape="triangle-down")
            .encode(
                x="Date:T",
                y="Adj Close:Q",
                tooltip=[
                    "Date:T",
                    alt.Tooltip("Adj Close:Q", format=".2f"),
                    alt.Tooltip("Position:Q", title="Position"),
                ],
            )
        )

    chart = alt.layer(*layers)
    if requires_secondary_axis:
        chart = chart.resolve_scale(y="independent")

    return chart.interactive()


def build_equity_chart(
    equity_curve: pd.Series,
    buy_hold_curve: pd.Series,
    strategy: str,
    benchmark_curve: Optional[pd.Series] = None,
    benchmark_label: str = "Benchmark",
) -> alt.Chart:
    if equity_curve.empty or buy_hold_curve.empty:
        raise ValueError("Equity curves must not be empty when building equity chart.")

    primary_index = equity_curve.index
    curve_dict = {
        strategy: _force_series(equity_curve, primary_index),
        "Buy & Hold": _force_series(buy_hold_curve, primary_index),
    }
    if benchmark_curve is not None and not benchmark_curve.empty:
        curve_dict[benchmark_label] = _force_series(benchmark_curve, primary_index)

    equity_df = pd.concat(curve_dict, axis=1)
    equity_df = equity_df.replace([np.inf, -np.inf], np.nan)
    equity_df = equity_df.dropna(axis=1, how="all")
    equity_df = equity_df.dropna(how="all")
    equity_df.index = pd.to_datetime(equity_df.index)
    equity_df = equity_df.reset_index().rename(columns={"index": "Date"})
    equity_long = equity_df.melt("Date", var_name="Series", value_name="Equity")
    equity_long = equity_long.dropna(subset=["Equity"])

    return (
        alt.Chart(equity_long)
        .mark_line()
        .encode(
            x="Date:T",
            y=alt.Y("Equity:Q", title="Equity"),
            color=alt.Color("Series:N", legend=alt.Legend(title="Strategy")),
            tooltip=["Date:T", "Series:N", alt.Tooltip("Equity:Q", format=".2f")],
        )
        .interactive()
    )


def _create_download_package(results: pd.DataFrame, metrics: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        results_csv = results.reset_index().to_csv(index=False)
        metrics_csv = metrics.to_csv(index=False)
        zip_file.writestr("results.csv", results_csv)
        zip_file.writestr("metrics.csv", metrics_csv)
    buffer.seek(0)
    return buffer.getvalue()


def _prepare_benchmark_curve(
    ticker: str,
    start_date: date,
    end_date: date,
    align_index: Iterable[pd.Timestamp],
) -> Tuple[Optional[pd.Series], Optional[str]]:
    if not ticker:
        return None, None

    try:
        benchmark_data = download_data(ticker, start_date, end_date)
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)

    benchmark_returns = _force_series(
        _clean_numeric_series(benchmark_data["Adj Close"].pct_change(), fill_value=0.0),
        benchmark_data.index,
    )
    benchmark_curve = _force_series(_compute_equity_curve(benchmark_returns), benchmark_returns.index)
    benchmark_curve = benchmark_curve.reindex(pd.Index(align_index)).ffill().bfill()
    benchmark_curve = _force_series(benchmark_curve, benchmark_curve.index)
    if benchmark_curve.isna().all():
        return None, "Benchmark data contained only missing values."

    return benchmark_curve, None


def _render_app() -> None:
    st.title("ASX Stock Strategy Backtester")
    st.caption("Historical performance does not guarantee future results. Use for educational purposes only.")

    with st.sidebar:
        st.header("Backtest Settings")
        selected_common = st.selectbox("Common Tickers", options=COMMON_TICKERS, index=0)
        custom_ticker = st.text_input("Custom Ticker (optional)", value="")

        benchmark_candidates = ["None"] + [ticker for ticker in DEFAULT_BENCHMARKS + COMMON_TICKERS if ticker != selected_common]
        benchmark_choice = st.selectbox("Benchmark Ticker", options=benchmark_candidates, index=0)
        benchmark_custom = st.text_input("Custom Benchmark (optional)", value="")

        default_start = max(date(2000, 1, 1), date.today() - timedelta(days=365 * 5))
        start_date = st.date_input("Start Date", value=default_start)
        end_date = st.date_input("End Date", value=date.today())
        strategy_choice = st.selectbox("Strategy", STRATEGY_OPTIONS)

        with st.expander("Strategy Parameters", expanded=True):
            short_window_input = int(
                st.number_input("Short Moving Average Window", min_value=1, max_value=365, value=13, step=1)
            )
            long_window_input = int(
                st.number_input(
                    "Long Moving Average Window",
                    min_value=short_window_input + 1,
                    max_value=600,
                    value=max(55, short_window_input + 1),
                    step=1,
                )
            )
            rsi_lower_input = float(
                st.number_input("RSI Oversold Threshold", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
            )
            rsi_upper_input = float(
                st.number_input(
                    "RSI Overbought Threshold",
                    min_value=min(100.0, rsi_lower_input + 1.0),
                    max_value=100.0,
                    value=min(100.0, max(70.0, rsi_lower_input + 1.0)),
                    step=1.0,
                )
            )
            bollinger_window_input = int(
                st.number_input("Bollinger Window", min_value=1, max_value=365, value=20, step=1)
            )
            bollinger_width_input = float(
                st.number_input("Bollinger Band Width", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
            )
            momentum_window_input = int(
                st.number_input("Momentum Window", min_value=1, max_value=252, value=12, step=1)
            )
            mean_reversion_window_input = int(
                st.number_input("Mean Reversion Window", min_value=1, max_value=252, value=20, step=1)
            )
            breakout_lookback_input = int(
                st.number_input("Breakout Lookback", min_value=1, max_value=252, value=20, step=1)
            )

        run_backtest = st.button("Run Backtest", type="primary")

    ticker = get_selected_ticker(selected_common, custom_ticker)
    benchmark_choice = benchmark_choice.strip().upper()
    benchmark_custom = benchmark_custom.strip().upper()
    if benchmark_custom:
        benchmark_ticker = benchmark_custom
    elif benchmark_choice and benchmark_choice != "NONE":
        benchmark_ticker = benchmark_choice
    else:
        benchmark_ticker = ""

    params = StrategyParameters(
        short_window=short_window_input,
        long_window=long_window_input,
        rsi_lower=rsi_lower_input,
        rsi_upper=rsi_upper_input,
        bollinger_window=bollinger_window_input,
        bollinger_width=bollinger_width_input,
        momentum_window=momentum_window_input,
        mean_reversion_window=mean_reversion_window_input,
        breakout_lookback=breakout_lookback_input,
    )
    params = _sanitize_parameters(params)

    if run_backtest:
        st.session_state["last_run"] = {
            "ticker": ticker,
            "benchmark": benchmark_ticker,
            "start": start_date,
            "end": end_date,
            "strategy": strategy_choice,
            "params": params,
        }

    last_run = st.session_state.get("last_run")
    if last_run and not run_backtest:
        ticker = last_run["ticker"]
        benchmark_ticker = last_run["benchmark"]
        start_date = last_run["start"]
        end_date = last_run["end"]
        strategy_choice = last_run["strategy"]
        params = last_run["params"]

    if run_backtest or last_run:
        try:
            price_data = download_data(ticker, start_date, end_date)
            results, equity_curve, buy_hold_curve = apply_strategy(price_data, strategy_choice, params)

            if results.empty:
                raise ValueError("No results generated for the selected parameters.")

            benchmark_curve: Optional[pd.Series] = None
            benchmark_error: Optional[str] = None
            if benchmark_ticker and benchmark_ticker != ticker:
                benchmark_curve, benchmark_error = _prepare_benchmark_curve(
                    benchmark_ticker, start_date, end_date, results.index
                )
                if benchmark_error:
                    st.warning(f"Benchmark data unavailable: {benchmark_error}")
                if benchmark_curve is not None:
                    results[f"{benchmark_ticker} Equity"] = benchmark_curve

            metrics_table = compute_metrics(
                equity_curve,
                buy_hold_curve,
                results["Strategy Return"],
                benchmark_curve=benchmark_curve,
            )

            st.success(f"Backtest completed for {ticker} using the {strategy_choice} strategy.")

            st.subheader("Price Chart & Signals")
            st.altair_chart(build_price_chart(results, strategy_choice, params), use_container_width=True)

            st.subheader("Equity Curve Comparison")
            st.altair_chart(
                build_equity_chart(
                    equity_curve,
                    buy_hold_curve,
                    strategy_choice,
                    benchmark_curve=benchmark_curve,
                    benchmark_label=benchmark_ticker or "Benchmark",
                ),
                use_container_width=True,
            )

            metrics_lookup = metrics_table.set_index("Metric")
            strategy_total_raw = metrics_lookup.loc["Strategy Total Return", "RawValue"]
            buy_hold_total_raw = metrics_lookup.loc["Buy & Hold Total Return", "RawValue"]
            total_delta = (
                _format_percentage(strategy_total_raw - buy_hold_total_raw)
                if np.isfinite(strategy_total_raw) and np.isfinite(buy_hold_total_raw)
                else None
            )

            st.subheader("Key Performance Metrics")
            summary_cols = st.columns(3)
            summary_cols[0].metric(
                "Total Return",
                metrics_lookup.loc["Strategy Total Return", "Display"],
                delta=total_delta,
            )
            summary_cols[1].metric("CAGR", metrics_lookup.loc["CAGR", "Display"])
            summary_cols[2].metric("Sharpe Ratio", metrics_lookup.loc["Sharpe Ratio", "Display"])

            st.dataframe(metrics_lookup[["Display"]].rename(columns={"Display": "Value"}))

            export_data = _create_download_package(results, metrics_table)
            st.download_button(
                label="Download Results & Metrics (ZIP)",
                data=export_data,
                file_name=f"backtest_{ticker}_{strategy_choice.replace(' ', '_').lower()}.zip",
                mime="application/zip",
            )

            with st.expander("Show Raw Data"):
                st.dataframe(results.round(4))

        except Exception as exc:  # noqa: BLE001
            st.error(f"Backtest failed: {exc}")
    else:
        st.info("Configure your parameters and click 'Run Backtest' to evaluate the strategy.")


# -------------------- Internal Self Tests -------------------- #


def _create_mock_price_data(rows: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=rows, freq="B")
    base_price = np.linspace(100, 120, rows) + rng.normal(0, 1, rows).cumsum() * 0.1
    close = pd.Series(base_price, index=dates)
    data = pd.DataFrame(
        {
            "Open": close.shift(1, fill_value=close.iloc[0]),
            "High": close + rng.uniform(0.5, 1.5, rows),
            "Low": close - rng.uniform(0.5, 1.5, rows),
            "Close": close,
            "Adj Close": close,
            "Volume": rng.integers(1000, 5000, rows),
        },
        index=dates,
    )
    return data


def _validate_strategy_output(results: pd.DataFrame, equity_curve: pd.Series, buy_hold_curve: pd.Series) -> Dict[str, bool]:
    checks: Dict[str, bool] = {}
    required_columns = {
        "Adj Close",
        "Return",
        "Signal",
        "Position",
        "Strategy Return",
        "Equity Curve",
        "Buy & Hold Equity",
    }
    checks["columns_present"] = required_columns.issubset(results.columns)
    checks["adj_close_float"] = "Adj Close" in results.columns and np.issubdtype(results["Adj Close"].dtype, np.floating)
    checks["no_signal_nan"] = not results["Signal"].isna().any()
    checks["signal_bounds"] = bool(((results["Signal"] >= 0) & (results["Signal"] <= 1)).all())
    checks["position_nan"] = not results["Position"].isna().any()
    checks["equity_non_negative"] = bool((equity_curve >= 0).all())
    checks["buy_hold_non_negative"] = bool((buy_hold_curve >= 0).all())
    checks["series_are_1d"] = all(getattr(results[column], "ndim", 1) == 1 for column in required_columns)
    checks["curves_are_1d"] = getattr(equity_curve, "ndim", 1) == 1 and getattr(buy_hold_curve, "ndim", 1) == 1
    checks["params_attached"] = isinstance(results.attrs.get("strategy_parameters"), StrategyParameters)
    return checks


def _run_self_tests() -> None:
    import unittest
    from unittest import mock

    class BacktesterTests(unittest.TestCase):
        def setUp(self) -> None:
            self.mock_data = _create_mock_price_data()
            self.custom_params = StrategyParameters(
                short_window=5,
                long_window=21,
                rsi_lower=25.0,
                rsi_upper=75.0,
                bollinger_window=15,
                bollinger_width=2.5,
                momentum_window=10,
                mean_reversion_window=15,
                breakout_lookback=15,
            )

        def test_download_data_success(self) -> None:
            with mock.patch("yfinance.download", return_value=self.mock_data.copy()):
                data = download_data("BHP.AX", date(2020, 1, 1), date(2020, 12, 31))
                self.assertFalse(data.empty)
                for col in ["Adj Close", "Close", "High", "Low"]:
                    self.assertIn(col, data.columns)

        def test_download_data_invalid_dates(self) -> None:
            with self.assertRaises(ValueError):
                download_data("BHP.AX", date(2020, 1, 1), date(2020, 1, 1))

        def test_download_data_empty_response(self) -> None:
            with mock.patch("yfinance.download", return_value=pd.DataFrame()):
                with self.assertRaises(ValueError):
                    download_data("BHP.AX", date(2020, 1, 1), date(2020, 2, 1))

        def test_download_data_handles_multiindex_columns(self) -> None:
            multiindex_data = self.mock_data.copy()
            multiindex_data.columns = pd.MultiIndex.from_product([multiindex_data.columns, ["BHP.AX"]])
            with mock.patch("yfinance.download", return_value=multiindex_data):
                data = download_data("BHP.AX", date(2020, 1, 1), date(2020, 12, 31))
                self.assertIn("Adj Close", data.columns)
                self.assertTrue(np.issubdtype(data["Adj Close"].dtype, np.floating))

        def test_download_data_creates_adj_close_from_close(self) -> None:
            without_adj = self.mock_data.drop(columns=["Adj Close"]).copy()
            with mock.patch("yfinance.download", return_value=without_adj):
                data = download_data("BHP.AX", date(2020, 1, 1), date(2020, 12, 31))
                self.assertIn("Adj Close", data.columns)
                self.assertTrue(np.allclose(data["Adj Close"], data["Close"]))

        def test_apply_strategy_outputs(self) -> None:
            for strategy in STRATEGY_OPTIONS:
                results, equity_curve, buy_hold_curve = apply_strategy(
                    self.mock_data, strategy, self.custom_params
                )
                checks = _validate_strategy_output(results, equity_curve, buy_hold_curve)
                self.assertTrue(all(checks.values()), msg=f"Strategy {strategy} failed checks: {checks}")

        def test_compute_metrics_valid(self) -> None:
            results, equity_curve, buy_hold_curve = apply_strategy(
                self.mock_data, "Buy and Hold", self.custom_params
            )
            metrics = compute_metrics(
                equity_curve,
                buy_hold_curve,
                results["Strategy Return"],
            )
            self.assertFalse(metrics.empty)
            self.assertIn("Metric", metrics.columns)
            self.assertIn("RawValue", metrics.columns)
            self.assertIn("Display", metrics.columns)
            self.assertTrue(all(isinstance(val, str) for val in metrics["Display"]))

        def test_compute_metrics_with_benchmark(self) -> None:
            results, equity_curve, buy_hold_curve = apply_strategy(
                self.mock_data, "Momentum", self.custom_params
            )
            benchmark_curve = buy_hold_curve * 1.01
            metrics = compute_metrics(
                equity_curve,
                buy_hold_curve,
                results["Strategy Return"],
                benchmark_curve=benchmark_curve,
            )
            self.assertIn("Benchmark Total Return", metrics["Metric"].values)

        def test_compute_metrics_handles_non_positive_cagr(self) -> None:
            dates = pd.date_range("2021-01-01", periods=2, freq="B")
            equity_curve = pd.Series([1.0, 0.0], index=dates)
            buy_hold_curve = pd.Series([1.0, 1.05], index=dates)
            returns = pd.Series([0.0, -1.0], index=dates)
            metrics = compute_metrics(equity_curve, buy_hold_curve, returns)
            cagr_display = metrics.loc[metrics["Metric"] == "CAGR", "Display"].iloc[0]
            self.assertEqual(cagr_display, "N/A")

        def test_build_price_chart(self) -> None:
            results, _, _ = apply_strategy(
                self.mock_data, "Momentum", self.custom_params
            )
            chart = build_price_chart(results, "Momentum", self.custom_params)
            chart_dict = chart.to_dict()
            self.assertIsInstance(chart_dict, dict)
            self.assertIn("layer", chart_dict)

        def test_build_equity_chart(self) -> None:
            results, equity_curve, buy_hold_curve = apply_strategy(
                self.mock_data, "Mean Reversion", self.custom_params
            )
            benchmark_curve = buy_hold_curve * 0.95
            chart = build_equity_chart(
                equity_curve,
                buy_hold_curve,
                "Mean Reversion",
                benchmark_curve=benchmark_curve,
                benchmark_label="Benchmark Test",
            )
            chart_dict = chart.to_dict()
            self.assertIsInstance(chart_dict, dict)
            self.assertIn("data", chart_dict)

        def test_download_package_contains_files(self) -> None:
            results, equity_curve, buy_hold_curve = apply_strategy(
                self.mock_data, "Buy and Hold", self.custom_params
            )
            metrics = compute_metrics(
                equity_curve,
                buy_hold_curve,
                results["Strategy Return"],
            )
            zip_bytes = _create_download_package(results, metrics)
            with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zip_file:
                self.assertIn("results.csv", zip_file.namelist())
                self.assertIn("metrics.csv", zip_file.namelist())

        def test_parameter_sanitization(self) -> None:
            dirty_params = StrategyParameters(
                short_window=-5,
                long_window=3,
                rsi_lower=90,
                rsi_upper=20,
                bollinger_window=0,
                bollinger_width=-1,
                momentum_window=0,
                mean_reversion_window=-10,
                breakout_lookback=0,
            )
            clean_params = _sanitize_parameters(dirty_params)
            self.assertGreater(clean_params.long_window, clean_params.short_window)
            self.assertLess(clean_params.rsi_lower, clean_params.rsi_upper)
            self.assertGreater(clean_params.bollinger_window, 0)
            self.assertGreater(clean_params.bollinger_width, 0)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BacktesterTests)
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise AssertionError("Self-tests failed")


def main() -> None:
    _render_app()


if __name__ == "__main__":
    _run_self_tests()
    print("ALL TESTS PASSED")
    try:
        main()
    except Exception as exc:  # pragma: no cover - allow running without Streamlit runtime
        print(f"Streamlit runtime not available outside `streamlit run`: {exc}")
