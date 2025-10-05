"""ASX Backtester Streamlit application with internal self-tests."""
import math
from datetime import date, timedelta
from typing import Dict, List, Tuple

try:  # pragma: no cover - fallback ensures self-tests run without Altair installed
    import altair as alt
except ModuleNotFoundError:  # pragma: no cover
    from types import SimpleNamespace

    class _FallbackChart:
        def __init__(self, data=None):
            self._spec: Dict[str, object] = {"encoding": {}}
            if data is not None:
                self._spec["data"] = {"values": self._to_records(data)}
            else:
                self._spec["data"] = None

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


def _ensure_price_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Ensure essential OHLC columns exist by backfilling from available data."""
    df = data.copy()
    if "Adj Close" not in df.columns:
        if "Close" in df.columns:
            df["Adj Close"] = df["Close"].copy()
        else:
            raise ValueError("Price data must contain either 'Adj Close' or 'Close'.")
    if "Close" not in df.columns:
        df["Close"] = df["Adj Close"].copy()
    if "High" not in df.columns:
        df["High"] = df[["Adj Close", "Close"]].max(axis=1)
    if "Low" not in df.columns:
        df["Low"] = df[["Adj Close", "Close"]].min(axis=1)
    if "Open" not in df.columns:
        df["Open"] = df["Close"]
    if "Volume" not in df.columns:
        df["Volume"] = 0
    df = df.ffill().bfill()
    return df


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
    )
    if raw.empty:
        raise ValueError("No price data returned. Confirm the ticker and date range.")

    raw = raw.dropna(how="all")
    if raw.empty:
        raise ValueError("Price data only contains missing values after cleaning.")

    raw.sort_index(inplace=True)
    return _ensure_price_columns(raw)


def apply_strategy(data: pd.DataFrame, strategy: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    if data is None or data.empty:
        raise ValueError("Input price data must not be empty.")

    df = _ensure_price_columns(data)
    df = df.sort_index().copy()
    df["Adj Close"] = df["Adj Close"].astype(float)

    df["Return"] = df["Adj Close"].pct_change().fillna(0.0)
    signals = pd.Series(0.0, index=df.index, dtype=float)

    strategy = strategy.strip()

    if strategy == "Buy and Hold":
        signals.loc[:] = 1.0

    elif strategy == "Moving Average Crossover (13/55)":
        df["SMA_13"] = df["Adj Close"].rolling(window=13, min_periods=1).mean()
        df["SMA_55"] = df["Adj Close"].rolling(window=55, min_periods=1).mean()
        signals = pd.Series(np.where(df["SMA_13"] > df["SMA_55"], 1.0, 0.0), index=df.index, dtype=float)

    elif strategy == "RSI Strategy":
        rsi_indicator = ta.momentum.RSIIndicator(close=df["Adj Close"], window=14, fillna=True)
        df["RSI"] = rsi_indicator.rsi().clip(0, 100)
        raw_signal = np.where(df["RSI"] < 30, 1.0, np.where(df["RSI"] > 70, 0.0, np.nan))
        signals = pd.Series(raw_signal, index=df.index, dtype=float).ffill().fillna(0.0)

    elif strategy == "MACD Strategy":
        macd_indicator = ta.trend.MACD(close=df["Adj Close"], window_slow=26, window_fast=12, window_sign=9, fillna=True)
        df["MACD"] = macd_indicator.macd()
        df["MACD_Signal"] = macd_indicator.macd_signal()
        df["MACD_Hist"] = macd_indicator.macd_diff()
        signals = pd.Series(np.where(df["MACD_Hist"] > 0, 1.0, 0.0), index=df.index, dtype=float)

    elif strategy == "Bollinger Bands":
        bb_indicator = ta.volatility.BollingerBands(close=df["Adj Close"], window=20, window_dev=2, fillna=True)
        df["BB_Middle"] = bb_indicator.bollinger_mavg()
        df["BB_Upper"] = bb_indicator.bollinger_hband()
        df["BB_Lower"] = bb_indicator.bollinger_lband()
        long_signal = df["Adj Close"] < df["BB_Lower"]
        flat_signal = df["Adj Close"] > df["BB_Upper"]
        raw_signal = np.where(long_signal, 1.0, np.where(flat_signal, 0.0, np.nan))
        signals = pd.Series(raw_signal, index=df.index, dtype=float).ffill().fillna(0.0)

    elif strategy == "Momentum":
        roc_indicator = ta.momentum.ROCIndicator(close=df["Adj Close"], window=12, fillna=True)
        df["ROC"] = roc_indicator.roc()
        signals = pd.Series(np.where(df["ROC"] > 0, 1.0, 0.0), index=df.index, dtype=float)

    elif strategy == "Mean Reversion":
        df["SMA_20"] = df["Adj Close"].rolling(window=20, min_periods=1).mean()
        signals = pd.Series(np.where(df["Adj Close"] < df["SMA_20"], 1.0, 0.0), index=df.index, dtype=float)

    elif strategy == "Golden/Death Cross":
        df["SMA_50"] = df["Adj Close"].rolling(window=50, min_periods=1).mean()
        df["SMA_200"] = df["Adj Close"].rolling(window=200, min_periods=1).mean()
        signals = pd.Series(np.where(df["SMA_50"] > df["SMA_200"], 1.0, 0.0), index=df.index, dtype=float)

    elif strategy == "Breakout":
        df["High_20"] = df["High"].rolling(window=20, min_periods=1).max()
        df["Low_20"] = df["Low"].rolling(window=20, min_periods=1).min()
        raw_signal = np.where(
            df["Close"] > df["High_20"].shift(1),
            1.0,
            np.where(df["Close"] < df["Low_20"].shift(1), 0.0, np.nan),
        )
        signals = pd.Series(raw_signal, index=df.index, dtype=float).ffill().fillna(0.0)

    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    signals = signals.clip(lower=0.0, upper=1.0)
    positions = signals.shift(1).fillna(0.0)
    strategy_returns = positions * df["Return"].fillna(0.0)

    equity_curve = (1.0 + strategy_returns).replace([np.inf, -np.inf], np.nan)
    equity_curve = equity_curve.fillna(0.0)
    equity_curve = equity_curve.add(1.0).cumprod()
    equity_curve.replace([np.inf, -np.inf], np.nan, inplace=True)
    equity_curve.fillna(method="ffill", inplace=True)
    equity_curve.fillna(1.0, inplace=True)

    buy_hold_curve = (1.0 + df["Return"]).replace([np.inf, -np.inf], np.nan)
    buy_hold_curve = buy_hold_curve.fillna(0.0)
    buy_hold_curve = buy_hold_curve.add(1.0).cumprod()
    buy_hold_curve.replace([np.inf, -np.inf], np.nan, inplace=True)
    buy_hold_curve.fillna(method="ffill", inplace=True)
    buy_hold_curve.fillna(1.0, inplace=True)

    results = df.copy()
    results["Signal"] = signals
    results["Position"] = positions
    results["Strategy Return"] = strategy_returns
    results["Equity Curve"] = equity_curve
    results["Buy & Hold Equity"] = buy_hold_curve

    return results, equity_curve, buy_hold_curve


def compute_metrics(equity_curve: pd.Series, buy_hold_curve: pd.Series, strategy_returns: pd.Series) -> pd.DataFrame:
    if equity_curve.empty or buy_hold_curve.empty:
        raise ValueError("Equity curves are empty; cannot compute performance metrics.")

    total_return = equity_curve.iloc[-1] - 1.0
    buy_hold_return = buy_hold_curve.iloc[-1] - 1.0

    trading_days = len(equity_curve)
    years = trading_days / 252 if trading_days > 0 else np.nan
    cagr = np.nan
    if pd.notna(years) and years > 0 and equity_curve.iloc[-1] > 0:
        cagr = equity_curve.iloc[-1] ** (1 / years) - 1

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    max_drawdown = drawdown.min()

    active_returns = strategy_returns[strategy_returns != 0]
    win_rate = (active_returns > 0).mean() if not active_returns.empty else np.nan

    volatility = np.nan
    if not strategy_returns.empty:
        volatility = strategy_returns.std(ddof=0) * math.sqrt(252)

    sharpe = np.nan
    daily_std = strategy_returns.std(ddof=0)
    if pd.notna(daily_std) and daily_std > 0:
        sharpe = (strategy_returns.mean() * 252) / (daily_std * math.sqrt(252))

    formatted_metrics = [
        ("Strategy Total Return", total_return),
        ("Buy & Hold Total Return", buy_hold_return),
        ("CAGR", cagr),
        ("Max Drawdown", max_drawdown),
        ("Win Rate", win_rate),
        ("Annualized Volatility", volatility),
        ("Sharpe Ratio", sharpe),
    ]

    metric_rows = []
    for label, value in formatted_metrics:
        if pd.notna(value):
            if "Sharpe" in label:
                metric_rows.append({"Metric": label, "Value": f"{value:.2f}"})
            else:
                metric_rows.append({"Metric": label, "Value": f"{value:.2%}"})
        else:
            metric_rows.append({"Metric": label, "Value": "N/A"})

    metrics = pd.DataFrame(metric_rows)
    return metrics


def _add_indicator_layers(price_df: pd.DataFrame, strategy: str) -> Tuple[List[alt.Chart], bool]:
    layers: List[alt.Chart] = []
    requires_secondary_axis = False

    if {"SMA_13", "SMA_55"}.issubset(price_df.columns):
        layers.append(
            alt.Chart(price_df)
            .mark_line(color="#ff7f0e")
            .encode(x="Date:T", y="SMA_13:Q", tooltip=["Date:T", alt.Tooltip("SMA_13:Q", format=".2f", title="SMA 13")])
        )
        layers.append(
            alt.Chart(price_df)
            .mark_line(color="#2ca02c")
            .encode(x="Date:T", y="SMA_55:Q", tooltip=["Date:T", alt.Tooltip("SMA_55:Q", format=".2f", title="SMA 55")])
        )

    if {"SMA_50", "SMA_200"}.issubset(price_df.columns):
        layers.append(
            alt.Chart(price_df)
            .mark_line(color="#9467bd", strokeDash=[5, 2])
            .encode(x="Date:T", y="SMA_50:Q", tooltip=["Date:T", alt.Tooltip("SMA_50:Q", format=".2f", title="SMA 50")])
        )
        layers.append(
            alt.Chart(price_df)
            .mark_line(color="#8c564b", strokeDash=[3, 2])
            .encode(x="Date:T", y="SMA_200:Q", tooltip=["Date:T", alt.Tooltip("SMA_200:Q", format=".2f", title="SMA 200")])
        )

    if {"BB_Middle", "BB_Upper", "BB_Lower"}.issubset(price_df.columns):
        band = (
            alt.Chart(price_df)
            .mark_area(color="#c6dbef", opacity=0.4)
            .encode(x="Date:T", y="BB_Lower:Q", y2="BB_Upper:Q")
        )
        layers.append(band)
        layers.append(
            alt.Chart(price_df)
            .mark_line(color="#1f77b4", strokeDash=[4, 2])
            .encode(x="Date:T", y="BB_Middle:Q", tooltip=["Date:T", alt.Tooltip("BB_Middle:Q", format=".2f", title="BB Middle")])
        )

    if "SMA_20" in price_df.columns and strategy == "Mean Reversion":
        layers.append(
            alt.Chart(price_df)
            .mark_line(color="#bcbd22")
            .encode(x="Date:T", y="SMA_20:Q", tooltip=["Date:T", alt.Tooltip("SMA_20:Q", format=".2f", title="SMA 20")])
        )

    if "ROC" in price_df.columns:
        requires_secondary_axis = True
        layers.append(
            alt.Chart(price_df)
            .mark_line(color="#17becf", strokeDash=[2, 2])
            .encode(
                x="Date:T",
                y=alt.Y("ROC:Q", axis=alt.Axis(title="ROC", titleColor="#17becf"), scale=alt.Scale(zero=False)),
                tooltip=["Date:T", alt.Tooltip("ROC:Q", format=".2f")],
            )
        )

    if "RSI" in price_df.columns:
        requires_secondary_axis = True
        layers.append(
            alt.Chart(price_df)
            .mark_line(color="#d62728")
            .encode(
                x="Date:T",
                y=alt.Y("RSI:Q", axis=alt.Axis(title="RSI", titleColor="#d62728"), scale=alt.Scale(domain=[0, 100])),
                tooltip=["Date:T", alt.Tooltip("RSI:Q", format=".2f")],
            )
        )
        rsi_thresholds = pd.DataFrame({"RSI_Level": [30, 70]})
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
                y=alt.Y("MACD_Signal:Q", axis=alt.Axis(title="MACD Signal", titleColor="#8c564b"), scale=alt.Scale(zero=False)),
                tooltip=["Date:T", alt.Tooltip("MACD_Signal:Q", format=".2f")],
            )
        )

    if {"High_20", "Low_20"}.issubset(price_df.columns):
        layers.append(
            alt.Chart(price_df)
            .mark_line(color="#ff9896", strokeDash=[6, 2])
            .encode(x="Date:T", y="High_20:Q", tooltip=["Date:T", alt.Tooltip("High_20:Q", format=".2f", title="20D High")])
        )
        layers.append(
            alt.Chart(price_df)
            .mark_line(color="#c5b0d5", strokeDash=[6, 2])
            .encode(x="Date:T", y="Low_20:Q", tooltip=["Date:T", alt.Tooltip("Low_20:Q", format=".2f", title="20D Low")])
        )

    return layers, requires_secondary_axis


def build_price_chart(results: pd.DataFrame, strategy: str) -> alt.Chart:
    if results.empty:
        raise ValueError("Results dataframe cannot be empty when building price chart.")

    price_df = results.reset_index().rename(columns={"index": "Date"})
    if "Date" not in price_df.columns:
        price_df.rename(columns={price_df.columns[0]: "Date"}, inplace=True)
    price_df["Date"] = pd.to_datetime(price_df["Date"])

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
    indicator_layers, requires_secondary_axis = _add_indicator_layers(price_df, strategy)
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
                tooltip=["Date:T", alt.Tooltip("Adj Close:Q", format=".2f"), alt.Tooltip("Position:Q", title="Position")],
            )
        )

    if not sell_points.empty:
        layers.append(
            alt.Chart(sell_points)
            .mark_point(color="#d62728", size=80, shape="triangle-down")
            .encode(
                x="Date:T",
                y="Adj Close:Q",
                tooltip=["Date:T", alt.Tooltip("Adj Close:Q", format=".2f"), alt.Tooltip("Position:Q", title="Position")],
            )
        )

    chart = alt.layer(*layers)
    if requires_secondary_axis:
        chart = chart.resolve_scale(y="independent")

    return chart.interactive()


def build_equity_chart(equity_curve: pd.Series, buy_hold_curve: pd.Series, strategy: str) -> alt.Chart:
    if equity_curve.empty or buy_hold_curve.empty:
        raise ValueError("Equity curves must not be empty when building equity chart.")

    equity_df = pd.DataFrame(
        {
            "Date": equity_curve.index,
            strategy: equity_curve.values,
            "Buy & Hold": buy_hold_curve.values,
        }
    )
    equity_long = equity_df.melt("Date", var_name="Series", value_name="Equity")
    equity_long["Date"] = pd.to_datetime(equity_long["Date"])

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


def _render_app() -> None:
    st.title("ASX Stock Strategy Backtester")
    st.caption("Historical performance does not guarantee future results. Use for educational purposes only.")

    with st.sidebar:
        st.header("Backtest Settings")
        selected_common = st.selectbox("Common Tickers", options=COMMON_TICKERS, index=0)
        custom_ticker = st.text_input("Custom Ticker (optional)", value="")
        start_date = st.date_input("Start Date", value=date(2015, 1, 1))
        end_date = st.date_input("End Date", value=date.today())
        strategy_choice = st.selectbox("Strategy", STRATEGY_OPTIONS)
        run_backtest = st.button("Run Backtest", type="primary")

    ticker = get_selected_ticker(selected_common, custom_ticker)

    if run_backtest:
        try:
            price_data = download_data(ticker, start_date, end_date)
            results, equity_curve, buy_hold_curve = apply_strategy(price_data, strategy_choice)

            if results.empty:
                raise ValueError("No results generated for the selected parameters.")

            metrics_table = compute_metrics(equity_curve, buy_hold_curve, results["Strategy Return"])

            st.success(f"Backtest completed for {ticker} using the {strategy_choice} strategy.")

            st.subheader("Price Chart & Signals")
            st.altair_chart(build_price_chart(results, strategy_choice), use_container_width=True)

            st.subheader("Equity Curve Comparison")
            st.altair_chart(build_equity_chart(equity_curve, buy_hold_curve, strategy_choice), use_container_width=True)

            st.subheader("Performance Metrics")
            st.dataframe(metrics_table.set_index("Metric"))

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
    checks["no_signal_nan"] = not results["Signal"].isna().any()
    checks["signal_bounds"] = bool(((results["Signal"] >= 0) & (results["Signal"] <= 1)).all())
    checks["position_nan"] = not results["Position"].isna().any()
    checks["equity_positive"] = bool((equity_curve > 0).all())
    checks["buy_hold_positive"] = bool((buy_hold_curve > 0).all())
    return checks


def _run_self_tests() -> None:
    import unittest
    from unittest import mock

    class BacktesterTests(unittest.TestCase):
        def setUp(self) -> None:
            self.mock_data = _create_mock_price_data()

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

        def test_apply_strategy_outputs(self) -> None:
            for strategy in STRATEGY_OPTIONS:
                results, equity_curve, buy_hold_curve = apply_strategy(self.mock_data, strategy)
                checks = _validate_strategy_output(results, equity_curve, buy_hold_curve)
                self.assertTrue(all(checks.values()), msg=f"Strategy {strategy} failed checks: {checks}")

        def test_compute_metrics_valid(self) -> None:
            results, equity_curve, buy_hold_curve = apply_strategy(self.mock_data, "Buy and Hold")
            metrics = compute_metrics(equity_curve, buy_hold_curve, results["Strategy Return"])
            self.assertFalse(metrics.empty)
            self.assertIn("Metric", metrics.columns)
            self.assertIn("Value", metrics.columns)
            self.assertTrue(all(isinstance(val, str) for val in metrics["Value"]))

        def test_build_price_chart(self) -> None:
            results, equity_curve, buy_hold_curve = apply_strategy(self.mock_data, "Momentum")
            chart = build_price_chart(results, "Momentum")
            chart_dict = chart.to_dict()
            self.assertIsInstance(chart_dict, dict)
            self.assertIn("layer", chart_dict)

        def test_build_equity_chart(self) -> None:
            results, equity_curve, buy_hold_curve = apply_strategy(self.mock_data, "Mean Reversion")
            chart = build_equity_chart(equity_curve, buy_hold_curve, "Mean Reversion")
            chart_dict = chart.to_dict()
            self.assertIsInstance(chart_dict, dict)
            self.assertIn("data", chart_dict)

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
    main()
