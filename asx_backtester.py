import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
import ta
from datetime import date
from typing import Dict, List

st.set_page_config(page_title="ASX Backtester", layout="wide")

CURATED_TICKERS: List[str] = [
    "BHP.AX",
    "CBA.AX",
    "NAB.AX",
    "WBC.AX",
    "ANZ.AX",
    "WES.AX",
    "WOW.AX",
    "TLS.AX",
    "FMG.AX",
    "CSL.AX",
    "MQG.AX",
    "WPL.AX",
    "COL.AX",
    "GMG.AX",
    "TCL.AX",
    "SUN.AX",
    "QBE.AX",
    "BXB.AX",
    "ORG.AX",
    "ALL.AX",
    "SCG.AX",
    "AZJ.AX",
    "RIO.AX",
    "AMP.AX",
    "ORI.AX",
    "QAN.AX",
    "SEK.AX",
    "S32.AX",
    "STO.AX",
    "TWE.AX",
]

STRATEGIES: List[str] = [
    "Buy and Hold",
    "Moving Average Crossover (13/55)",
    "RSI Strategy (14, 30/70)",
    "MACD Strategy",
    "Bollinger Bands (20, 2σ)",
    "Momentum (ROC > 0)",
    "Mean Reversion (20-day SMA)",
    "Dual Moving Average (50/200)",
    "Breakout Strategy (20-day High/Low)",
]


def download_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    if not ticker:
        raise ValueError("A ticker symbol is required.")

    if start >= end:
        raise ValueError("Start date must be before end date.")

    data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError("No data found for the specified ticker and date range.")

    data = data.dropna()
    if data.empty:
        raise ValueError("Data contains only missing values after cleaning.")

    return data


def initialize_results(data: pd.DataFrame) -> pd.DataFrame:
    results = data.copy()
    results["Adj Close"] = results["Adj Close"].fillna(results["Close"])
    results["Return"] = results["Adj Close"].pct_change().fillna(0.0)
    return results


def apply_strategy(data: pd.DataFrame, strategy: str) -> pd.DataFrame:
    results = initialize_results(data)

    signals = pd.Series(index=results.index, dtype=float)
    overlays: Dict[str, pd.Series] = {}

    if strategy == "Buy and Hold":
        signals[:] = 1.0

    elif strategy == "Moving Average Crossover (13/55)":
        results["SMA_13"] = results["Adj Close"].rolling(window=13, min_periods=1).mean()
        results["SMA_55"] = results["Adj Close"].rolling(window=55, min_periods=1).mean()
        signals = (results["SMA_13"] > results["SMA_55"]).astype(float)
        overlays.update({"SMA_13": results["SMA_13"], "SMA_55": results["SMA_55"]})

    elif strategy == "RSI Strategy (14, 30/70)":
        rsi_indicator = ta.momentum.RSIIndicator(close=results["Adj Close"], window=14)
        results["RSI_14"] = rsi_indicator.rsi()
        rsi_signals = pd.Series(np.nan, index=results.index, dtype=float)
        rsi_signals[results["RSI_14"] < 30] = 1.0
        rsi_signals[results["RSI_14"] > 70] = 0.0
        signals = rsi_signals.ffill().fillna(0.0)
        overlays["RSI_14"] = results["RSI_14"]

    elif strategy == "MACD Strategy":
        macd_indicator = ta.trend.MACD(close=results["Adj Close"], window_slow=26, window_fast=12, window_sign=9)
        results["MACD"] = macd_indicator.macd()
        results["MACD_Signal"] = macd_indicator.macd_signal()
        results["MACD_Diff"] = macd_indicator.macd_diff()
        signals = (results["MACD_Diff"] > 0).astype(float)
        overlays.update({"MACD": results["MACD"], "MACD_Signal": results["MACD_Signal"]})

    elif strategy == "Bollinger Bands (20, 2σ)":
        sma = results["Adj Close"].rolling(window=20, min_periods=1).mean()
        std = results["Adj Close"].rolling(window=20, min_periods=1).std().fillna(0.0)
        results["BB_Middle"] = sma
        results["BB_Upper"] = sma + 2 * std
        results["BB_Lower"] = sma - 2 * std
        signals_series = pd.Series(np.nan, index=results.index, dtype=float)
        signals_series[results["Adj Close"] < results["BB_Lower"]] = 1.0
        signals_series[results["Adj Close"] > results["BB_Upper"]] = 0.0
        signals = signals_series.ffill().fillna(0.0)
        overlays.update({"BB_Upper": results["BB_Upper"], "BB_Middle": results["BB_Middle"], "BB_Lower": results["BB_Lower"]})

    elif strategy == "Momentum (ROC > 0)":
        roc_indicator = ta.momentum.ROCIndicator(close=results["Adj Close"], window=12)
        results["ROC_12"] = roc_indicator.roc()
        signals = (results["ROC_12"] > 0).astype(float)
        overlays["ROC_12"] = results["ROC_12"]

    elif strategy == "Mean Reversion (20-day SMA)":
        results["SMA_20"] = results["Adj Close"].rolling(window=20, min_periods=1).mean()
        signals = (results["Adj Close"] < results["SMA_20"]).astype(float)
        overlays["SMA_20"] = results["SMA_20"]

    elif strategy == "Dual Moving Average (50/200)":
        results["SMA_50"] = results["Adj Close"].rolling(window=50, min_periods=1).mean()
        results["SMA_200"] = results["Adj Close"].rolling(window=200, min_periods=1).mean()
        signals = (results["SMA_50"] > results["SMA_200"]).astype(float)
        overlays.update({"SMA_50": results["SMA_50"], "SMA_200": results["SMA_200"]})

    elif strategy == "Breakout Strategy (20-day High/Low)":
        rolling_high = results["Adj Close"].rolling(window=20, min_periods=1).max()
        rolling_low = results["Adj Close"].rolling(window=20, min_periods=1).min()
        prior_high = rolling_high.shift(1, fill_value=rolling_high.iloc[0])
        prior_low = rolling_low.shift(1, fill_value=rolling_low.iloc[0])
        breakout_signals = pd.Series(np.nan, index=results.index, dtype=float)
        breakout_signals[results["Adj Close"] > prior_high] = 1.0
        breakout_signals[results["Adj Close"] < prior_low] = 0.0
        signals = breakout_signals.ffill().fillna(0.0)
        overlays.update({"High_20": prior_high, "Low_20": prior_low})

    else:
        raise ValueError("Unsupported strategy selected.")

    positions = pd.Series(signals, index=results.index, dtype=float).ffill().fillna(0.0)
    traded_positions = positions.shift(1)
    if not traded_positions.empty:
        traded_positions.iloc[0] = positions.iloc[0]
    traded_positions = traded_positions.fillna(0.0)
    strategy_returns = traded_positions * results["Return"]

    results["Signal"] = signals
    results["Position"] = positions
    results["Strategy Position"] = traded_positions
    results["Strategy Return"] = strategy_returns
    results["Equity Curve"] = (1 + strategy_returns).cumprod()
    results["Buy & Hold Equity"] = (1 + results["Return"]).cumprod()

    for name, series in overlays.items():
        results[name] = series

    return results


def compute_metrics(results: pd.DataFrame) -> pd.DataFrame:
    equity_curve = results["Equity Curve"]
    buy_hold_curve = results["Buy & Hold Equity"]
    strategy_returns = results["Strategy Return"]

    if equity_curve.empty:
        raise ValueError("Equity curve is empty; cannot compute metrics.")

    total_return = equity_curve.iloc[-1] - 1
    buy_hold_return = buy_hold_curve.iloc[-1] - 1

    trading_days = len(equity_curve)
    if trading_days <= 1 or equity_curve.iloc[0] <= 0:
        cagr = np.nan
    else:
        years = trading_days / 252
        cagr = equity_curve.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_drawdown = drawdown.min()

    active_days = results["Strategy Position"] != 0
    if active_days.any():
        win_rate = (strategy_returns[active_days] > 0).mean()
    else:
        win_rate = np.nan

    volatility = strategy_returns.std(ddof=1) * np.sqrt(252)
    sharpe_ratio = np.nan
    if pd.notna(volatility) and volatility > 0:
        sharpe_ratio = (strategy_returns.mean() * 252) / volatility

    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Strategy Total Return",
                "Buy & Hold Total Return",
                "CAGR",
                "Max Drawdown",
                "Win Rate",
                "Volatility",
                "Sharpe Ratio",
            ],
            "Value": [
                f"{total_return:.2%}" if pd.notna(total_return) else "N/A",
                f"{buy_hold_return:.2%}" if pd.notna(buy_hold_return) else "N/A",
                f"{cagr:.2%}" if pd.notna(cagr) else "N/A",
                f"{max_drawdown:.2%}" if pd.notna(max_drawdown) else "N/A",
                f"{win_rate:.2%}" if pd.notna(win_rate) else "N/A",
                f"{volatility:.2%}" if pd.notna(volatility) else "N/A",
                f"{sharpe_ratio:.2f}" if pd.notna(sharpe_ratio) else "N/A",
            ],
        }
    )

    return metrics_df


def build_price_chart(results: pd.DataFrame, strategy: str) -> alt.Chart:
    price_df = results.reset_index().rename(columns={"index": "Date"})
    price_df["Date"] = pd.to_datetime(price_df["Date"])

    price_line = (
        alt.Chart(price_df)
        .mark_line(color="#1f77b4")
        .encode(
            x="Date:T",
            y=alt.Y("Adj Close:Q", title="Price"),
            tooltip=["Date:T", alt.Tooltip("Adj Close:Q", format=".2f")],
        )
    )

    layers = [price_line]

    overlay_columns = {
        "SMA_13": "#ff7f0e",
        "SMA_55": "#2ca02c",
        "SMA_20": "#9467bd",
        "SMA_50": "#ff9896",
        "SMA_200": "#17becf",
        "BB_Upper": "#bcbd22",
        "BB_Middle": "#7f7f7f",
        "BB_Lower": "#bcbd22",
        "High_20": "#8c564b",
        "Low_20": "#8c564b",
    }

    for column, color in overlay_columns.items():
        if column in price_df.columns:
            layers.append(
                alt.Chart(price_df)
                .mark_line(color=color, strokeDash=[4, 2] if "BB" in column or column in {"High_20", "Low_20"} else None)
                .encode(x="Date:T", y=f"{column}:Q", tooltip=["Date:T", alt.Tooltip(f"{column}:Q", format=".2f")])
            )

    position_changes = price_df["Position"].diff().fillna(price_df["Position"])
    buy_points = price_df[position_changes > 0]
    sell_points = price_df[position_changes < 0]

    if not buy_points.empty:
        layers.append(
            alt.Chart(buy_points)
            .mark_point(color="#2ca02c", shape="triangle-up", size=90)
            .encode(
                x="Date:T",
                y="Adj Close:Q",
                tooltip=["Date:T", alt.Tooltip("Adj Close:Q", format=".2f"), alt.Tooltip("Position:Q", title="Position")],
            )
        )

    if not sell_points.empty:
        layers.append(
            alt.Chart(sell_points)
            .mark_point(color="#d62728", shape="triangle-down", size=90)
            .encode(
                x="Date:T",
                y="Adj Close:Q",
                tooltip=["Date:T", alt.Tooltip("Adj Close:Q", format=".2f"), alt.Tooltip("Position:Q", title="Position")],
            )
        )

    chart = alt.layer(*layers).resolve_scale(y="shared").interactive()
    return chart


def build_equity_chart(results: pd.DataFrame, strategy: str) -> alt.Chart:
    equity_df = pd.DataFrame(
        {
            "Date": results.index,
            strategy: results["Equity Curve"].values,
            "Buy & Hold": results["Buy & Hold Equity"].values,
        }
    )
    equity_long = equity_df.melt("Date", var_name="Series", value_name="Equity")
    equity_long["Date"] = pd.to_datetime(equity_long["Date"])

    chart = (
        alt.Chart(equity_long)
        .mark_line()
        .encode(
            x="Date:T",
            y=alt.Y("Equity:Q", title="Equity"),
            color="Series:N",
            tooltip=["Date:T", "Series:N", alt.Tooltip("Equity:Q", format=".3f")],
        )
        .interactive()
    )
    return chart


st.title("ASX Stock Backtester")
st.caption("Select a strategy to evaluate ASX equities using historical OHLCV data from Yahoo Finance.")

with st.sidebar:
    st.header("Backtest Settings")
    default_index = CURATED_TICKERS.index("BHP.AX") if "BHP.AX" in CURATED_TICKERS else 0
    selected_ticker = st.selectbox("Select ASX Ticker", options=CURATED_TICKERS, index=default_index)
    custom_ticker = st.text_input("Or enter custom ticker", value="").strip().upper()
    ticker = custom_ticker if custom_ticker else selected_ticker

    start_date = st.date_input("Start Date", value=date(2020, 1, 1))
    end_date = st.date_input("End Date", value=date.today())
    strategy = st.selectbox("Strategy", STRATEGIES)
    run_backtest = st.button("Run Backtest")

if run_backtest:
    try:
        historical_data = download_data(ticker, start_date, end_date)
        strategy_results = apply_strategy(historical_data, strategy)
        metrics_table = compute_metrics(strategy_results)

        st.success(f"Backtest completed for {ticker} using {strategy}.")

        st.subheader("Price & Trade Signals")
        st.altair_chart(build_price_chart(strategy_results, strategy), use_container_width=True)

        st.subheader("Equity Curve Comparison")
        st.altair_chart(build_equity_chart(strategy_results, strategy), use_container_width=True)

        st.subheader("Performance Metrics")
        st.dataframe(metrics_table.set_index("Metric"))

    except Exception as error:
        st.error(f"Backtest failed: {error}")
else:
    st.info("Configure the settings in the sidebar and click 'Run Backtest' to execute a backtest.")
