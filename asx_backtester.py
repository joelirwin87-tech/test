import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
import ta
from datetime import date
from typing import Tuple

st.set_page_config(page_title="ASX Backtester", layout="wide")


def download_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    if start >= end:
        raise ValueError("Start date must be before end date.")

    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if data.empty:
        raise ValueError("No data found for the specified ticker and date range.")

    data = data.dropna()
    if data.empty:
        raise ValueError("Data contains only missing values after cleaning.")

    return data


def apply_strategy(data: pd.DataFrame, strategy: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = data.copy()
    df["Adj Close"] = df["Adj Close"].fillna(df["Close"])
    df["Return"] = df["Adj Close"].pct_change().fillna(0.0)

    signals = pd.Series(0, index=df.index, dtype=float)

    if strategy == "Buy and Hold":
        signals[:] = 1

    elif strategy == "Moving Average Crossover":
        short_window = 13
        long_window = 55
        df["SMA_Short"] = df["Adj Close"].rolling(window=short_window, min_periods=1).mean()
        df["SMA_Long"] = df["Adj Close"].rolling(window=long_window, min_periods=1).mean()
        signals = np.where(df["SMA_Short"] > df["SMA_Long"], 1.0, 0.0)
        signals = pd.Series(signals, index=df.index)

    elif strategy == "RSI Strategy":
        rsi_indicator = ta.momentum.RSIIndicator(close=df["Adj Close"], window=14)
        df["RSI"] = rsi_indicator.rsi()
        signals = np.where(df["RSI"] < 30, 1.0, np.where(df["RSI"] > 70, 0.0, np.nan))
        signals = pd.Series(signals, index=df.index).ffill().fillna(0.0)

    elif strategy == "MACD Strategy":
        macd_indicator = ta.trend.MACD(close=df["Adj Close"], window_slow=26, window_fast=12, window_sign=9)
        df["MACD"] = macd_indicator.macd()
        df["MACD_Signal"] = macd_indicator.macd_signal()
        df["MACD_Diff"] = macd_indicator.macd_diff()
        signals = np.where(df["MACD_Diff"] > 0, 1.0, 0.0)
        signals = pd.Series(signals, index=df.index)

    else:
        raise ValueError("Unsupported strategy selected.")

    signals = signals.astype(float)
    positions = signals.shift(1).fillna(0.0)
    strategy_returns = positions * df["Return"]
    equity_curve = (1 + strategy_returns).cumprod()
    buy_hold_curve = (1 + df["Return"]).cumprod()

    results = df.copy()
    results["Signal"] = signals
    results["Position"] = positions
    results["Strategy Return"] = strategy_returns
    results["Equity Curve"] = equity_curve
    results["Buy & Hold Equity"] = buy_hold_curve

    return results, equity_curve, buy_hold_curve


def compute_metrics(equity_curve: pd.Series, buy_hold_curve: pd.Series, strategy_returns: pd.Series) -> pd.DataFrame:
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

    active_returns = strategy_returns[strategy_returns != 0]
    if active_returns.empty:
        win_rate = np.nan
    else:
        win_rate = (active_returns > 0).mean()

    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Strategy Total Return",
                "Buy & Hold Total Return",
                "CAGR",
                "Max Drawdown",
                "Win Rate",
            ],
            "Value": [
                f"{total_return:.2%}" if pd.notna(total_return) else "N/A",
                f"{buy_hold_return:.2%}" if pd.notna(buy_hold_return) else "N/A",
                f"{cagr:.2%}" if pd.notna(cagr) else "N/A",
                f"{max_drawdown:.2%}" if pd.notna(max_drawdown) else "N/A",
                f"{win_rate:.2%}" if pd.notna(win_rate) else "N/A",
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
        .encode(x="Date:T", y=alt.Y("Adj Close:Q", title="Price"), tooltip=["Date:T", "Adj Close:Q"])
    )

    layers = [price_line]

    if strategy == "Moving Average Crossover":
        layers.append(
            alt.Chart(price_df)
            .mark_line(color="#ff7f0e")
            .encode(x="Date:T", y="SMA_Short:Q", tooltip=["Date:T", "SMA_Short:Q"])
        )
        layers.append(
            alt.Chart(price_df)
            .mark_line(color="#2ca02c")
            .encode(x="Date:T", y="SMA_Long:Q", tooltip=["Date:T", "SMA_Long:Q"])
        )

    if strategy == "RSI Strategy":
        layers.append(
            alt.Chart(price_df)
            .mark_line(color="#d62728", strokeDash=[5, 3])
            .encode(x="Date:T", y="RSI:Q", tooltip=["Date:T", "RSI:Q"])
        )

    if strategy == "MACD Strategy":
        layers.append(
            alt.Chart(price_df)
            .mark_line(color="#9467bd")
            .encode(x="Date:T", y="MACD:Q", tooltip=["Date:T", "MACD:Q"])
        )
        layers.append(
            alt.Chart(price_df)
            .mark_line(color="#8c564b", strokeDash=[4, 2])
            .encode(x="Date:T", y="MACD_Signal:Q", tooltip=["Date:T", "MACD_Signal:Q"])
        )

    signals_diff = price_df["Signal"].diff().fillna(price_df["Signal"])
    buy_points = price_df.loc[signals_diff > 0]
    sell_points = price_df.loc[signals_diff < 0]

    if not buy_points.empty:
        layers.append(
            alt.Chart(buy_points)
            .mark_point(color="#2ca02c", size=80, shape="triangle-up")
            .encode(x="Date:T", y="Adj Close:Q", tooltip=["Date:T", "Adj Close:Q", alt.Tooltip("Signal:Q", title="Signal")])
        )

    if not sell_points.empty:
        layers.append(
            alt.Chart(sell_points)
            .mark_point(color="#d62728", size=80, shape="triangle-down")
            .encode(x="Date:T", y="Adj Close:Q", tooltip=["Date:T", "Adj Close:Q", alt.Tooltip("Signal:Q", title="Signal")])
        )

    chart = alt.layer(*layers).resolve_scale(y="independent").interactive()
    return chart


def build_equity_chart(equity_curve: pd.Series, buy_hold_curve: pd.Series, strategy: str) -> alt.Chart:
    equity_df = pd.DataFrame(
        {
            "Date": equity_curve.index,
            strategy: equity_curve.values,
            "Buy & Hold": buy_hold_curve.values,
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
            tooltip=["Date:T", "Series:N", alt.Tooltip("Equity:Q", format=".2f")],
        )
        .interactive()
    )
    return chart


st.title("ASX Stock Backtester")

with st.sidebar:
    st.header("Backtest Settings")
    ticker = st.text_input("ASX Ticker", value="BHP.AX").strip().upper()
    start_date = st.date_input("Start Date", value=date(2020, 1, 1))
    end_date = st.date_input("End Date", value=date.today())
    strategy = st.selectbox(
        "Strategy",
        [
            "Buy and Hold",
            "Moving Average Crossover",
            "RSI Strategy",
            "MACD Strategy",
        ],
    )
    run_backtest = st.button("Run Backtest")

if run_backtest:
    try:
        raw_data = download_data(ticker, start_date, end_date)
        results, equity_curve, buy_hold_curve = apply_strategy(raw_data, strategy)
        metrics_table = compute_metrics(equity_curve, buy_hold_curve, results["Strategy Return"])

        st.success(f"Backtest completed for {ticker} using {strategy} strategy.")

        price_chart = build_price_chart(results, strategy)
        st.subheader("Price & Signals")
        st.altair_chart(price_chart, use_container_width=True)

        equity_chart = build_equity_chart(equity_curve, buy_hold_curve, strategy)
        st.subheader("Equity Curve Comparison")
        st.altair_chart(equity_chart, use_container_width=True)

        st.subheader("Performance Metrics")
        st.dataframe(metrics_table.set_index("Metric"))

    except Exception as err:
        st.error(f"Backtest failed: {err}")
else:
    st.info("Set parameters in the sidebar and click 'Run Backtest' to begin.")
