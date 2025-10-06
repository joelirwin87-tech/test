"""Local CSV-backed price history caching utilities."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _prepare_download_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized DataFrame with a ``Date`` column."""

    if frame.empty:
        return frame

    data = frame.copy()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [str(level[0]) if isinstance(level, tuple) else str(level) for level in data.columns]
    else:
        data.columns = [str(col) for col in data.columns]

    if "Date" not in data.columns:
        index_name = data.index.name or "Date"
        data = data.reset_index().rename(columns={index_name: "Date"})
    else:
        data["Date"] = pd.to_datetime(data["Date"])

    data["Date"] = pd.to_datetime(data["Date"], utc=True).dt.tz_localize(None)
    data = data.sort_values("Date")

    return data


def update_ticker_csv(ticker: str) -> pd.DataFrame:
    """Download and persist historical data for ``ticker`` returning the full dataset."""

    if not ticker:
        raise ValueError("ticker must be provided")

    path = DATA_DIR / f"{ticker}.csv"

    if path.exists():
        df_old = pd.read_csv(path, parse_dates=["Date"])
        df_old["Date"] = pd.to_datetime(df_old["Date"], utc=True).dt.tz_localize(None)
        last = df_old["Date"].max()
        try:
            df_new = yf.download(
                ticker,
                start=last + pd.Timedelta(days=1),
                interval="1d",
                progress=False,
            )
        except Exception as err:  # pragma: no cover - network dependent
            raise ValueError(f"Failed to update data for {ticker}: {err}") from err
        if not df_new.empty:
            df_new = _prepare_download_frame(df_new)
            combined = pd.concat([df_old, df_new], ignore_index=True)
            combined = combined.drop_duplicates(subset=["Date"]).sort_values("Date")
            combined.to_csv(path, index=False)
            return combined.reset_index(drop=True)
        return df_old.sort_values("Date").reset_index(drop=True)

    try:
        df = yf.download(ticker, period="max", interval="1d", progress=False)
    except Exception as err:  # pragma: no cover - network dependent
        raise ValueError(f"Failed to download data for {ticker}: {err}") from err

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    df = _prepare_download_frame(df)
    df.to_csv(path, index=False)
    return df.reset_index(drop=True)


def fetch_price_history(ticker: str, start: date) -> pd.DataFrame:
    """Return cached daily price history for ``ticker`` from ``start`` onward."""

    if start is None:
        raise ValueError("start date must be provided")

    df = update_ticker_csv(ticker)
    if df.empty:
        raise ValueError(f"No cached data available for {ticker}")

    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    filtered = df[df["Date"] >= pd.Timestamp(start)].copy()
    if filtered.empty:
        raise ValueError(f"No data available for {ticker} on or after {start}")

    filtered.rename(columns={"Date": "date"}, inplace=True)
    filtered.set_index("date", inplace=True)
    filtered.sort_index(inplace=True)

    if "Adj Close" in filtered.columns and "AdjClose" not in filtered.columns:
        filtered["AdjClose"] = filtered["Adj Close"]
    elif "AdjClose" in filtered.columns and "Adj Close" not in filtered.columns:
        filtered["Adj Close"] = filtered["AdjClose"]

    return filtered
