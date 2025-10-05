"""Streamlit ASX200 daily signal provider using Golden Cross with profit targets."""
from __future__ import annotations

import argparse
import io
import sys
import time
import unittest
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

try:  # pragma: no cover - optional dependency for charts
    import altair as alt
except Exception:  # pragma: no cover - fallback if Altair unavailable
    alt = None  # type: ignore


ASX200_CSV = """ticker,name,sector,market_cap_billion
A2M.AX,The a2 Milk Company Ltd,Consumer Staples,4.1
ABB.AX,Austal Ltd,Industrials,1.6
ABC.AX,Adbri Ltd,Materials,1.5
ABP.AX,Abacus Group,Real Estate,2.2
AD8.AX,Audinate Group Ltd,Information Technology,1.5
AGL.AX,AGL Energy Ltd,Utilities,6.2
AHY.AX,Asaleo Care Ltd,Consumer Staples,0.8
ALD.AX,Ampol Ltd,Energy,6.9
ALG.AX,Ardent Leisure Group,Consumer Discretionary,0.6
ALQ.AX,ALS Ltd,Industrials,6.0
ALL.AX,Aristocrat Leisure Ltd,Consumer Discretionary,24.0
ALU.AX,Altium Ltd,Information Technology,6.2
ALX.AX,Atlas Arteria,Industrials,6.5
AMA.AX,AMA Group Ltd,Industrials,0.6
AMC.AX,Amcor PLC,Materials,19.5
AMP.AX,AMP Ltd,Financials,3.5
ANN.AX,Ansell Ltd,Health Care,4.3
ANZ.AX,ANZ Banking Group,Financials,74.2
APA.AX,APA Group,Utilities,14.8
APE.AX,Eagers Automotive Ltd,Consumer Discretionary,3.7
APM.AX,APM Human Services,Industrials,1.8
APX.AX,Appen Ltd,Information Technology,0.6
ARB.AX,ARB Corporation Ltd,Consumer Discretionary,2.3
ARF.AX,Arena REIT,Real Estate,1.5
ARG.AX,Argo Investments Ltd,Financials,6.5
ASX.AX,ASX Ltd,Financials,15.1
AUB.AX,AUB Group Ltd,Financials,3.0
AWC.AX,Alumina Ltd,Materials,4.0
AX1.AX,Accent Group Ltd,Consumer Discretionary,1.3
AZJ.AX,Aurizon Holdings Ltd,Industrials,7.0
BAP.AX,Bapcor Ltd,Consumer Discretionary,2.0
BEN.AX,Bendigo and Adelaide Bank Ltd,Financials,5.2
BFG.AX,Bell Financial Group Ltd,Financials,0.5
BHP.AX,BHP Group Ltd,Materials,220.0
BKL.AX,Blackmores Ltd,Health Care,1.2
BLD.AX,Boral Ltd,Materials,5.0
BPT.AX,Beach Energy Ltd,Energy,3.3
BRG.AX,Breville Group Ltd,Consumer Discretionary,4.5
BSL.AX,BlueScope Steel Ltd,Materials,8.2
BVS.AX,Bravura Solutions Ltd,Information Technology,0.6
BWP.AX,BWP Trust,Real Estate,2.6
BXB.AX,Brambles Ltd,Industrials,19.0
CAR.AX,Car Group Ltd,Communication Services,7.2
CBA.AX,Commonwealth Bank of Australia,Financials,170.0
CCP.AX,Credit Corp Group Ltd,Financials,1.6
CCX.AX,City Chic Collective,Consumer Discretionary,0.7
CDP.AX,Carindale Property Trust,Real Estate,1.2
CGF.AX,Challenger Ltd,Financials,6.1
CHC.AX,Charter Hall Group,Real Estate,7.8
CHN.AX,Chalice Mining Ltd,Materials,2.4
CIA.AX,Champion Iron Ltd,Materials,3.5
CIM.AX,CIMIC Group Ltd,Industrials,7.0
CKF.AX,Collins Foods Ltd,Consumer Discretionary,1.2
CLW.AX,Charter Hall Long WALE REIT,Real Estate,2.9
CMW.AX,Centuria Office REIT,Real Estate,1.1
CNI.AX,Centuria Capital Group,Real Estate,2.2
COF.AX,Centuria Office REIT,Real Estate,1.3
COH.AX,Cochlear Ltd,Health Care,15.0
COL.AX,Coles Group Ltd,Consumer Staples,25.6
CPU.AX,Computershare Ltd,Information Technology,15.9
CQR.AX,Charter Hall Retail REIT,Real Estate,2.1
CSL.AX,CSL Ltd,Health Care,140.0
CSR.AX,CSR Ltd,Materials,3.9
CTD.AX,Corporate Travel Management,Consumer Discretionary,2.5
CUV.AX,Clinuvel Pharmaceuticals,Health Care,1.4
CWY.AX,Cleanaway Waste Management Ltd,Industrials,6.0
CXO.AX,Core Lithium Ltd,Materials,2.0
DBI.AX,Dalrymple Bay Infrastructure,Industrials,1.0
DDR.AX,Dicker Data Ltd,Information Technology,2.0
DEG.AX,De Grey Mining Ltd,Materials,2.5
DMP.AX,Domino's Pizza Enterprises Ltd,Consumer Discretionary,5.8
DOW.AX,Downer EDI Ltd,Industrials,3.0
DRR.AX,Deterra Royalties Ltd,Materials,2.5
DXS.AX,Dexus,Real Estate,10.5
EBO.AX,Ebos Group Ltd,Health Care,7.8
EDV.AX,Endeavour Group Ltd,Consumer Staples,12.0
ELD.AX,Elders Ltd,Consumer Staples,2.1
EML.AX,EML Payments Ltd,Information Technology,1.5
EHE.AX,Estia Health Ltd,Health Care,0.9
EVN.AX,Evolution Mining Ltd,Materials,7.5
EVT.AX,Event Hospitality,Consumer Discretionary,2.3
FBU.AX,Fletcher Building Ltd,Industrials,4.3
FLT.AX,Flight Centre Travel Group Ltd,Consumer Discretionary,4.2
FMG.AX,Fortescue Metals Group Ltd,Materials,70.0
FPH.AX,Fisher & Paykel Healthcare,Health Care,14.2
GMG.AX,Goodman Group,Real Estate,46.0
GMA.AX,Genworth Mortgage Insurance,Financials,0.9
GNC.AX,GrainCorp Ltd,Consumer Staples,1.9
GOR.AX,Gold Road Resources Ltd,Materials,1.2
GPT.AX,GPT Group,Real Estate,8.7
GUD.AX,GUD Holdings Ltd,Consumer Discretionary,1.3
GWA.AX,GWA Group Ltd,Industrials,0.9
HLS.AX,Healius Ltd,Health Care,2.8
HMC.AX,HMC Capital Ltd,Financials,1.5
HSN.AX,Hansen Technologies Ltd,Information Technology,1.0
HUB.AX,HUB24 Ltd,Financials,2.0
HVN.AX,Harvey Norman Holdings Ltd,Consumer Discretionary,5.2
IAG.AX,Insurance Australia Group Ltd,Financials,12.5
IEL.AX,IDP Education Ltd,Consumer Discretionary,7.0
IFL.AX,Insignia Financial Ltd,Financials,2.0
IFT.AX,Infratil Ltd,Utilities,5.7
IGO.AX,IGO Ltd,Materials,11.0
ILU.AX,Iluka Resources Ltd,Materials,5.0
IMU.AX,Imugene Ltd,Health Care,1.0
INA.AX,Ingenia Communities Group,Real Estate,1.6
IPL.AX,Incitec Pivot Ltd,Materials,6.8
IPH.AX,IPH Ltd,Industrials,1.9
IRE.AX,Iress Ltd,Information Technology,2.8
IVC.AX,InvoCare Ltd,Consumer Discretionary,1.4
JHX.AX,James Hardie Industries PLC,Materials,24.0
JIN.AX,Jumbo Interactive,Consumer Discretionary,1.1
JBH.AX,JB Hi-Fi Ltd,Consumer Discretionary,5.4
JLG.AX,Johns Lyng Group,Industrials,2.0
KAR.AX,Karoon Energy Ltd,Energy,2.5
KGN.AX,Kogan.com Ltd,Consumer Discretionary,1.0
LLC.AX,Lendlease Group,Real Estate,6.6
LNK.AX,Link Administration Holdings Ltd,Information Technology,2.8
LTR.AX,Liontown Resources Ltd,Materials,3.0
LYC.AX,Lynas Rare Earths Ltd,Materials,8.1
MAQ.AX,Macquarie Telecom Group,Information Technology,1.2
MCY.AX,Mercury NZ Ltd,Utilities,5.0
MEZ.AX,Meridian Energy Ltd,Utilities,7.0
MFG.AX,Magellan Financial Group Ltd,Financials,3.4
MGR.AX,Mirvac Group,Real Estate,10.2
MIN.AX,Mineral Resources Ltd,Materials,13.5
MND.AX,Monadelphous Group,Industrials,1.6
MPL.AX,Medibank Private Ltd,Health Care,8.0
MP1.AX,Megaport Ltd,Information Technology,2.0
MQG.AX,Macquarie Group Ltd,Financials,70.0
MSB.AX,Mesoblast Ltd,Health Care,1.1
MTS.AX,Metcash Ltd,Consumer Staples,4.3
MYR.AX,Myer Holdings Ltd,Consumer Discretionary,0.8
MYX.AX,Mayne Pharma Group Ltd,Health Care,0.6
NAB.AX,National Australia Bank Ltd,Financials,95.0
NAN.AX,Nanosonics Ltd,Health Care,2.0
NCK.AX,Nick Scali Ltd,Consumer Discretionary,1.0
NCM.AX,Newcrest Mining Ltd,Materials,20.0
NHF.AX,NIB Holdings Ltd,Financials,3.2
NIC.AX,Nickel Industries Ltd,Materials,2.4
NSR.AX,National Storage REIT,Real Estate,3.0
NST.AX,Northern Star Resources Ltd,Materials,14.0
NUF.AX,Nufarm Ltd,Materials,2.2
NWH.AX,NRW Holdings Ltd,Industrials,1.2
NWL.AX,Netwealth Group Ltd,Financials,6.1
NXT.AX,Nextdc Ltd,Information Technology,6.5
OBL.AX,Omni Bridgeway,Financials,1.0
OML.AX,Ooh!Media Ltd,Communication Services,1.1
ORA.AX,Orora Ltd,Materials,3.9
ORG.AX,Origin Energy Ltd,Energy,15.0
ORI.AX,Orica Ltd,Materials,6.8
OSH.AX,Oil Search Ltd,Energy,11.0
OZL.AX,Oz Minerals Ltd,Materials,9.0
PDN.AX,Paladin Energy Ltd,Energy,1.4
PGH.AX,Peet Ltd,Real Estate,0.8
PLS.AX,Pilbara Minerals Ltd,Materials,12.5
PME.AX,Pro Medicus Ltd,Health Care,6.0
PNI.AX,Pinnacle Investment Mgmt,Financials,3.5
PNV.AX,Polynovo Ltd,Health Care,1.3
PPM.AX,Pepper Money Ltd,Financials,1.0
PRN.AX,Perenti Global,Industrials,0.9
PRU.AX,Perseus Mining Ltd,Materials,2.5
PTM.AX,Platinum Asset Management Ltd,Financials,1.5
QAN.AX,Qantas Airways Ltd,Industrials,10.0
QBE.AX,QBE Insurance Group Ltd,Financials,17.0
QUB.AX,Qube Holdings Ltd,Industrials,6.0
REA.AX,REA Group Ltd,Communication Services,22.0
RED.AX,Red 5 Ltd,Materials,0.9
REG.AX,Regis Healthcare Ltd,Health Care,0.9
REH.AX,Reece Ltd,Industrials,12.0
RHC.AX,Ramsay Health Care Ltd,Health Care,14.0
RIO.AX,Rio Tinto Ltd,Materials,45.0
RMD.AX,Resmed Inc,Health Care,35.0
RRL.AX,Regis Resources Ltd,Materials,1.5
RSG.AX,Resolute Mining Ltd,Materials,1.0
RWC.AX,Reliance Worldwide Corporation,Industrials,3.7
S32.AX,South32 Ltd,Materials,18.0
SAR.AX,Saracen Mineral Holdings,Materials,1.2
SBM.AX,St Barbara Ltd,Materials,0.9
SCG.AX,Scentre Group,Real Estate,14.5
SCP.AX,Shopping Centres Australasia,Real Estate,3.0
SDF.AX,Steadfast Group,Financials,5.0
SEK.AX,Seek Ltd,Communication Services,10.2
SFR.AX,Sandfire Resources Ltd,Materials,2.8
SGM.AX,Sims Ltd,Materials,3.0
SGP.AX,Stockland,Real Estate,11.0
SGR.AX,The Star Entertainment Group,Consumer Discretionary,2.7
SHL.AX,Sonic Healthcare Ltd,Health Care,16.0
SIQ.AX,Smartgroup Corp Ltd,Industrials,1.1
SIT.AX,SiteMinder Ltd,Information Technology,1.3
SKC.AX,SkyCity Entertainment,Consumer Discretionary,1.6
SKI.AX,Spark Infrastructure,Utilities,8.5
SKT.AX,Sky Network Television,Communication Services,0.7
SLR.AX,Silver Lake Resources,Materials,1.2
SLK.AX,Sealink Travel Group,Industrials,2.1
SM1.AX,Synlait Milk Ltd,Consumer Staples,0.4
SNZ.AX,Spark New Zealand,Communication Services,8.0
SOL.AX,Washington H Soul Pattinson,Financials,9.8
SPK.AX,Spark New Zealand Ltd,Communication Services,8.0
SPL.AX,Starpharma Holdings Ltd,Health Care,0.5
SQ2.AX,Block Inc,Information Technology,40.0
"""

@dataclass(frozen=True)
class Trade:
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float

    @property
    def pct_return(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.exit_price / self.entry_price) - 1.0


def _load_metadata() -> pd.DataFrame:
    """Load ASX200 metadata from the embedded CSV."""

    df = pd.read_csv(io.StringIO(ASX200_CSV))
    df["ticker"] = df["ticker"].str.strip().str.upper()
    df["sector"] = df["sector"].str.strip()
    df["market_cap_billion"] = pd.to_numeric(df["market_cap_billion"], errors="coerce")
    df = df.dropna(subset=["ticker"]).drop_duplicates("ticker").set_index("ticker", drop=False)
    return df


@st.cache_data(show_spinner=False)
def get_metadata() -> pd.DataFrame:
    return _load_metadata()


def _empty_metadata_frame(columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Return an empty metadata frame with the expected columns and index."""

    default_columns: Tuple[str, ...] = (
        "ticker",
        "name",
        "sector",
        "market_cap_billion",
        "exchange",
        "type",
    )
    resolved_columns: Tuple[str, ...] = tuple(columns) if columns is not None else default_columns
    frame = pd.DataFrame(columns=resolved_columns)

    if "ticker" in frame.columns:
        frame = frame.set_index("ticker", drop=False)

    if "market_cap_billion" in frame.columns:
        frame["market_cap_billion"] = frame["market_cap_billion"].astype(float)

    return frame


def _call_with_optional_repair(
    func: Callable[..., pd.DataFrame],
    *,
    args: Optional[Iterable[object]] = None,
    kwargs: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """Invoke a callable that may or may not accept a ``repair`` keyword.

    yfinance introduced a ``repair`` flag to automatically patch missing data. Older
    versions of the library reject the argument, so we optimistically attempt the call
    with ``repair=True`` and gracefully retry without the flag if it is unsupported.
    """

    call_args = tuple(args or ())
    base_kwargs = dict(kwargs or {})

    if "repair" not in base_kwargs:
        repair_kwargs = dict(base_kwargs)
        repair_kwargs["repair"] = True
        try:
            return func(*call_args, **repair_kwargs)
        except TypeError as exc:
            message = str(exc).lower()
            if "repair" not in message:
                raise
        # Fall through to retry without the repair flag.

    return func(*call_args, **base_kwargs)


def _download_with_backoff(ticker: str, start: date, *, attempts: int = 3) -> pd.DataFrame:
    """Download price history with retry and fallback handling."""

    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            data = _call_with_optional_repair(
                yf.download,
                args=(ticker,),
                kwargs={
                    "start": start,
                    "progress": False,
                    "auto_adjust": False,
                    "rounding": True,
                    "threads": False,
                },
            )
        except Exception as err:  # pragma: no cover - network dependent
            last_error = err
            data = pd.DataFrame()

        if not data.empty:
            return data

        time.sleep(min(2.0 * attempt, 6.0))

    ticker_client = yf.Ticker(ticker)
    try:
        data = ticker_client.history(start=start, interval="1d", auto_adjust=False, actions=False)
    except Exception as err:  # pragma: no cover - network dependent
        last_error = err
        data = pd.DataFrame()

    if data.empty:
        try:
            fallback = ticker_client.history(period="max", interval="1d", auto_adjust=False, actions=False)
        except Exception as err:  # pragma: no cover - network dependent
            last_error = err
            fallback = pd.DataFrame()
        if not fallback.empty:
            data = fallback[fallback.index >= pd.Timestamp(start)]

    if data.empty:
        detail = f": {last_error}" if last_error else ""
        raise ValueError(f"No data returned for {ticker}{detail}")

    return data


def _fetch_price_history_uncached(ticker: str, start: date) -> pd.DataFrame:
    """Fetch daily OHLCV data for a ticker using yfinance."""

    data = _download_with_backoff(ticker, start)
    data = data.reset_index().rename(columns={"Date": "date"})
    data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
    data = data.set_index("date").sort_index()
    for column in ["Open", "High", "Low", "Close", "Adj Close"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if "Adj Close" not in data.columns or data["Adj Close"].isna().all():
        data["Adj Close"] = data["Close"]
    data["Adj Close"] = data["Adj Close"].fillna(data["Close"])
    data["Volume"] = pd.to_numeric(data.get("Volume"), errors="coerce").fillna(0)
    data = data[[col for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if col in data.columns]]
    return data


@st.cache_data(show_spinner=False)
def fetch_price_history(ticker: str, start: date) -> pd.DataFrame:
    return _fetch_price_history_uncached(ticker, start)


def _ensure_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        raise KeyError(f"Missing column {column}")
    series = pd.to_numeric(df[column], errors="coerce")
    series = series.astype(float).dropna()
    return series


@dataclass
class StrategyResult:
    ticker: str
    trades: List[Trade]
    equity_curve: pd.Series
    indicators: pd.DataFrame
    stats: Dict[str, float]
    last_signal: Optional[str]
    last_signal_date: Optional[pd.Timestamp]
    entry_price: Optional[float]
    target_price: Optional[float]


def golden_cross_strategy(
    ticker: str,
    price_data: pd.DataFrame,
    profit_target: float,
) -> StrategyResult:
    """Run a Golden Cross strategy with a profit target on the supplied data."""

    if price_data.empty:
        raise ValueError("Price data cannot be empty")
    if profit_target <= 0:
        raise ValueError("Profit target must be positive")

    price = price_data.get("Adj Close")
    if price is None or price.isna().all():
        price = price_data.get("Close")
    if price is None:
        raise ValueError("Price data missing close information")

    price = price.astype(float)
    price = price.replace([np.inf, -np.inf], np.nan).dropna()
    if price.empty:
        raise ValueError("Price series empty after cleaning")

    df = price.to_frame(name="price")
    df["sma50"] = df["price"].rolling(window=50, min_periods=50).mean()
    df["sma200"] = df["price"].rolling(window=200, min_periods=200).mean()
    df = df.dropna(subset=["sma200"]).copy()
    if df.empty:
        raise ValueError("Insufficient history for moving averages")

    prev_sma50 = df["sma50"].shift(1)
    prev_sma200 = df["sma200"].shift(1)
    crosses_up = (df["sma50"] > df["sma200"]) & (prev_sma50 <= prev_sma200)

    trades: List[Trade] = []
    entry_price: Optional[float] = None
    entry_date: Optional[pd.Timestamp] = None
    last_signal: Optional[str] = None
    last_signal_date: Optional[pd.Timestamp] = None

    equity = pd.Series(np.nan, index=df.index, dtype=float)
    current_equity = 1.0
    entry_equity: Optional[float] = None

    for current_date, row in df.iterrows():
        price_value = row["price"]
        if entry_price is not None and entry_equity is not None:
            equity_value = entry_equity * (price_value / entry_price)
        else:
            equity_value = current_equity

        target_price = entry_price * (1 + profit_target) if entry_price is not None else None
        if entry_price is None:
            if crosses_up.loc[current_date]:
                entry_price = price_value
                entry_date = current_date
                entry_equity = current_equity
                last_signal = "Buy"
                last_signal_date = current_date
        else:
            exit_reason = None
            if target_price is not None and price_value >= target_price:
                exit_reason = "target"
            elif row["sma50"] < row["sma200"]:
                exit_reason = "death_cross"
            if exit_reason is not None and entry_date is not None and entry_equity is not None:
                trades.append(Trade(ticker, entry_date, current_date, entry_price, price_value))
                current_equity = entry_equity * (price_value / entry_price)
                equity_value = current_equity
                entry_price = None
                entry_date = None
                entry_equity = None
                last_signal = "Sell"
                last_signal_date = current_date

        equity.loc[current_date] = equity_value

    equity = equity.sort_index().ffill().bfill()
    if equity.empty:
        equity = pd.Series([1.0], index=[df.index[0]])

    if entry_price is not None and entry_date is not None:
        last_signal = "Hold"
        last_signal_date = df.index[-1]
    elif last_signal is None:
        last_signal = "None"
        last_signal_date = df.index[-1]

    stats = compute_statistics(df.index, trades, equity)
    indicators = df
    target_price = entry_price * (1 + profit_target) if entry_price is not None else None

    return StrategyResult(
        ticker=ticker,
        trades=trades,
        equity_curve=equity,
        indicators=indicators,
        stats=stats,
        last_signal=last_signal,
        last_signal_date=last_signal_date,
        entry_price=entry_price,
        target_price=target_price,
    )


def compute_statistics(index: pd.Index, trades: Iterable[Trade], equity_curve: pd.Series) -> Dict[str, float]:
    """Compute win rate, average trade return, CAGR, and max drawdown."""

    trades = list(trades)
    trade_returns = np.array([trade.pct_return for trade in trades], dtype=float)
    wins = (trade_returns > 0).sum()
    total_trades = len(trades)
    win_rate = float(wins / total_trades) if total_trades else 0.0
    avg_return = float(trade_returns.mean()) if total_trades else 0.0

    if len(index) > 1:
        start_date = pd.Timestamp(index[0])
        end_date = pd.Timestamp(index[-1])
        years = max((end_date - start_date).days / 365.25, 0.0001)
    else:
        years = 0.0001

    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)
    cagr = float((1 + total_return) ** (1 / years) - 1) if years > 0 else 0.0

    running_max = equity_curve.cummax()
    drawdowns = (equity_curve / running_max) - 1.0
    max_drawdown = float(drawdowns.min()) if not drawdowns.empty else 0.0

    return {
        "win_rate": win_rate,
        "average_return": avg_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "total_trades": float(total_trades),
        "total_return": total_return,
    }


def scan_tickers(
    tickers: Iterable[str],
    start_date: date | datetime | pd.Timestamp,
    profit_target: float,
    win_rate_threshold: float,
    cagr_threshold: float,
    *,
    price_fetcher: Optional[
        Callable[[str, date | datetime | pd.Timestamp], pd.DataFrame]
    ] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the strategy for each ticker and build summary tables."""

    if profit_target <= 0:
        raise ValueError("profit_target must be positive")

    normalized_start: Optional[date]
    if isinstance(start_date, datetime):
        normalized_start = start_date.date()
    elif isinstance(start_date, pd.Timestamp):
        normalized_start = start_date.date()
    elif isinstance(start_date, date):
        normalized_start = start_date
    else:
        raise TypeError("start_date must be a date instance")

    if normalized_start > date.today():
        raise ValueError("start_date cannot be in the future")

    start_date = normalized_start

    metadata = get_metadata()
    fetcher = price_fetcher or fetch_price_history
    signals_rows: List[Dict[str, object]] = []
    history_rows: List[Dict[str, object]] = []

    for ticker in filtered_tickers:
        try:
            price_data = fetcher(ticker, start=start_date)
        except Exception as err:
            history_rows.append(
                {
                    "ticker": ticker,
                    "win_rate": 0.0,
                    "average_return": 0.0,
                    "cagr": 0.0,
                    "max_drawdown": 0.0,
                    "total_trades": 0.0,
                    "total_return": 0.0,
                    "status": f"Data error: {err}",
                    "sector": None,
                    "market_cap_billion": None,
                }
            )
            continue

        try:
            result = golden_cross_strategy(ticker, price_data, profit_target)
        except Exception as err:
            history_rows.append(
                {
                    "ticker": ticker,
                    "win_rate": 0.0,
                    "average_return": 0.0,
                    "cagr": 0.0,
                    "max_drawdown": 0.0,
                    "total_trades": 0.0,
                    "total_return": 0.0,
                    "status": f"Strategy error: {err}",
                    "sector": None,
                    "market_cap_billion": None,
                }
            )
            continue

        row_meta = metadata_df.loc[ticker] if ticker in metadata_df.index else None
        win_rate = result.stats.get("win_rate", 0.0)
        cagr = result.stats.get("cagr", 0.0)
        meets_thresholds = win_rate >= win_rate_threshold and cagr > cagr_threshold

        history_rows.append(
            {
                "ticker": ticker,
                "win_rate": win_rate,
                "average_return": result.stats.get("average_return", 0.0),
                "cagr": cagr,
                "max_drawdown": result.stats.get("max_drawdown", 0.0),
                "total_trades": result.stats.get("total_trades", 0.0),
                "total_return": result.stats.get("total_return", 0.0),
                "status": "ok",
                "sector": row_meta["sector"] if row_meta is not None else None,
                "market_cap_billion": row_meta["market_cap_billion"] if row_meta is not None else None,
            }
        )

        if not meets_thresholds:
            continue

        signal = result.last_signal or "None"
        signals_rows.append(
            {
                "ticker": ticker,
                "signal": signal,
                "entry_price": result.entry_price,
                "target_price": result.target_price,
                "historical_win_rate": win_rate,
                "historical_cagr": cagr,
                "last_signal_date": result.last_signal_date,
                "sector": row_meta["sector"] if row_meta is not None else None,
                "market_cap_billion": row_meta["market_cap_billion"] if row_meta is not None else None,
            }
        )

    signals_df = pd.DataFrame(signals_rows)
    history_df = pd.DataFrame(history_rows)
    if not signals_df.empty and "historical_win_rate" in signals_df.columns:
        signals_df = signals_df.sort_values(
            ["signal", "historical_win_rate"], ascending=[False, False]
        )
    if not history_df.empty and "win_rate" in history_df.columns:
        history_df = history_df.sort_values("win_rate", ascending=False)

    return signals_df, history_df


def _format_percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_percentage_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    formatted = df.copy()
    for column in columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].apply(
                lambda value: _format_percentage(value) if pd.notnull(value) else value
            )
    return formatted


def build_streamlit_app() -> None:
    st.set_page_config(page_title="ASX200 Daily Signals", layout="wide")
    st.title("ASX200 Daily Golden Cross Signals")

    metadata = _empty_metadata_frame()
    filtered_metadata = metadata
    search_query = ""
    search_results_limit = 25
    lookup_query = ""
    lookup_category = "All"
    lookup_count = 25
    cap_range: Optional[Tuple[float, float]] = None
    ticker_source = "ASX200 Universe"
    custom_ticker = ""

    with st.sidebar:
        st.header("Universe & Filters")
        ticker_source = st.selectbox(
            "Ticker source",
            options=("ASX200 Universe", "Yahoo Finance Search", "Yahoo Finance Lookup"),
            index=0,
        )

        if ticker_source == "ASX200 Universe":
            metadata = get_metadata().copy()
            sectors = sorted(metadata["sector"].dropna().unique().tolist())
            selected_sectors = st.multiselect(
                "Sectors", options=sectors, default=sectors
            )

            min_cap = float(metadata["market_cap_billion"].min())
            max_cap = float(metadata["market_cap_billion"].max())
            cap_range = st.slider(
                "Market Cap Range (AUD billions)",
                min_value=float(np.floor(min_cap)),
                max_value=float(np.ceil(max_cap)),
                value=(float(np.floor(min_cap)), float(np.ceil(max_cap))),
                step=0.5,
            )

            custom_ticker = st.text_input(
                "Custom ticker override (optional)", value=""
            )

            filtered_metadata = metadata[
                metadata["sector"].isin(selected_sectors)
                & (metadata["market_cap_billion"] >= cap_range[0])
                & (metadata["market_cap_billion"] <= cap_range[1])
            ]
        elif ticker_source == "Yahoo Finance Search":
            search_query = st.text_input("Search query", value="").strip()
            search_results_limit = st.slider(
                "Max search results", min_value=5, max_value=50, value=25, step=5
            )
            if search_query:
                try:
                    metadata = search_ticker_universe(
                        search_query, max_results=search_results_limit
                    )
                except RuntimeError as err:
                    st.error(str(err))
                    metadata = _empty_metadata_frame()
            else:
                st.info("Enter a search term to discover tickers via Yahoo Finance.")
                metadata = _empty_metadata_frame()

            filtered_metadata = metadata
        else:
            lookup_query = st.text_input("Lookup query", value="").strip()
            lookup_category = st.selectbox(
                "Lookup category", options=tuple(LOOKUP_CATEGORIES.keys())
            )
            lookup_count = st.slider(
                "Lookup result count", min_value=5, max_value=100, value=25, step=5
            )
            if lookup_query:
                try:
                    metadata = lookup_ticker_universe(
                        lookup_query, category=lookup_category, count=lookup_count
                    )
                except RuntimeError as err:
                    st.error(str(err))
                    metadata = _empty_metadata_frame()
            else:
                st.info("Enter a lookup term to discover tickers via Yahoo Finance.")
                metadata = _empty_metadata_frame()

            filtered_metadata = metadata

        st.header("Strategy Settings")
        profit_target = st.slider("Profit target", min_value=0.01, max_value=0.25, value=0.05, step=0.01)
        history_years = st.slider("History (years)", min_value=5, max_value=10, value=7)
        win_rate_threshold = st.slider("Minimum win rate", min_value=0.4, max_value=0.9, value=0.55, step=0.01)
        cagr_threshold = st.slider("Minimum CAGR", min_value=-0.2, max_value=0.5, value=0.0, step=0.01)

        run_button = st.button("Run Scan")

    if ticker_source == "ASX200 Universe":
        if custom_ticker:
            custom_ticker = custom_ticker.strip().upper()
            if custom_ticker and custom_ticker not in filtered_metadata.index:
                filtered_metadata = pd.concat(
                    [
                        filtered_metadata,
                        pd.DataFrame(
                            [
                                {
                                    "ticker": custom_ticker,
                                    "sector": "Custom",
                                    "market_cap_billion": np.nan,
                                }
                            ],
                            index=[custom_ticker],
                        ),
                    ]
                )

        if cap_range is not None:
            st.write(
                "Scanning **{count}** tickers between {low:.1f} and {high:.1f} "
                "billion AUD.".format(
                    count=len(filtered_metadata), low=cap_range[0], high=cap_range[1]
                )
            )
    elif ticker_source == "Yahoo Finance Search":
        if search_query:
            st.write(
                f"Scanning **{len(filtered_metadata)}** tickers from Yahoo Finance Search "
                f"results for \"{search_query}\"."
            )
    else:
        if lookup_query:
            st.write(
                f"Scanning **{len(filtered_metadata)}** tickers from Yahoo Finance Lookup "
                f"results for \"{lookup_query}\" ({lookup_category})."
            )

    if ticker_source != "ASX200 Universe" and not filtered_metadata.empty:
        display_columns = [
            col for col in ["ticker", "name", "exchange", "type"] if col in filtered_metadata.columns
        ]
        st.dataframe(filtered_metadata.reset_index(drop=True)[display_columns])

    start_date = date.today() - timedelta(days=history_years * 365)

    tickers_to_scan = [
        ticker
        for ticker in filtered_metadata["ticker"].tolist()
        if isinstance(ticker, str) and ticker.strip()
    ]

    scan_params = {
        "tickers": tuple(tickers_to_scan),
        "start": start_date.isoformat(),
        "profit_target": float(profit_target),
        "win_rate_threshold": float(win_rate_threshold),
        "cagr_threshold": float(cagr_threshold),
    }
    previous_params = st.session_state.get("last_scan_params")
    should_run = run_button or previous_params != scan_params

    if should_run:
        with st.spinner("Running strategy across tickers..."):
            try:
                signals_df, history_df = scan_tickers(
                    tickers_to_scan,
                    start_date=start_date,
                    profit_target=profit_target,
                    win_rate_threshold=win_rate_threshold,
                    cagr_threshold=cagr_threshold,
                )
            except Exception as exc:
                st.error(f"Unable to complete scan: {exc}")
                signals_df = pd.DataFrame()
                history_df = pd.DataFrame()
            else:
                st.session_state["last_scan_params"] = scan_params
                st.session_state["scan_results"] = {
                    "signals": signals_df.copy(),
                    "history": history_df.copy(),
                }
    else:
        cached_results = st.session_state.get("scan_results") or {}
        signals_df = cached_results.get("signals", pd.DataFrame()).copy()
        history_df = cached_results.get("history", pd.DataFrame()).copy()
        if signals_df.empty and history_df.empty:
            # No cached results yet and button not pressed; trigger initial scan on next run.
            st.session_state.pop("last_scan_params", None)

    if not history_df.empty and "status" in history_df.columns:
        error_rows = history_df[history_df["status"].str.contains("Data error", na=False)]
        if not error_rows.empty:
            unique_errors = sorted({ticker for ticker in error_rows["ticker"].dropna()})
            st.warning(
                "Failed to download data for the following tickers: "
                + ", ".join(unique_errors)
            )

    st.subheader("Today's Signals")
    if signals_df.empty:
        st.info("No actionable signals matched the historical filters today.")
    else:
        display_df = _format_percentage_columns(
            signals_df, ["historical_win_rate", "historical_cagr"]
        )
        st.dataframe(display_df)
        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Signals CSV", csv, file_name="asx_signals.csv", mime="text/csv")
        excel_buffer = io.BytesIO()
        display_df.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)
        st.download_button(
            "Download Signals Excel",
            excel_buffer.getvalue(),
            file_name="asx_signals.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.subheader("Historical Performance Snapshot")
    with st.expander("View metrics", expanded=True):
        if history_df.empty:
            st.info("Run the scan to generate historical statistics.")
        else:
            display_history = _format_percentage_columns(
                history_df,
                ["win_rate", "average_return", "cagr", "max_drawdown", "total_return"],
            )
            st.dataframe(display_history)
            csv = display_history.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Historical Stats CSV",
                csv,
                file_name="asx_history.csv",
                mime="text/csv",
            )
            excel_buffer = io.BytesIO()
            display_history.to_excel(excel_buffer, index=False)
            excel_buffer.seek(0)
            st.download_button(
                "Download Historical Stats Excel",
                excel_buffer.getvalue(),
                file_name="asx_history.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    if not history_df.empty:
        selected_ticker = st.selectbox(
            "Select ticker for detail",
            options=["(none)"] + history_df["ticker"].dropna().unique().tolist(),
        )
        if selected_ticker and selected_ticker != "(none)":
            try:
                price_data = fetch_price_history(selected_ticker, start=start_date)
                result = golden_cross_strategy(selected_ticker, price_data, profit_target)
                show_ticker_details(result)
            except Exception as err:  # pragma: no cover - UI fallback
                st.error(f"Unable to load detail for {selected_ticker}: {err}")


def show_ticker_details(result: StrategyResult) -> None:
    st.markdown(f"### Detailed View: {result.ticker}")
    df = result.indicators.copy()
    df = df.dropna()
    df = df.reset_index().rename(columns={"index": "date"})

    if alt is not None:
        base = alt.Chart(df).encode(x="date:T")
        price_line = base.mark_line(color="steelblue").encode(y="price:Q")
        sma50_line = base.mark_line(color="orange").encode(y="sma50:Q")
        sma200_line = base.mark_line(color="green").encode(y="sma200:Q")
        st.altair_chart(alt.layer(price_line, sma50_line, sma200_line).resolve_scale(y="independent"), use_container_width=True)

    equity_df = result.equity_curve.reset_index()
    equity_df.columns = ["date", "equity"]
    if alt is not None:
        st.altair_chart(
            alt.Chart(equity_df).mark_line(color="purple").encode(x="date:T", y="equity:Q"),
            use_container_width=True,
        )
    else:  # pragma: no cover
        st.line_chart(equity_df.set_index("date"))

    trade_df = pd.DataFrame(
        [
            {
                "entry_date": trade.entry_date,
                "exit_date": trade.exit_date,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "return_pct": trade.pct_return,
            }
            for trade in result.trades
        ]
    )
    if trade_df.empty:
        st.info("No completed trades yet for this lookback period.")
    else:
        trade_df["return_pct"] = trade_df["return_pct"].map(_format_percentage)
        st.dataframe(trade_df)


# -------------------
# Self Tests
# -------------------


TEST_ANCHOR_DATE = date(2024, 1, 2)


def _generate_synthetic_price(
    start_price: float,
    days: int,
    drift: float = 0.0005,
    *,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    anchor = end_date or TEST_ANCHOR_DATE
    rng = pd.date_range(end=anchor, periods=days, freq="B")
    price = start_price * (1 + drift) ** np.arange(len(rng))
    df = pd.DataFrame({"Adj Close": price}, index=rng)
    df["Close"] = df["Adj Close"]
    return df


class StrategyTests(unittest.TestCase):
    def test_golden_cross_profit_target_exit(self):
        df = _generate_synthetic_price(100.0, 400, drift=0.002, end_date=TEST_ANCHOR_DATE)
        result = golden_cross_strategy("TEST", df, profit_target=0.05)
        self.assertGreater(result.stats["win_rate"], 0)
        self.assertEqual(result.last_signal, "Hold")

    def test_batch_scan_returns_signals(self):
        metadata = get_metadata()
        tickers = metadata.head(3)["ticker"].tolist()
        synthetic_data = _generate_synthetic_price(50.0, 400, drift=0.002, end_date=TEST_ANCHOR_DATE)

        def fake_fetch(ticker: str, start: date) -> pd.DataFrame:
            return synthetic_data

        signals, history = scan_tickers(
            tickers,
            start_date=TEST_ANCHOR_DATE - timedelta(days=365 * 5),
            profit_target=0.05,
            win_rate_threshold=0.1,
            cagr_threshold=-0.1,
            price_fetcher=fake_fetch,
        )
        self.assertFalse(signals.empty)
        self.assertFalse(history.empty)

    def test_threshold_filters(self):
        df = _generate_synthetic_price(100, 400, drift=-0.001, end_date=TEST_ANCHOR_DATE)
        result = golden_cross_strategy("BEAR", df, profit_target=0.05)
        history = pd.DataFrame(
            [{"ticker": "BEAR", "win_rate": result.stats["win_rate"], "cagr": result.stats["cagr"]}]
        )
        filtered = history[(history["win_rate"] >= 0.55) & (history["cagr"] > 0)]
        self.assertTrue(filtered.empty)

    def test_scan_handles_fetch_error(self):
        tickers = ["GOOD.AX", "BAD.AX"]

        def fake_fetch(ticker: str, start: date) -> pd.DataFrame:
            if ticker == "BAD.AX":
                raise ValueError("Missing data")
            return _generate_synthetic_price(50.0, 200, drift=0.001, end_date=TEST_ANCHOR_DATE)

        signals, history = scan_tickers(
            tickers,
            start_date=TEST_ANCHOR_DATE - timedelta(days=365),
            profit_target=0.05,
            win_rate_threshold=0.0,
            cagr_threshold=-1.0,
            price_fetcher=fake_fetch,
        )

        self.assertIn("BAD.AX", history["ticker"].values)
        bad_row = history.loc[history["ticker"] == "BAD.AX"].iloc[0]
        self.assertEqual(bad_row["win_rate"], 0.0)
        self.assertEqual(bad_row["total_trades"], 0.0)
        self.assertTrue(str(bad_row["status"]).startswith("Data error"))
        self.assertFalse(signals.empty)

    def test_optional_repair_fallback(self):
        calls: List[Dict[str, object]] = []

        def fake_download(*args, **kwargs):
            calls.append(dict(kwargs))
            if "repair" in kwargs:
                raise TypeError("unexpected keyword argument 'repair'")
            return pd.DataFrame({"value": [1.0]})

        result = _call_with_optional_repair(fake_download, kwargs={"start": TEST_ANCHOR_DATE})
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
        self.assertGreaterEqual(len(calls), 2)
        self.assertIn("repair", calls[0])
        self.assertNotIn("repair", calls[-1])


class SimpleTestResult:
    def __init__(self, passed: bool):
        self.passed = passed


def run_tests() -> SimpleTestResult:
    suite = unittest.TestSuite()
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(StrategyTests))
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    return SimpleTestResult(result.wasSuccessful())


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="ASX200 Golden Cross signal provider")
    parser.add_argument("--run-tests", action="store_true", help="Execute self-tests and exit")
    args = parser.parse_args(argv)
    if args.run_tests:
        outcome = run_tests()
        if not outcome.passed:
            raise SystemExit(1)
        return

    build_streamlit_app()


if __name__ == "__main__":
    main()
