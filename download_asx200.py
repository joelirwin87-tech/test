"""Download full historical OHLCV data for ASX200 tickers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd
import yfinance as yf

ASX200_TICKERS: List[str] = [
    "A2M.AX","ABB.AX","ABC.AX","ABP.AX","AD8.AX","AGL.AX","AHY.AX","ALD.AX","ALG.AX","ALQ.AX",
    "ALL.AX","ALU.AX","ALX.AX","AMA.AX","AMC.AX","AMP.AX","ANN.AX","ANZ.AX","APA.AX","APE.AX",
    "APM.AX","APX.AX","ARB.AX","ARF.AX","ARG.AX","ASX.AX","AUB.AX","AWC.AX","AX1.AX","AZJ.AX",
    "BAP.AX","BEN.AX","BFG.AX","BHP.AX","BKL.AX","BLD.AX","BPT.AX","BRG.AX","BSL.AX","BVS.AX",
    "BWP.AX","BXB.AX","CAR.AX","CBA.AX","CCP.AX","CCX.AX","CDP.AX","CGF.AX","CHC.AX","CHN.AX",
    "CIA.AX","CIM.AX","CKF.AX","CLW.AX","CMW.AX","CNI.AX","COF.AX","COH.AX","COL.AX","CPU.AX",
    "CQR.AX","CSL.AX","CSR.AX","CTD.AX","CUV.AX","CWY.AX","CXO.AX","DBI.AX","DDR.AX","DEG.AX",
    "DMP.AX","DOW.AX","DRR.AX","DXS.AX","EBO.AX","EDV.AX","ELD.AX","EML.AX","EHE.AX","EVN.AX",
    "EVT.AX","FBU.AX","FLT.AX","FMG.AX","FPH.AX","GMG.AX","GMA.AX","GNC.AX","GOR.AX","GPT.AX",
    "GUD.AX","GWA.AX","HLS.AX","HMC.AX","HSN.AX","HUB.AX","HVN.AX","IAG.AX","IEL.AX","IFL.AX",
    "IFT.AX","IGO.AX","ILU.AX","IMU.AX","INA.AX","IPL.AX","IPH.AX","IRE.AX","IVC.AX","JHX.AX",
    "JIN.AX","JBH.AX","JLG.AX","KAR.AX","KGN.AX","LLC.AX","LNK.AX","LTR.AX","LYC.AX","MAQ.AX",
    "MCY.AX","MEZ.AX","MFG.AX","MGR.AX","MIN.AX","MND.AX","MPL.AX","MP1.AX","MQG.AX","MSB.AX",
    "MTS.AX","MYR.AX","MYX.AX","NAB.AX","NAN.AX","NCK.AX","NCM.AX","NHF.AX","NIC.AX","NSR.AX",
    "NST.AX","NUF.AX","NWH.AX","NWL.AX","NXT.AX","OBL.AX","OML.AX","ORA.AX","ORG.AX","ORI.AX",
    "OSH.AX","OZL.AX","PDN.AX","PGH.AX","PLS.AX","PME.AX","PNI.AX","PNV.AX","PPM.AX","PRN.AX",
    "PRU.AX","PTM.AX","QAN.AX","QBE.AX","QUB.AX","REA.AX","RED.AX","REG.AX","REH.AX","RHC.AX",
    "RIO.AX","RMD.AX","RRL.AX","RSG.AX","RWC.AX","S32.AX","SAR.AX","SBM.AX","SCG.AX","SCP.AX",
    "SDF.AX","SEK.AX","SFR.AX","SGM.AX","SGP.AX","SGR.AX","SHL.AX","SIQ.AX","SIT.AX","SKC.AX",
    "SKI.AX","SKT.AX","SLR.AX","SLK.AX","SM1.AX","SNZ.AX","SOL.AX","SPK.AX","SPL.AX","SQ2.AX"
]

DATA_DIR = Path("data")
EXPECTED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _prepare_download_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    if not isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.copy()
        frame.index = pd.to_datetime(frame.index, errors="coerce")

    frame = frame.loc[~frame.index.duplicated()].copy()

    normalized = frame.reset_index().rename(columns={"index": "Date", "Date": "Date"})
    normalized["Date"] = pd.to_datetime(normalized["Date"], utc=True, errors="coerce").dt.tz_localize(None)
    normalized = normalized.dropna(subset=["Date"]).sort_values("Date")

    for column in EXPECTED_COLUMNS[1:]:
        if column not in normalized.columns:
            normalized[column] = pd.NA

    normalized = normalized[EXPECTED_COLUMNS]
    return normalized


def _download_history(ticker: str) -> pd.DataFrame:
    kwargs = {
        "period": "max",
        "interval": "1d",
        "auto_adjust": False,
        "progress": False,
        "threads": True,
        "rounding": True,
    }
    data = yf.download(ticker, **kwargs)
    return _prepare_download_frame(data)


def _save_history(ticker: str, data: pd.DataFrame) -> None:
    if data.empty:
        raise ValueError(f"No data available for {ticker}")
    output_path = DATA_DIR / f"{ticker}.csv"
    data.to_csv(output_path, index=False)


def download_all(tickers: Iterable[str]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for ticker in tickers:
        try:
            history = _download_history(ticker)
            if history.empty:
                print(f"No data for {ticker}")
                continue
            _save_history(ticker, history)
            print(f"Saved {ticker}")
        except Exception as exc:  # pragma: no cover - network interactions
            print(f"Failed {ticker}: {exc}")


if __name__ == "__main__":
    download_all(ASX200_TICKERS)
