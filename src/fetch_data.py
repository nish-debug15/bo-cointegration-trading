import yfinance as yf
import pandas as pd
from pathlib import Path

from data_loader import ROOT_DIR

def fetch_ko_pep():
    ROOT_DIR = Path(__file__).resolve().parents[1]
    data_dir = ROOT_DIR / "data"

    data_dir.mkdir(parents=True, exist_ok=True)

    tickers = ["KO", "PEP"]

    data = yf.download(
        tickers=tickers,
        start="2000-01-01",
        auto_adjust=True,
        progress=False,
        threads=False      
    )

    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]

    data = data.dropna()
    data.columns = ["KO", "PEP"]

    out_path = data_dir / "ko_pep.csv"
    data.to_csv(out_path)

    print(f"Saved {out_path.resolve()}")
    print(data.tail())

if __name__ == "__main__":
    fetch_ko_pep()
