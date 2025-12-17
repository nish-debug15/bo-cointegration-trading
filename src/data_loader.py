import pandas as pd
import numpy as np
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "ko_pep.csv"

def load_prices():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found at {DATA_PATH}. "
            f"Run: python src/fetch_data.py first."
        )

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df = df.sort_index()
    return np.log(df)
