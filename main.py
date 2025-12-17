import pandas as pd
import src.config as cfg
from src.data_loader import load_prices
from src.rolling import rolling_backtest

prices = load_prices()
results = rolling_backtest(prices, cfg)

df = pd.DataFrame(results)

print("\n========== SUMMARY ==========")
print(df[["johansen_sharpe", "bo_sharpe"]].describe())

win_rate = (df["bo_sharpe"] > df["johansen_sharpe"]).mean()
print(f"\nBO win rate: {win_rate:.2%}")

print("\nAverage Entry Z (BO):", df["bo_entry"].mean())
