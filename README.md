# Improving Basket Trading Using Bayesian Optimization

This project explores how **Bayesian Optimization (BO)** can be used to improve
cointegration-based trading strategies by directly optimizing **trading performance**
instead of relying purely on statistical cointegration tests.

Traditional approaches (e.g., Johansen test) produce in-sample optimal cointegrating
vectors, but these weights often fail to generalize out-of-sample.
This project reframes basket construction as a **black-box optimization problem**
where the objective is out-of-sample tradability.

---

## Core Idea

Instead of trusting cointegration vectors derived purely from statistical tests,
we treat the trading strategy itself as the objective.

Bayesian Optimization is used to search for:
- Cointegrating weights
- Trading entry thresholds

that maximize **risk-adjusted performance** on rolling out-of-sample windows.

---

## Key Features

- Real market data (Yahoo Finance)
- Johansen cointegration baseline
- Bayesian Optimization with Gaussian Processes
- Rolling out-of-sample backtesting
- Adaptive entry thresholds
- Strict no-lookahead discipline
- Research-grade experiment design

---

## Project Structure

```text
beyond-johansen/
│
├── data/
│   └── ko_pep.csv
│
├── src/
│   ├── config.py
│   ├── fetch_data.py
│   ├── data_loader.py
│   ├── johansen.py
│   ├── spread.py
│   ├── trading.py
│   ├── metrics.py
│   ├── bayes_opt.py
│   └── rolling.py
│
├── main.py
├── requirements.txt
└── README.md
```
---

## Data

- Assets: **KO (Coca-Cola)** and **PEP (Pepsi)**
- Source: Yahoo Finance
- Frequency: Daily adjusted close prices
- Period: 2000 → present

Fetch data by running:

```bash
python src/fetch_data.py
