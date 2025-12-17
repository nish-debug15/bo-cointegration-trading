import numpy as np

def compute_spread(log_prices, weights):
    return log_prices.values @ weights

def zscore(spread):
    mean = spread.mean()
    std = spread.std()
    return (spread - mean) / std, mean, std
