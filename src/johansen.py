import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def johansen_weights(log_prices):
    result = coint_johansen(log_prices.values, det_order=0, k_ar_diff=1)
    w = result.evec[:, 0]
    w = w / w[0]
    return w
