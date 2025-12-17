import numpy as np

def sharpe(pnl):
    if pnl.std() == 0:
        return 0.0
    return np.sqrt(252) * pnl.mean() / pnl.std()
