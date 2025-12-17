import numpy as np
from skopt import gp_minimize
from skopt.space import Real

from .spread import compute_spread, zscore
from .trading import trade
from .metrics import sharpe

def optimize_weight_and_entry(train_prices, entry, exit):
    def objective(params):
        w2, entry_z = params

        weights = np.array([1.0, w2])
        spread = compute_spread(train_prices, weights)
        z, _, _ = zscore(spread)

        pnl = trade(z, entry_z, exit)

        if pnl.std() == 0:
            return 10.0

        return -sharpe(pnl)

    res = gp_minimize(
        objective,
        dimensions=[
            Real(-5.0, 5.0),    
            Real(1.0, 2.5)      
        ],
        n_calls=25,
        n_initial_points=12,
        random_state=42
    )

    w2_opt, entry_opt = res.x
    return np.array([1.0, w2_opt]), entry_opt
