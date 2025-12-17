import time

from .johansen import johansen_weights
from .bayes_opt import optimize_weight_and_entry
from .spread import compute_spread, zscore
from .trading import trade
from .metrics import sharpe


def rolling_backtest(prices, cfg):
    results = []

    n_windows = (len(prices) - cfg.TRAIN_DAYS - cfg.TEST_DAYS) // cfg.STEP_DAYS
    print(f"Starting rolling backtest with {n_windows} windows")

    start_time = time.time()
    window_id = 0

    for i in range(0, len(prices) - cfg.TRAIN_DAYS - cfg.TEST_DAYS, cfg.STEP_DAYS):
        window_id += 1
        print(f"\nWindow {window_id}/{n_windows}")

        train = prices.iloc[i : i + cfg.TRAIN_DAYS]
        test = prices.iloc[i + cfg.TRAIN_DAYS : i + cfg.TRAIN_DAYS + cfg.TEST_DAYS]

        w_j = johansen_weights(train)

        spread_train_j = compute_spread(train, w_j)
        z_train_j, mean_j, std_j = zscore(spread_train_j)

        spread_test_j = compute_spread(test, w_j)
        z_test_j = (spread_test_j - mean_j) / std_j

        pnl_j = trade(z_test_j, cfg.ENTRY_Z, cfg.EXIT_Z)
        sharpe_j = sharpe(pnl_j)

        print("  Optimizing weights + entry via BO...")

        w_bo, entry_bo = optimize_weight_and_entry(
            train,
            cfg.ENTRY_Z,
            cfg.EXIT_Z
        )

        spread_train_bo = compute_spread(train, w_bo)
        z_train_bo, mean_bo, std_bo = zscore(spread_train_bo)

        spread_test_bo = compute_spread(test, w_bo)
        z_test_bo = (spread_test_bo - mean_bo) / std_bo

        pnl_bo = trade(z_test_bo, entry_bo, cfg.EXIT_Z)
        sharpe_bo = sharpe(pnl_bo)

        
        elapsed_min = (time.time() - start_time) / 60

        print(f"  Johansen Sharpe: {sharpe_j:.3f}")
        print(f"  BO Sharpe:       {sharpe_bo:.3f}")
        print(f"  BO Entry Z:      {entry_bo:.2f}")
        print(f"  Elapsed time:    {elapsed_min:.1f} min")

        results.append({
            "window": window_id,
            "johansen_sharpe": sharpe_j,
            "bo_sharpe": sharpe_bo,
            "bo_entry": entry_bo
        })

    return results
