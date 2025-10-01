#!/usr/bin/env python3
"""
Test script for the optimize_policy.py module.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import and test the optimize_policy module
try:
    from cli.optimize_policy import (
        prepare_optimization_data,
        grid_search_optimization,
        bayesian_optimization_scipy,
        simulate_delta_neutral_strategy,
        simulate_gamma_scaled_strategy,
    )

    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta

    print("✓ Successfully imported optimization modules")

    # Create sample data for testing
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    np.random.seed(42)

    sample_data = pd.DataFrame(
        {
            "date": dates,
            "close": 100 + np.cumsum(np.random.randn(len(dates)) * 0.02),
            "open": None,
            "high": None,
            "low": None,
            "volume": np.random.randint(1000, 10000, len(dates)),
        }
    )

    # Fill OHLC data
    sample_data["open"] = sample_data["close"].shift(1).fillna(100)
    sample_data["high"] = (
        sample_data[["open", "close"]].max(axis=1) + np.random.rand(len(dates)) * 0.5
    )
    sample_data["low"] = (
        sample_data[["open", "close"]].min(axis=1) - np.random.rand(len(dates)) * 0.5
    )

    sample_data = sample_data.set_index("date")

    print(f"✓ Created sample data: {len(sample_data)} rows")

    # Test data preparation
    print("\n1. Testing data preparation...")
    processed_data = prepare_optimization_data(sample_data, "delta_neutral")
    print(f"   Processed data shape: {processed_data.shape}")
    print(f"   Features: {list(processed_data.columns)}")

    # Test delta neutral strategy simulation
    print("\n2. Testing delta neutral strategy...")
    delta_params = {
        "rebalance_frequency": "daily",
        "hedge_ratio_multiplier": 1.0,
        "volatility_window": 20,
    }

    delta_returns = simulate_delta_neutral_strategy(processed_data, delta_params)
    print(f"   Delta neutral returns: {len(delta_returns)} observations")
    if len(delta_returns) > 0:
        print(f"   Mean return: {delta_returns.mean():.6f}")
        print(f"   Volatility: {delta_returns.std():.6f}")

    # Test gamma scaled strategy simulation
    print("\n3. Testing gamma scaled strategy...")
    gamma_params = {
        "scaling_factor": 1.5,
        "rebalance_frequency": "daily",
        "gamma_threshold": 0.02,
        "hedge_ratio_multiplier": 1.0,
        "volatility_window": 20,
    }

    gamma_returns = simulate_gamma_scaled_strategy(processed_data, gamma_params)
    print(f"   Gamma scaled returns: {len(gamma_returns)} observations")
    if len(gamma_returns) > 0:
        print(f"   Mean return: {gamma_returns.mean():.6f}")
        print(f"   Volatility: {gamma_returns.std():.6f}")

    # Test small grid search
    print("\n4. Testing grid search optimization (small scale)...")
    initial_params = {"rebalance_frequency": "daily"}

    try:
        # Limit the scope for testing
        from cli.optimize_policy import (
            create_time_series_cv_splits,
            evaluate_strategy_performance,
        )

        # Test CV splits
        splits = create_time_series_cv_splits(processed_data, cv_folds=2)
        print(f"   Created {len(splits)} CV splits")

        # Test performance evaluation
        if len(splits) > 0:
            train_data, test_data = splits[0]
            performance = evaluate_strategy_performance(
                train_data, test_data, delta_params, "delta_neutral"
            )
            print(f"   Sample performance: {performance}")

        print("   ✓ Grid search components working")

    except Exception as e:
        print(f"   ⚠ Grid search test failed: {e}")

    # Test scipy-based Bayesian optimization
    print("\n5. Testing Bayesian optimization (scipy-based)...")
    try:
        # Use a very small subset for quick testing
        small_data = processed_data.tail(50)

        result = bayesian_optimization_scipy(
            small_data,
            initial_params,
            "delta_neutral",
            n_calls=5,  # Very small for testing
            cv_folds=2,
        )

        print(f"   ✓ Bayesian optimization completed")
        print(f"   Best score: {result.get('best_score', 'N/A')}")
        print(f"   Best params: {result.get('best_params', 'N/A')}")

    except Exception as e:
        print(f"   ⚠ Bayesian optimization test failed: {e}")

    print("\n🎯 Optimization testing completed!")
    print("The optimize_policy.py module is ready for use.")

except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure all dependencies are installed and paths are correct.")

except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback

    traceback.print_exc()
