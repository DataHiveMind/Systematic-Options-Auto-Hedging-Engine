"""
==================================================================================
Title: Optimize Policy CLI
Author: Kenneth LeGare
Date: 2024-06-25

Description:
    This module provides a command-line interface (CLI) for optimizing hedging policies
    using historical data and various optimization algorithms.

Dependencies:
    - argparse
    - logging
    - pandas
    - numpy
    - scipy
    - sklearn
    - statsmodels
    - openbb
==================================================================================
"""

# Standard Library
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional

# Add current directory to path for local imports BEFORE other imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third Party Libraries
import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.model_selection import ParameterGrid
import warnings

warnings.filterwarnings("ignore")

# Try to import Bayesian optimization libraries
try:
    import importlib.util

    skopt_spec = importlib.util.find_spec("skopt")
    BAYESIAN_AVAILABLE = skopt_spec is not None
    if not BAYESIAN_AVAILABLE:
        print(
            "Warning: scikit-optimize not available. Bayesian optimization will use scipy.optimize instead."
        )
except ImportError:
    BAYESIAN_AVAILABLE = False
    print(
        "Warning: scikit-optimize not available. Bayesian optimization will use scipy.optimize instead."
    )

# OpenBB for data
try:
    from openbb import obb

    OPENBB_AVAILABLE = True
except ImportError:
    OPENBB_AVAILABLE = False
    print("Warning: OpenBB not available. Using mock data loader.")


# Local Modules (now that path is set up)
def load_data(tickers: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Mock data loader when OpenBB is not available."""
    if OPENBB_AVAILABLE:
        try:
            # Try to use OpenBB data loader
            from hedging_engine.data_loader import load_data as openbb_load_data

            return openbb_load_data(tickers, start_date, end_date)
        except ImportError:
            pass

    # Fallback to mock data
    print("Using mock data for demonstration...")
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    ticker_list = [t.strip() for t in tickers.split(",")]

    np.random.seed(42)  # For reproducible mock data
    data = {}
    for ticker in ticker_list:
        # Generate realistic mock price data
        n_days = len(dates)
        returns = np.random.normal(0.001, 0.02, n_days)  # Daily returns
        prices = 100 * np.exp(np.cumsum(returns))  # Exponential random walk
        data[f"{ticker}_price"] = prices
        data[f"{ticker}_volume"] = np.random.lognormal(15, 1, n_days)

    df = pd.DataFrame(data, index=dates)
    return df


def prepare_optimization_data(
    price_data: pd.DataFrame, policy_type: str
) -> pd.DataFrame:
    """
    Prepare data for optimization by adding technical indicators and features.

    Args:
        price_data (pd.DataFrame): Raw price data from data loader.
        policy_type (str): Type of hedging policy.

    Returns:
        pd.DataFrame: Processed data with features for optimization.
    """
    try:
        # Create a copy to avoid modifying original data
        data = price_data.copy()

        # Ensure we have required columns
        required_cols = ["close", "open", "high", "low", "volume"]
        available_cols = [col for col in required_cols if col in data.columns]

        if not available_cols:
            print("Warning: No standard OHLCV columns found. Using available data.")
            return data

        # Calculate returns
        if "close" in data.columns:
            data["returns"] = data["close"].pct_change()
            data["log_returns"] = np.log(data["close"] / data["close"].shift(1))

            # Calculate rolling volatility (multiple windows)
            for window in [10, 20, 60]:
                data[f"volatility_{window}"] = data["returns"].rolling(
                    window
                ).std() * np.sqrt(252)

            # Calculate moving averages
            for window in [5, 10, 20, 50]:
                data[f"ma_{window}"] = data["close"].rolling(window).mean()

            # Calculate price momentum indicators
            data["momentum_5"] = data["close"] / data["close"].shift(5) - 1
            data["momentum_20"] = data["close"] / data["close"].shift(20) - 1

            # Calculate RSI (only if we have numeric data)
            try:
                if pd.api.types.is_numeric_dtype(data["close"]):
                    delta = data["close"].diff().astype(float)
                    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                    loss = (-delta).where(delta < 0, 0).rolling(window=14).mean()
                    with np.errstate(divide="ignore", invalid="ignore"):
                        rs = gain / (
                            loss + 1e-10
                        )  # Add small epsilon to avoid division by zero
                        data["rsi"] = 100 - (100 / (1 + rs))
            except Exception:
                # Skip RSI calculation if it fails
                pass

        # Add policy-specific features
        if policy_type == "delta_neutral":
            # Add delta-related features
            if "returns" in data.columns:
                data["return_volatility"] = data["returns"].rolling(30).std()
                data["skewness"] = data["returns"].rolling(60).skew()
                data["kurtosis"] = data["returns"].rolling(60).kurt()

        elif policy_type == "gamma_scaled":
            # Add gamma-related features
            if "returns" in data.columns:
                # Gamma is related to second-order price movements
                data["return_squared"] = data["returns"] ** 2
                data["gamma_proxy"] = data["return_squared"].rolling(20).mean()

        # Drop rows with NaN values
        data = data.dropna()

        print(
            f"Prepared optimization data: {len(data)} rows, {len(data.columns)} features"
        )
        return data

    except Exception as e:
        print(f"Error in prepare_optimization_data: {e}")
        return price_data


def create_time_series_cv_splits(data: pd.DataFrame, cv_folds: int = 3) -> list:
    """
    Create time series cross-validation splits.

    Args:
        data (pd.DataFrame): Input data with time series index.
        cv_folds (int): Number of cross-validation folds.

    Returns:
        list: List of (train_data, test_data) tuples.
    """
    splits = []

    # Calculate split points
    total_length = len(data)
    test_size = total_length // (cv_folds + 1)

    for i in range(cv_folds):
        # Expanding window approach for time series
        train_end = total_length - (cv_folds - i) * test_size
        test_start = train_end
        test_end = test_start + test_size

        if test_end > total_length:
            test_end = total_length

        train_data = data.iloc[:train_end]
        test_data = data.iloc[test_start:test_end]

        if len(train_data) > 0 and len(test_data) > 0:
            splits.append((train_data, test_data))

    return splits


def evaluate_strategy_performance(
    train_data: pd.DataFrame, test_data: pd.DataFrame, params: dict, policy_type: str
) -> dict:
    """
    Evaluate strategy performance given parameters.

    Args:
        train_data (pd.DataFrame): Training data for parameter estimation.
        test_data (pd.DataFrame): Test data for performance evaluation.
        params (dict): Strategy parameters.
        policy_type (str): Type of hedging policy.

    Returns:
        dict: Performance metrics.
    """
    try:
        # Simulate hedging strategy performance
        if "returns" not in test_data.columns or "close" not in test_data.columns:
            # Fallback: use basic return calculation
            if "close" in test_data.columns:
                returns = test_data["close"].pct_change().dropna()
            else:
                # If no price data, return default metrics
                return {
                    "total_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "volatility": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.5,
                }
        else:
            returns = test_data["returns"].dropna()

        if len(returns) == 0:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "volatility": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.5,
            }

        # Calculate hedge performance based on policy type
        if policy_type == "delta_neutral":
            hedge_returns = simulate_delta_neutral_strategy(test_data, params)
        elif policy_type == "gamma_scaled":
            hedge_returns = simulate_gamma_scaled_strategy(test_data, params)
        else:
            hedge_returns = returns  # No hedging

        # Calculate performance metrics
        if len(hedge_returns) == 0:
            hedge_returns = returns

        try:
            # Convert to numpy array for safer calculations
            returns_array = (
                hedge_returns.values
                if hasattr(hedge_returns, "values")
                else np.array(hedge_returns)
            )

            # Calculate cumulative return
            if len(returns_array) > 0:
                cumulative_return = np.prod(1 + returns_array)
                total_return = float(cumulative_return - 1)
            else:
                total_return = 0.0

            # Ensure finite value
            if not np.isfinite(total_return):
                total_return = 0.0

        except (TypeError, ValueError, OverflowError):
            total_return = 0.0
        volatility = hedge_returns.std() * np.sqrt(252)  # Annualized

        # Sharpe ratio (assuming risk-free rate of 0)
        if volatility > 0:
            sharpe_ratio = hedge_returns.mean() / hedge_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Maximum drawdown
        cumulative_returns = (1 + hedge_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        win_rate = (hedge_returns > 0).mean()

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "volatility": volatility,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "num_trades": len(hedge_returns),
        }

    except Exception as e:
        print(f"Error in evaluate_strategy_performance: {e}")
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "volatility": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.5,
        }


def simulate_delta_neutral_strategy(data: pd.DataFrame, params: dict) -> pd.Series:
    """
    Simulate delta neutral hedging strategy.

    Args:
        data (pd.DataFrame): Market data.
        params (dict): Strategy parameters.

    Returns:
        pd.Series: Strategy returns.
    """
    try:
        if "returns" not in data.columns:
            return pd.Series([])

        returns = data["returns"].copy()

        # Get parameters
        hedge_ratio_multiplier = params.get("hedge_ratio_multiplier", 1.0)
        volatility_window = params.get("volatility_window", 20)

        # Calculate dynamic hedge ratios
        volatility = returns.rolling(volatility_window).std()

        # Simple delta neutral simulation
        # In practice, this would use actual options Greeks
        hedge_ratios = []
        for i in range(len(returns)):
            if i < volatility_window:
                hedge_ratio = 0.5  # Default
            else:
                vol = volatility.iloc[i]
                if pd.isna(vol):
                    vol = 0.2  # Default volatility

                # Adjust hedge ratio based on volatility and parameters
                base_hedge = 0.5  # Neutral delta
                vol_adjustment = vol * hedge_ratio_multiplier
                hedge_ratio = min(max(base_hedge - vol_adjustment, -1.0), 1.0)

            hedge_ratios.append(hedge_ratio)

        hedge_ratios = pd.Series(hedge_ratios, index=returns.index)

        # Calculate hedged returns
        # Simplified: hedge_return = stock_return + hedge_ratio * (-stock_return)
        hedged_returns = returns * (1 - hedge_ratios)

        # Apply rebalancing frequency
        rebalance_freq = params.get("rebalance_frequency", "daily")
        if rebalance_freq == "weekly":
            # Reduce trading by averaging weekly
            hedged_returns = hedged_returns.resample("W").mean().dropna()
        elif rebalance_freq == "biweekly":
            hedged_returns = hedged_returns.resample("2W").mean().dropna()

        return hedged_returns.dropna()

    except Exception as e:
        print(f"Error in simulate_delta_neutral_strategy: {e}")
        return pd.Series([])


def simulate_gamma_scaled_strategy(data: pd.DataFrame, params: dict) -> pd.Series:
    """
    Simulate gamma scaled hedging strategy.

    Args:
        data (pd.DataFrame): Market data.
        params (dict): Strategy parameters.

    Returns:
        pd.Series: Strategy returns.
    """
    try:
        if "returns" not in data.columns:
            return pd.Series([])

        returns = data["returns"].copy()

        # Get parameters
        scaling_factor = params.get("scaling_factor", 1.5)
        gamma_threshold = params.get("gamma_threshold", 0.02)
        hedge_ratio_multiplier = params.get("hedge_ratio_multiplier", 1.0)
        volatility_window = params.get("volatility_window", 20)

        # Calculate gamma proxy (second-order price movements)
        returns_squared = returns**2
        gamma_proxy = returns_squared.rolling(volatility_window).mean()

        # Calculate hedge ratios based on gamma scaling
        hedge_ratios = []
        for i in range(len(returns)):
            if i < volatility_window:
                hedge_ratio = 0.5  # Default
            else:
                gamma = gamma_proxy.iloc[i]
                if pd.isna(gamma):
                    gamma = 0.01  # Default gamma

                # Scale hedge ratio based on gamma
                if gamma > gamma_threshold:
                    scale = min(gamma * scaling_factor, 2.0)  # Cap scaling
                else:
                    scale = 1.0

                hedge_ratio = 0.5 * scale * hedge_ratio_multiplier
                hedge_ratio = min(max(hedge_ratio, 0.0), 2.0)  # Constrain

            hedge_ratios.append(hedge_ratio)

        hedge_ratios = pd.Series(hedge_ratios, index=returns.index)

        # Calculate hedged returns with gamma scaling
        hedged_returns = returns * (1 - hedge_ratios * 0.5)

        # Apply rebalancing frequency
        rebalance_freq = params.get("rebalance_frequency", "daily")
        if rebalance_freq == "weekly":
            hedged_returns = hedged_returns.resample("W").mean().dropna()
        elif rebalance_freq == "biweekly":
            hedged_returns = hedged_returns.resample("2W").mean().dropna()

        return hedged_returns.dropna()

    except Exception as e:
        print(f"Error in simulate_gamma_scaled_strategy: {e}")
        return pd.Series([])


def aggregate_cv_metrics(cv_metrics: list) -> dict:
    """
    Aggregate cross-validation metrics.

    Args:
        cv_metrics (list): List of metric dictionaries from each CV fold.

    Returns:
        dict: Aggregated metrics.
    """
    if not cv_metrics:
        return {}

    # Get all metric names
    metric_names = cv_metrics[0].keys()

    aggregated = {}
    for metric in metric_names:
        values = [m[metric] for m in cv_metrics if metric in m]
        if values:
            aggregated[f"{metric}_mean"] = np.mean(values)
            aggregated[f"{metric}_std"] = np.std(values)
            aggregated[f"{metric}_min"] = np.min(values)
            aggregated[f"{metric}_max"] = np.max(values)

    return aggregated


def optimize_policy(
    tickers: str,
    start_date: str,
    end_date: str,
    initial_params: dict,
    optimization_method: str = "grid_search",
    output_file: Optional[str] = None,
    policy_type: str = "delta_neutral",
    n_calls: int = 50,
    cv_folds: int = 3,
) -> dict:
    """
    Optimize hedging policy parameters using historical data.

    Args:
        tickers (str): Comma-separated string of stock tickers.
        start_date (str): Start date for historical data (YYYY-MM-DD).
        end_date (str): End date for historical data (YYYY-MM-DD).
        initial_params (dict): Initial parameters for the hedging strategy.
        optimization_method (str): Optimization method to use. Defaults to "grid_search".
        output_file (Optional[str]): File path to save optimized parameters. If None, does not save.
        policy_type (str): Type of hedging policy ('delta_neutral' or 'gamma_scaled').
        n_calls (int): Number of calls for Bayesian optimization.
        cv_folds (int): Number of cross-validation folds.

    Returns:
        dict: Optimized parameters with performance metrics.
    """
    print(f"Starting optimization for {policy_type} policy...")
    print(f"Tickers: {tickers}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Method: {optimization_method}")

    # Load historical data
    try:
        price_data = load_data(tickers, start_date, end_date)
        if price_data.empty:
            print("No data available for the specified date range.")
            return {}
    except Exception as e:
        print(f"Error loading data: {e}")
        return {}

    # Prepare data for optimization
    processed_data = prepare_optimization_data(price_data, policy_type)

    print(f"Loaded {len(processed_data)} data points for optimization")

    # Run optimization based on method
    if optimization_method == "grid_search":
        optimized_params = grid_search_optimization(
            processed_data, initial_params, policy_type, cv_folds
        )
    elif optimization_method == "bayesian_optimization":
        optimized_params = bayesian_optimization(
            processed_data, initial_params, policy_type, n_calls, cv_folds
        )
    else:
        print(f"Unknown optimization method: {optimization_method}")
        return {}

    # Add metadata to results
    optimized_params.update(
        {
            "optimization_method": optimization_method,
            "policy_type": policy_type,
            "data_points": len(processed_data),
            "optimization_date": datetime.now().isoformat(),
            "tickers": tickers,
            "date_range": f"{start_date} to {end_date}",
        }
    )

    # Save optimized parameters to file if specified
    if output_file:
        import json

        with open(output_file, "w") as f:
            json.dump(optimized_params, f, indent=2, default=str)
        print(f"Optimized parameters saved to {output_file}")

    return optimized_params


def grid_search_optimization(
    processed_data: pd.DataFrame,
    initial_params: dict,
    policy_type: str,
    cv_folds: int = 3,
) -> dict:
    """
    Perform grid search optimization of hedging strategy parameters.

    Args:
        processed_data (pd.DataFrame): Processed historical data with features.
        initial_params (dict): Initial parameters for the hedging strategy.
        policy_type (str): Type of policy being optimized.
        cv_folds (int): Number of cross-validation folds.

    Returns:
        dict: Optimized parameters with performance metrics.
    """
    print("Starting grid search optimization...")

    # Define parameter search spaces based on policy type
    if policy_type == "delta_neutral":
        param_grid = {
            "rebalance_frequency": ["daily", "weekly", "biweekly"],
            "delta_threshold": [0.05, 0.1, 0.15, 0.2],
            "volatility_window": [10, 20, 30, 60],
            "hedge_ratio_multiplier": [0.8, 0.9, 1.0, 1.1, 1.2],
        }
    elif policy_type == "gamma_scaled":
        param_grid = {
            "scaling_factor": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            "rebalance_frequency": ["daily", "weekly", "biweekly"],
            "gamma_threshold": [0.01, 0.02, 0.05, 0.1],
            "volatility_window": [10, 20, 30, 60],
            "hedge_ratio_multiplier": [0.8, 0.9, 1.0, 1.1, 1.2],
        }
    else:
        print(f"Unknown policy type: {policy_type}")
        return initial_params

    # Generate all parameter combinations
    param_combinations = list(ParameterGrid(param_grid))
    print(f"Testing {len(param_combinations)} parameter combinations...")

    best_params = None
    best_score = float("-inf")
    best_metrics = None
    results = []

    # Evaluate each parameter combination
    for i, params in enumerate(param_combinations):
        if i % 10 == 0:
            print(f"Progress: {i + 1}/{len(param_combinations)} combinations tested")

        try:
            # Perform cross-validation
            cv_scores = []
            cv_metrics = []

            # Split data for cross-validation
            data_splits = create_time_series_cv_splits(processed_data, cv_folds)

            for train_data, test_data in data_splits:
                # Calculate strategy performance
                performance = evaluate_strategy_performance(
                    train_data, test_data, params, policy_type
                )
                cv_scores.append(performance["sharpe_ratio"])
                cv_metrics.append(performance)

            # Calculate average performance across folds
            avg_score = np.mean(cv_scores)
            avg_metrics = aggregate_cv_metrics(cv_metrics)

            # Track results
            result = {
                "params": params.copy(),
                "cv_score": avg_score,
                "cv_std": np.std(cv_scores),
                "metrics": avg_metrics,
            }
            results.append(result)

            # Update best parameters
            if avg_score > best_score:
                best_score = avg_score
                best_params = params.copy()
                best_metrics = avg_metrics

        except Exception as e:
            print(f"Error evaluating parameters {params}: {e}")
            continue

    print(f"Grid search completed. Best Sharpe ratio: {best_score:.4f}")

    # Return optimized parameters with full results
    return {
        "best_params": best_params,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "all_results": results,
        "total_combinations_tested": len([r for r in results if "cv_score" in r]),
    }


def bayesian_optimization(
    processed_data: pd.DataFrame,
    initial_params: dict,
    policy_type: str,
    n_calls: int = 50,
    cv_folds: int = 3,
) -> dict:
    """
    Perform Bayesian optimization of hedging strategy parameters.

    Args:
        processed_data (pd.DataFrame): Processed historical data with features.
        initial_params (dict): Initial parameters for the hedging strategy.
        policy_type (str): Type of policy being optimized.
        n_calls (int): Number of optimization calls.
        cv_folds (int): Number of cross-validation folds.

    Returns:
        dict: Optimized parameters with performance metrics.
    """
    print("Starting Bayesian optimization...")

    if BAYESIAN_AVAILABLE:
        return bayesian_optimization_skopt(
            processed_data, initial_params, policy_type, n_calls, cv_folds
        )
    else:
        return bayesian_optimization_scipy(
            processed_data, initial_params, policy_type, n_calls, cv_folds
        )


def bayesian_optimization_skopt(
    processed_data: pd.DataFrame,
    initial_params: dict,
    policy_type: str,
    n_calls: int = 50,
    cv_folds: int = 3,
) -> dict:
    """
    Perform Bayesian optimization using scikit-optimize.
    """
    if not BAYESIAN_AVAILABLE:
        print("scikit-optimize not available, falling back to scipy")
        return bayesian_optimization_scipy(
            processed_data, initial_params, policy_type, n_calls, cv_folds
        )

    # Import here to avoid errors when not available
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args

    # Define search space based on policy type
    if policy_type == "delta_neutral":
        dimensions = [
            Categorical(["daily", "weekly", "biweekly"], name="rebalance_frequency"),
            Real(0.01, 0.3, name="delta_threshold"),
            Integer(5, 100, name="volatility_window"),
            Real(0.5, 2.0, name="hedge_ratio_multiplier"),
        ]
    elif policy_type == "gamma_scaled":
        dimensions = [
            Real(0.1, 5.0, name="scaling_factor"),
            Categorical(["daily", "weekly", "biweekly"], name="rebalance_frequency"),
            Real(0.001, 0.2, name="gamma_threshold"),
            Integer(5, 100, name="volatility_window"),
            Real(0.5, 2.0, name="hedge_ratio_multiplier"),
        ]
    else:
        print(f"Unknown policy type: {policy_type}")
        return initial_params

    # Store results for analysis
    evaluation_results = []

    @use_named_args(dimensions)
    def objective(**params):
        """Objective function for Bayesian optimization."""
        try:
            # Perform cross-validation
            cv_scores = []
            data_splits = create_time_series_cv_splits(processed_data, cv_folds)

            for train_data, test_data in data_splits:
                performance = evaluate_strategy_performance(
                    train_data, test_data, params, policy_type
                )
                cv_scores.append(performance["sharpe_ratio"])

            avg_score = np.mean(cv_scores)

            # Store result (note: we minimize, so negate the score)
            evaluation_results.append(
                {"params": params.copy(), "score": avg_score, "cv_scores": cv_scores}
            )

            return -avg_score  # Minimize negative Sharpe ratio

        except Exception as e:
            print(f"Error in objective function: {e}")
            return 1000  # Large positive value for minimization

    # Run Bayesian optimization
    print(f"Running Bayesian optimization with {n_calls} calls...")
    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=n_calls,
        random_state=42,
        acq_func="EI",  # Expected Improvement
    )

    # Extract best parameters
    best_params = {}
    for i, dim in enumerate(dimensions):
        best_params[dim.name] = result.x[i]

    best_score = -result.fun  # Convert back from minimization

    print(f"Bayesian optimization completed. Best Sharpe ratio: {best_score:.4f}")

    # Calculate final metrics with best parameters
    cv_scores = []
    cv_metrics = []
    data_splits = create_time_series_cv_splits(processed_data, cv_folds)

    for train_data, test_data in data_splits:
        performance = evaluate_strategy_performance(
            train_data, test_data, best_params, policy_type
        )
        cv_scores.append(performance["sharpe_ratio"])
        cv_metrics.append(performance)

    best_metrics = aggregate_cv_metrics(cv_metrics)

    return {
        "best_params": best_params,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "optimization_result": result,
        "all_evaluations": evaluation_results,
        "total_evaluations": len(evaluation_results),
    }


def bayesian_optimization_scipy(
    processed_data: pd.DataFrame,
    initial_params: dict,
    policy_type: str,
    n_calls: int = 50,
    cv_folds: int = 3,
) -> dict:
    """
    Perform Bayesian optimization using scipy.optimize as fallback.
    """
    print("Using scipy.optimize for Bayesian-style optimization...")

    # Define parameter bounds based on policy type
    if policy_type == "delta_neutral":
        # Map categorical to numeric for scipy
        bounds = [
            (0, 2),  # rebalance_frequency (0=daily, 1=weekly, 2=biweekly)
            (0.01, 0.3),  # delta_threshold
            (5, 100),  # volatility_window
            (0.5, 2.0),  # hedge_ratio_multiplier
        ]
        param_names = [
            "rebalance_frequency",
            "delta_threshold",
            "volatility_window",
            "hedge_ratio_multiplier",
        ]
    elif policy_type == "gamma_scaled":
        bounds = [
            (0.1, 5.0),  # scaling_factor
            (0, 2),  # rebalance_frequency
            (0.001, 0.2),  # gamma_threshold
            (5, 100),  # volatility_window
            (0.5, 2.0),  # hedge_ratio_multiplier
        ]
        param_names = [
            "scaling_factor",
            "rebalance_frequency",
            "gamma_threshold",
            "volatility_window",
            "hedge_ratio_multiplier",
        ]
    else:
        return initial_params

    evaluation_results = []

    def objective(x):
        """Objective function for scipy optimization."""
        try:
            # Convert numeric parameters back to proper format
            params = {}
            for i, name in enumerate(param_names):
                if name == "rebalance_frequency":
                    freq_map = {0: "daily", 1: "weekly", 2: "biweekly"}
                    params[name] = freq_map[int(round(x[i]))]
                elif name in ["volatility_window"]:
                    params[name] = int(round(x[i]))
                else:
                    params[name] = x[i]

            # Perform cross-validation
            cv_scores = []
            data_splits = create_time_series_cv_splits(processed_data, cv_folds)

            for train_data, test_data in data_splits:
                performance = evaluate_strategy_performance(
                    train_data, test_data, params, policy_type
                )
                cv_scores.append(performance["sharpe_ratio"])

            avg_score = np.mean(cv_scores)

            evaluation_results.append(
                {"params": params.copy(), "score": avg_score, "cv_scores": cv_scores}
            )

            return -avg_score  # Minimize negative Sharpe ratio

        except Exception as e:
            print(f"Error in objective function: {e}")
            return 1000

    # Use differential evolution for global optimization
    print("Running differential evolution optimization...")
    result = optimize.differential_evolution(
        objective,
        bounds,
        maxiter=n_calls // 10,  # Adjust iterations for scipy
        tol=1e-6,
    )

    # Extract best parameters
    best_params = {}
    for i, name in enumerate(param_names):
        if name == "rebalance_frequency":
            freq_map = {0: "daily", 1: "weekly", 2: "biweekly"}
            best_params[name] = freq_map[int(round(result.x[i]))]
        elif name in ["volatility_window"]:
            best_params[name] = int(round(result.x[i]))
        else:
            best_params[name] = result.x[i]

    best_score = -result.fun

    print(f"Scipy optimization completed. Best Sharpe ratio: {best_score:.4f}")

    # Calculate final metrics
    cv_scores = []
    cv_metrics = []
    data_splits = create_time_series_cv_splits(processed_data, cv_folds)

    for train_data, test_data in data_splits:
        performance = evaluate_strategy_performance(
            train_data, test_data, best_params, policy_type
        )
        cv_scores.append(performance["sharpe_ratio"])
        cv_metrics.append(performance)

    best_metrics = aggregate_cv_metrics(cv_metrics)

    return {
        "best_params": best_params,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "optimization_result": result,
        "all_evaluations": evaluation_results,
        "total_evaluations": len(evaluation_results),
    }


def main():
    """
    Main function to parse command-line arguments and run the optimization.
    """
    parser = argparse.ArgumentParser(description="Optimize Hedging Policy Parameters")
    parser.add_argument(
        "--tickers", type=str, required=True, help="Comma-separated stock tickers"
    )
    parser.add_argument(
        "--start_date", type=str, required=True, help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end_date", type=str, required=True, help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--initial_params",
        type=str,
        required=True,
        help="Initial parameters as a dictionary string (e.g., \"{'param1': value1, 'param2': value2}\")",
    )
    parser.add_argument(
        "--optimization_method",
        type=str,
        default="grid_search",
        choices=["grid_search", "bayesian_optimization"],
        help="Optimization method to use",
    )
    parser.add_argument(
        "--policy_type",
        type=str,
        default="delta_neutral",
        choices=["delta_neutral", "gamma_scaled"],
        help="Type of hedging policy to optimize",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="File path to save optimized parameters (optional)",
    )
    parser.add_argument(
        "--n_calls",
        type=int,
        default=50,
        help="Number of calls for Bayesian optimization",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=3,
        help="Number of cross-validation folds",
    )

    args = parser.parse_args()

    # Convert initial_params string to dictionary
    try:
        initial_params = eval(args.initial_params)
        if not isinstance(initial_params, dict):
            raise ValueError
    except Exception:
        print("Invalid format for initial_params. Must be a dictionary string.")
        sys.exit(1)

    optimized_params = optimize_policy(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_params=initial_params,
        optimization_method=args.optimization_method,
        output_file=args.output_file,
        policy_type=args.policy_type,
        n_calls=args.n_calls,
        cv_folds=args.cv_folds,
    )

    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)

    if "best_params" in optimized_params:
        print(f"Best Parameters: {optimized_params['best_params']}")
        print(f"Best Score (Sharpe): {optimized_params.get('best_score', 'N/A'):.4f}")

        if "best_metrics" in optimized_params:
            metrics = optimized_params["best_metrics"]
            print("\nPerformance Metrics:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
    else:
        print("Optimization failed or returned incomplete results.")
        print(f"Available keys: {list(optimized_params.keys())}")


if __name__ == "__main__":
    main()
