"""
=========================================================================
Title: Backtest.py
Author: Kenneth LeGare
Date: 2024-06-25
Description:
    This module provides functions for backtesting hedging strategies
    using the hedging engine and metrics modules.

Dependencies:
    - pandas
    - numpy
    - hedging_engine
    - metrics
    - logging
=========================================================================
"""

# Standard Library
import logging
import os
import sys
from typing import Optional, Dict, Any

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
src_path = os.path.join(project_root, "src")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Third Party Libraries
import numpy as np
import pandas as pd

# Local Modules - import after path setup
try:
    from hedging_engine.data_loader import load_data, preprocess_data
    from metrics.hedging_error import calculate_hedging_error, hedging_error_statistics
    from metrics.pnl_attribution import attribute_pnl

    print("✓ Successfully imported all local modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Available paths:")
    for path in sys.path[:3]:
        print(f"  {path}")
    raise

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def backtest_hedging_strategy(
    price_data: pd.DataFrame,
    strategy_params: Dict[str, Any],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Backtest a hedging strategy over a specified date range.

    Args:
        price_data (pd.DataFrame): DataFrame containing historical price data.
        strategy_params (Dict[str, Any]): Parameters for the hedging strategy.
        start_date (Optional[str]): Start date for backtesting (YYYY-MM-DD).
        end_date (Optional[str]): End date for backtesting (YYYY-MM-DD).

    Returns:
        Dict[str, Any]: Dictionary containing backtest results and metrics.
    """
    logger.info(f"Price data shape before filtering: {price_data.shape}")
    logger.info(f"Price data index type: {type(price_data.index)}")
    logger.info(f"First few index values: {price_data.index[:3].tolist()}")

    # Ensure the index is datetime
    if not isinstance(price_data.index, pd.DatetimeIndex):
        logger.info("Converting index to DatetimeIndex")
        price_data.index = pd.to_datetime(price_data.index)

    # Filter data by date range
    if start_date:
        start_timestamp = pd.to_datetime(start_date)
        logger.info(f"Filtering from start_date: {start_timestamp}")
        price_data = price_data[price_data.index >= start_timestamp]
    if end_date:
        end_timestamp = pd.to_datetime(end_date)
        logger.info(f"Filtering to end_date: {end_timestamp}")
        price_data = price_data[price_data.index <= end_timestamp]

    logger.info(f"Price data shape after filtering: {price_data.shape}")

    if price_data.empty:
        logger.warning("No data available for the specified date range.")
        return {}

    # Initialize hedging strategy (placeholder implementation)
    # TODO: Implement proper HedgingEngine class
    print("Warning: HedgingEngine not implemented yet. Using placeholder.")

    # For now, use a simple placeholder that returns dummy hedge ratios
    # Align hedge_ratios with returns (which drops first NaN from pct_change)
    hedge_ratios = pd.Series(
        index=price_data.index[1:], data=0.5
    )  # 50% hedge ratio, aligned with returns
    logger.info(f"Hedge ratios shape: {hedge_ratios.shape}")

    # Calculate returns
    logger.info(f"Available columns: {price_data.columns.tolist()}")

    # Use the 'close' column (lowercase) based on OpenBB data format
    if "close" in price_data.columns:
        close_col = "close"
    elif "Close" in price_data.columns:
        close_col = "Close"
    else:
        # Fallback to first numeric column
        close_col = price_data.select_dtypes(include=[np.number]).columns[0]
        logger.warning(f"No 'close' column found. Using '{close_col}' instead.")

    returns = price_data[close_col].pct_change().dropna()
    logger.info(f"Calculated returns shape: {returns.shape}")

    # Calculate hedging errors
    hedging_errors = calculate_hedging_error(returns, hedge_ratios)

    # Compute hedging error statistics
    error_stats = hedging_error_statistics(hedging_errors)

    # Attribute PnL - ensure proper index alignment
    # Create aligned series for PnL calculation
    aligned_hedge_ratios = hedge_ratios.shift(1).dropna()
    aligned_returns = returns[aligned_hedge_ratios.index]

    pnl_series = aligned_returns * aligned_hedge_ratios
    factors_df = pd.DataFrame(
        {"Market": aligned_returns, "HedgeRatio": aligned_hedge_ratios},
        index=aligned_returns.index,
    )

    logger.info(f"PnL series shape: {pnl_series.shape}")
    logger.info(f"Factors DataFrame shape: {factors_df.shape}")

    pnl_contributions, residuals = attribute_pnl(pnl_series, factors_df)

    # Compile results
    results = {
        "hedge_ratios": hedge_ratios,
        "hedging_errors": hedging_errors,
        "error_statistics": error_stats,
        "pnl_contributions": pnl_contributions,
        "residuals": residuals,
    }

    return results


def monte_carlo_backtest(
    price_data: pd.DataFrame,
    strategy_params: Dict[str, Any],
    num_simulations: int = 100,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Perform Monte Carlo backtesting of a hedging strategy.

    Args:
        price_data (pd.DataFrame): DataFrame containing historical price data.
        strategy_params (Dict[str, Any]): Parameters for the hedging strategy.
        num_simulations (int): Number of Monte Carlo simulations to run.
        start_date (Optional[str]): Start date for backtesting (YYYY-MM-DD).
        end_date (Optional[str]): End date for backtesting (YYYY-MM-DD).

    Returns:
        pd.DataFrame: DataFrame containing summary statistics from simulations.
    """
    simulation_results = []

    for i in range(num_simulations):
        logger.info(f"Running simulation {i + 1}/{num_simulations}")

        # Randomly sample with replacement to create a bootstrap sample
        sampled_data = price_data.sample(frac=1, replace=True).reset_index(drop=True)

        # Restore datetime index to avoid issues with date filtering
        sampled_data.index = pd.date_range(
            start=price_data.index.min(), periods=len(sampled_data), freq="D"
        )

        # Backtest the strategy on the sampled data
        results = backtest_hedging_strategy(
            sampled_data, strategy_params, start_date, end_date
        )

        if results:
            simulation_results.append(results["error_statistics"])

    if not simulation_results:
        logger.warning("No valid simulation results were generated.")
        return pd.DataFrame()

    # Convert list of dicts to DataFrame
    results_df = pd.DataFrame(simulation_results)

    # Compute summary statistics across simulations
    summary_stats = results_df.agg(["mean", "std", "min", "max"])

    return summary_stats


def move_forwarding_window_backtest(
    price_data: pd.DataFrame,
    strategy_params: Dict[str, Any],
    window_size: int,
    step_size: int,
) -> pd.DataFrame:
    """
    Perform backtesting using a moving forward window approach.

    Args:
        price_data (pd.DataFrame): DataFrame containing historical price data.
        strategy_params (Dict[str, Any]): Parameters for the hedging strategy.
        window_size (int): Size of the training window in days.
        step_size (int): Step size to move the window in days.

    Returns:
        pd.DataFrame: DataFrame containing summary statistics from each window.
    """
    results = []
    start_idx = 0
    end_idx = window_size

    while end_idx <= len(price_data):
        window_data = price_data.iloc[start_idx:end_idx]
        logger.info(
            f"Backtesting window: {window_data.index.min()} to {window_data.index.max()}"
        )

        backtest_results = backtest_hedging_strategy(window_data, strategy_params)

        if backtest_results:
            error_stats = backtest_results["error_statistics"]
            error_stats["start_date"] = window_data.index.min()
            error_stats["end_date"] = window_data.index.max()
            results.append(error_stats)

        start_idx += step_size
        end_idx += step_size

    if not results:
        logger.warning("No valid backtest results were generated.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    return results_df


def summarize_backtest_results(results: Dict[str, Any]) -> None:
    """
    Print a summary of the backtest results.

    Args:
        results (Dict[str, Any]): Dictionary containing backtest results and metrics.
    """
    if not results:
        logger.info("No results to summarize.")
        return

    logger.info("Hedging Error Statistics:")
    for stat, value in results["error_statistics"].items():
        logger.info(f"  {stat}: {value:.6f}")

    logger.info("\nPnL Contributions:")
    logger.info(results["pnl_contributions"].head())

    logger.info("\nResiduals:")
    logger.info(results["residuals"].head())


def main():
    """
    Main function to demonstrate backtesting functionalities.
    """
    # Example data loading
    tickers = "AAPL"
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    price_data = load_data(tickers, start_date, end_date)

    if price_data.empty:
        logger.error("Failed to load price data. Exiting.")
        return

    # Define strategy parameters
    strategy_params = {"hedge_ratio_method": "delta", "lookback_period": 20}

    # Backtest the hedging strategy
    results = backtest_hedging_strategy(
        price_data, strategy_params, start_date, end_date
    )
    summarize_backtest_results(results)

    # Monte Carlo backtesting
    monte_carlo_summary = monte_carlo_backtest(
        price_data,
        strategy_params,
        num_simulations=50,
        start_date=start_date,
        end_date=end_date,
    )
    logger.info("\nMonte Carlo Backtest Summary:")
    logger.info(monte_carlo_summary)

    # Moving forward window backtesting
    window_size = 60  # days
    step_size = 30  # days
    moving_window_results = move_forwarding_window_backtest(
        price_data, strategy_params, window_size, step_size
    )
    logger.info("\nMoving Forward Window Backtest Results:")
    logger.info(moving_window_results)


if __name__ == "__main__":
    main()
