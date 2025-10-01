#!/usr/bin/env python3
"""
Example usage of the optimize_policy.py module.

This script demonstrates how to use the optimization strategies for hedging policies.
"""

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def example_delta_neutral_optimization():
    """Example of optimizing delta neutral strategy parameters."""
    print("Example: Delta Neutral Strategy Optimization")
    print("=" * 50)

    # Example parameters
    tickers = "AAPL,GOOGL,MSFT"
    start_date = "2023-01-01"
    end_date = "2023-12-31"

    # Initial parameters for delta neutral strategy
    initial_params = {
        "rebalance_frequency": "daily",
        "hedge_ratio_multiplier": 1.0,
        "volatility_window": 20,
    }

    print(f"Tickers: {tickers}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Initial parameters: {initial_params}")
    print("Policy type: delta_neutral")
    print("Optimization method: grid_search")

    # Command to run optimization
    cmd = f"""python src/cli/optimize_policy.py \\
    --tickers "{tickers}" \\
    --start_date "{start_date}" \\
    --end_date "{end_date}" \\
    --initial_params "{initial_params}" \\
    --policy_type "delta_neutral" \\
    --optimization_method "grid_search" \\
    --cv_folds 3 \\
    --output_file "optimized_delta_neutral.json"
    """

    print("\nCommand to run:")
    print(cmd)


def example_gamma_scaled_optimization():
    """Example of optimizing gamma scaled strategy parameters."""
    print("\nExample: Gamma Scaled Strategy Optimization")
    print("=" * 50)

    # Example parameters
    tickers = "AAPL,TSLA"
    start_date = "2023-01-01"
    end_date = "2023-12-31"

    # Initial parameters for gamma scaled strategy
    initial_params = {
        "scaling_factor": 1.5,
        "rebalance_frequency": "daily",
        "gamma_threshold": 0.02,
        "hedge_ratio_multiplier": 1.0,
        "volatility_window": 20,
    }

    print(f"Tickers: {tickers}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Initial parameters: {initial_params}")
    print("Policy type: gamma_scaled")
    print("Optimization method: bayesian_optimization")

    # Command to run optimization
    cmd = f"""python src/cli/optimize_policy.py \\
    --tickers "{tickers}" \\
    --start_date "{start_date}" \\
    --end_date "{end_date}" \\
    --initial_params "{initial_params}" \\
    --policy_type "gamma_scaled" \\
    --optimization_method "bayesian_optimization" \\
    --n_calls 50 \\
    --cv_folds 3 \\
    --output_file "optimized_gamma_scaled.json"
    """

    print("\nCommand to run:")
    print(cmd)


def show_available_parameters():
    """Show available parameters for each policy type."""
    print("\nAvailable Parameters by Policy Type")
    print("=" * 50)

    print("\nDelta Neutral Policy Parameters:")
    print("  - rebalance_frequency: 'daily', 'weekly', 'biweekly'")
    print("  - delta_threshold: 0.01 to 0.3 (sensitivity threshold)")
    print("  - volatility_window: 5 to 100 (rolling window for vol calculation)")
    print("  - hedge_ratio_multiplier: 0.5 to 2.0 (hedge ratio scaling)")

    print("\nGamma Scaled Policy Parameters:")
    print("  - scaling_factor: 0.1 to 5.0 (gamma scaling factor)")
    print("  - rebalance_frequency: 'daily', 'weekly', 'biweekly'")
    print("  - gamma_threshold: 0.001 to 0.2 (gamma threshold for scaling)")
    print("  - volatility_window: 5 to 100 (rolling window for vol calculation)")
    print("  - hedge_ratio_multiplier: 0.5 to 2.0 (hedge ratio scaling)")


def main():
    """Main function to display examples."""
    parser = argparse.ArgumentParser(description="Examples for optimize_policy.py")
    parser.add_argument(
        "--example",
        choices=["delta_neutral", "gamma_scaled", "all"],
        default="all",
        help="Which example to show",
    )

    args = parser.parse_args()

    print("Hedging Policy Optimization Examples")
    print("=" * 60)

    if args.example in ["delta_neutral", "all"]:
        example_delta_neutral_optimization()

    if args.example in ["gamma_scaled", "all"]:
        example_gamma_scaled_optimization()

    if args.example == "all":
        show_available_parameters()

    print("\n" + "=" * 60)
    print("NOTES:")
    print("1. Make sure you have historical data available for the specified tickers")
    print("2. Grid search is faster but less thorough than Bayesian optimization")
    print("3. Bayesian optimization is more thorough but takes longer")
    print("4. Results are saved to JSON files for further analysis")
    print("5. Use more CV folds (5-10) for better validation in production")
    print("\nFor help with command line options:")
    print("python src/cli/optimize_policy.py --help")


if __name__ == "__main__":
    main()
