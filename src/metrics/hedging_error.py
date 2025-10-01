"""
======================================================================
Title: hedging_error.py
Author: Kenneth LeGare
Date: 2024-06-15

Description:
    This module defines custom strategies for hedge deviations

Dependencies:
    - pandas
    - numpy
    - scipy
    - statsmodels
======================================================================
"""

# Standard Library
import warnings

# Third Party Libraries
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS


def calculate_hedging_error(returns: pd.Series, hedge_ratios: pd.Series) -> pd.Series:
    """
    Calculate the hedging error given asset returns and hedge ratios.

    Args:
        returns (pd.Series): Series of asset returns.
        hedge_ratios (pd.Series): Series of hedge ratios.

    Returns:
        pd.Series: Series of hedging errors.
    """
    if len(returns) != len(hedge_ratios):
        raise ValueError("Returns and hedge_ratios must be of the same length.")

    hedging_error = returns - hedge_ratios.shift(1) * returns
    return hedging_error.dropna()


def hedging_error_statistics(hedging_errors: pd.Series) -> dict:
    """
    Compute statistics for the hedging errors.

    Args:
        hedging_errors (pd.Series): Series of hedging errors.

    Returns:
        dict: Dictionary of hedging error statistics.
    """
    stats = {
        "mean": hedging_errors.mean(),
        "median": hedging_errors.median(),
        "std_dev": hedging_errors.std(),
        "min": hedging_errors.min(),
        "max": hedging_errors.max(),
    }
    return stats


def regression_hedging_error(returns: pd.Series, hedge_ratios: pd.Series) -> OLS:
    """
    Perform regression analysis to evaluate hedging effectiveness.

    Args:
        returns (pd.Series): Series of asset returns.
        hedge_ratios (pd.Series): Series of hedge ratios.

    Returns:
        OLS: Fitted OLS regression model.
    """
    if len(returns) != len(hedge_ratios):
        raise ValueError("Returns and hedge_ratios must be of the same length.")

    X = sm.add_constant(hedge_ratios.shift(1).dropna())
    y = returns.loc[X.index]

    model = OLS(y, X).fit()
    return model


def hedging_error_distribution(hedging_errors: pd.Series) -> stats.rv_continuous:
    """
    Fit a normal distribution to the hedging errors.

    Args:
        hedging_errors (pd.Series): Series of hedging errors.

    Returns:
        stats.rv_continuous: Fitted normal distribution.
    """
    mu, std = stats.norm.fit(hedging_errors)
    return stats.norm(loc=mu, scale=std)


def value_at_risk(hedging_errors: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate the Value at Risk (VaR) for the hedging errors.

    Args:
        hedging_errors (pd.Series): Series of hedging errors.
        confidence_level (float): Confidence level for VaR calculation.

    Returns:
        float: Value at Risk.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    var = np.percentile(hedging_errors, (1 - confidence_level) * 100)
    return var


def conditional_value_at_risk(
    hedging_errors: pd.Series, confidence_level: float = 0.95
) -> float:
    """
    Calculate the Conditional Value at Risk (CVaR) for the hedging errors.

    Args:
        hedging_errors (pd.Series): Series of hedging errors.
        confidence_level (float): Confidence level for CVaR calculation.

    Returns:
        float: Conditional Value at Risk.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    var = value_at_risk(hedging_errors, confidence_level)
    cvar = hedging_errors[hedging_errors <= var].mean()
    return cvar


def plot_hedging_errors(
    hedging_errors: pd.Series,
    save_path: str | None = None,
    report_name: str = "hedging_error",
):
    """
    Plot the distribution of hedging errors.

    Args:
        hedging_errors (pd.Series): Series of hedging errors.
        save_path (str, optional): Path to save the plot. If None, will create organized folder structure.
        report_name (str): Name of the report/analysis for folder organization.
    """
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Ensure we're using a backend that can display plots
    try:
        # Try to use interactive backend for environments that support it
        import matplotlib

        current_backend = matplotlib.get_backend()
        if current_backend == "Agg" and save_path is None:
            # Switch to a display backend if available
            try:
                matplotlib.use("TkAgg")
            except ImportError:
                try:
                    matplotlib.use("Qt5Agg")
                except ImportError:
                    # If no display backend available, save to file instead
                    save_path = "hedging_error_distribution.png"
                    print(f"No display backend available. Saving plot to {save_path}")
    except Exception:
        pass

    plt.figure(figsize=(10, 6))
    sns.histplot(hedging_errors.to_numpy(), bins=30, kde=True)
    plt.title("Hedging Error Distribution")
    plt.xlabel("Hedging Error")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)

    # Force display of the plot
    plt.tight_layout()

    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True,
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        # Create organized reports folder structure
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_dir = os.path.join("reports", report_name, timestamp)
        os.makedirs(reports_dir, exist_ok=True)

        chart_path = os.path.join(reports_dir, "hedging_error_distribution.png")
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        print(f"Chart saved to: {chart_path}")

    plt.show()

    # Keep the plot window open
    plt.pause(0.1)


def save_report(content: str, filename: str, report_format: str = "pdf") -> None:
    """
    Save the report content to a file in the specified format.

    Args:
        content (str): The content of the report.
        filename (str): The name of the file to save the report to.
        report_format (str): The format of the report ('pdf' or 'html').
    """
    try:
        if report_format == "pdf":
            from fpdf import FPDF

            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Arial", size=12)

            for line in content.split("\n"):
                pdf.cell(0, 10, line, ln=True)
            pdf.output(filename)

        elif report_format == "html":
            with open(filename, "w") as f:
                f.write(f"<html><body><pre>{content}</pre></body></html>")

        else:
            raise ValueError("Unsupported report format. Use 'pdf' or 'html'.")

        print(f"Report saved to {filename}")

    except Exception as e:
        print(f"Failed to save report: {type(e).__name__}: {e}")


def main():
    """
    Main function to demonstrate hedging error calculations.
    """
    # Example data
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.01, 1000))
    hedge_ratios = pd.Series(np.random.uniform(0, 1, 1000))

    # Calculate hedging errors
    hedging_errors = calculate_hedging_error(returns, hedge_ratios)

    # Compute statistics
    stats = hedging_error_statistics(hedging_errors)
    print("Hedging Error Statistics:", stats)

    # Regression analysis
    results = regression_hedging_error(returns, hedge_ratios)
    print(results.summary())

    # Distribution fitting
    distribution = hedging_error_distribution(hedging_errors)
    print("Fitted Distribution Mean:", distribution.mean())
    print("Fitted Distribution Std Dev:", distribution.std())

    # Risk measures
    var = value_at_risk(hedging_errors)
    cvar = conditional_value_at_risk(hedging_errors)
    print(f"Value at Risk (95%): {var}")
    print(f"Conditional Value at Risk (95%): {cvar}")

    # Plotting - save to organized reports folder
    plot_hedging_errors(hedging_errors, report_name="hedging_error_analysis")


if __name__ == "__main__":
    main()
