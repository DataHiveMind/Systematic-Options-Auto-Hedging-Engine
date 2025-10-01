"""
====================================================================================
Title: PnL Attribution Module
Author: Kenneth LeGare
Date: 2024-06-20

Description:
    This module provides functions to attribute profit and loss (PnL) to various factors
    such as market movements, hedging strategies, and transaction costs.

Dependencies:
    - pandas
    - numpy
====================================================================================
"""

# Standard Library
import warnings
from typing import Optional, Tuple
import logging
import time
from datetime import datetime

# Third Party Libraries
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from sklearn.linear_model import LinearRegression


def attribute_pnl(
    pnl_series: pd.Series, factors_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Attribute PnL to various factors using linear regression.

    Args:
        pnl_series (pd.Series): Series of profit and loss values.
        factors_df (pd.DataFrame): DataFrame of factors to attribute PnL to.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: DataFrame of factor contributions and Series of residuals.
    """
    if len(pnl_series) != len(factors_df):
        raise ValueError("PnL series and factors DataFrame must be of the same length.")

    # Align indices
    pnl_series = pnl_series.loc[factors_df.index]

    # Add constant for intercept
    X = sm.add_constant(factors_df)
    y = pnl_series

    model = OLS(y, X).fit()
    contributions = model.params * X.T
    contributions_df = pd.DataFrame(contributions.T, index=factors_df.index)

    residuals = model.resid

    return contributions_df, residuals


def pnl_statistics(pnl_series: pd.Series) -> dict:
    """
    Compute statistics for the PnL series.

    Args:
        pnl_series (pd.Series): Series of profit and loss values.

    Returns:
        dict: Dictionary of PnL statistics.
    """
    pnl_stats = {
        "mean": pnl_series.mean(),
        "median": pnl_series.median(),
        "std_dev": pnl_series.std(),
        "min": pnl_series.min(),
        "max": pnl_series.max(),
        "skewness": stats.skew(pnl_series),
        "kurtosis": stats.kurtosis(pnl_series),
    }
    return pnl_stats


def rolling_pnl_statistics(pnl_series: pd.Series, window: int) -> pd.DataFrame:
    """
    Compute rolling statistics for the PnL series.

    Args:
        pnl_series (pd.Series): Series of profit and loss values.
        window (int): Rolling window size.

    Returns:
        pd.DataFrame: DataFrame of rolling PnL statistics.
    """
    rolling_stats = pd.DataFrame(
        {
            "rolling_mean": pnl_series.rolling(window).mean(),
            "rolling_std_dev": pnl_series.rolling(window).std(),
            "rolling_min": pnl_series.rolling(window).min(),
            "rolling_max": pnl_series.rolling(window).max(),
        }
    )
    return rolling_stats


def preprocess_data(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Clean and preprocess the historical stock data for duplicate entries, NaN values, and holes in data.

    Args:
        df (pd.DataFrame): The DataFrame containing the historical stock data.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    if df is None or df.empty:
        warnings.warn("Input DataFrame is empty or None")
        return df

    print(f"Input data shape: {df.shape}")

    # Handle Duplicates - remove duplicate index entries
    initial_len = len(df)
    df = df[~df.index.duplicated(keep="first")]
    duplicates_removed = initial_len - len(df)
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate entries")

    # Handle NaN Values - use correct Python boolean and fix inplace usage
    initial_len = len(df)
    df = df.dropna()  # Remove inplace=True and assign result
    nan_removed = initial_len - len(df)
    if nan_removed > 0:
        print(f"Removed {nan_removed} rows with NaN values")

    # Handle holes in data - forward fill then backward fill
    df = df.ffill().bfill()  # Assign result back to df

    print(f"Final preprocessed data shape: {df.shape}")

    return df


def regression_pnl(pnl_series: pd.Series, factors_df: pd.DataFrame) -> OLS:
    """
    Perform regression analysis to evaluate PnL attribution effectiveness.

    Args:
        pnl_series (pd.Series): Series of profit and loss values.
        factors_df (pd.DataFrame): DataFrame of factors to attribute PnL to.

    Returns:
        OLS: Fitted OLS regression model.
    """
    if len(pnl_series) != len(factors_df):
        raise ValueError("PnL series and factors DataFrame must be of the same length.")

    # Align indices
    pnl_series = pnl_series.loc[factors_df.index]

    # Add constant for intercept
    X = sm.add_constant(factors_df)
    y = pnl_series

    model = OLS(y, X).fit()
    return model


def save_report(
    content: str,
    ticker: str,
    report_format: str = "pdf",
    report_name: str = "pnl_attribution",
) -> None:
    """
    Save the report content to a file in the specified format with organized folder structure.

    Args:
        content (str): The content of the report.
        ticker (str): The stock ticker symbol.
        report_format (str): The format of the report ("pdf" or "html"). Defaults to "pdf".
        report_name (str): The name of the report type for folder organization.
    """
    import os

    # Create organized folder structure: reports/report_name/timestamp/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = os.path.join("reports", report_name, timestamp)
    os.makedirs(reports_dir, exist_ok=True)

    filename = f"{ticker}_report.{report_format}"
    filepath = os.path.join(reports_dir, filename)

    try:
        if report_format == "pdf":
            from fpdf import FPDF

            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Arial", size=12)
            for line in content.split("\n"):
                pdf.cell(0, 10, line, ln=True)
            pdf.output(filepath)
        elif report_format == "html":
            with open(filepath, "w") as f:
                f.write(f"<html><body><pre>{content}</pre></body></html>")
        else:
            raise ValueError("Unsupported report format. Use 'pdf' or 'html'.")

        print(f"Report saved to: {filepath}")
    except Exception as e:
        print(f"Failed to save report: {type(e).__name__}: {e}")
        # Fallback: save to current directory as HTML
        fallback_path = f"{ticker}_report_fallback.html"
        with open(fallback_path, "w") as f:
            f.write(f"<html><body><pre>{content}</pre></body></html>")
        print(f"Report saved to fallback location: {fallback_path}")


def main():
    """
    Main function to demonstrate PnL attribution calculations.
    """
    # Example data
    np.random.seed(42)
    pnl = pd.Series(np.random.normal(0, 1, 1000))
    factors = pd.DataFrame(
        {
            "factor1": np.random.normal(0, 1, 1000),
            "factor2": np.random.normal(0, 1, 1000),
            "factor3": np.random.normal(0, 1, 1000),
        }
    )

    # Attribute PnL
    contributions, residuals = attribute_pnl(pnl, factors)
    print("Factor Contributions:\n", contributions.head())
    print("Residuals:\n", residuals.head())

    # Compute statistics
    stats = pnl_statistics(pnl)
    print("PnL Statistics:", stats)

    # Rolling statistics
    rolling_stats = rolling_pnl_statistics(pnl, window=30)
    print("Rolling PnL Statistics:\n", rolling_stats.head())

    # Regression analysis
    model = regression_pnl(pnl, factors)
    print(model.summary())

    # Save report with organized folder structure
    report_content = model.summary().as_text()
    save_report(
        report_content,
        ticker="TEST",
        report_format="html",
        report_name="pnl_attribution_analysis",
    )


if __name__ == "__main__":
    main()
