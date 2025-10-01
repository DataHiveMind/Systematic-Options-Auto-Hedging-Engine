"""
======================================================================================
Title: Data Loader
Author: Kenneth LeGare
Date: 2023-10-05

Description: 
    This module is responsible for loading and preprocessing data for the hedging engine.

Dependencies:
    - pandas
    - numpy
    - openBB
    - yfinance
    - scikit-learn
======================================================================================
"""

# Third Party Libraries
import numpy as np, pandas as pd, yfinance as yf
from sympy import true
from openbb import obb

def load_data(tickers: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load historical stock data from OpenBB API.

    args:
        tickers (str): A comma-separated string of stock tickers.
        start_date (str): The start date for the historical data (YYYY-MM-DD).
        end_date (str): The end date for the historical data (YYYY-MM-DD).

    returns:
        pd.DataFrame: A DataFrame containing the historical stock data.
    """
    data = {}
    for ticker in tickers:
        df = obb.get("stocks", "historical", ticker, start=start_date, end=end_date)
        data[ticker] = df
    return pd.concat(data.values(), axis=0)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the historical stock data for duplicate entries, NaN values, and holes in data.

    args:
        df (pd.DataFrame): The DataFrame containing the historical stock data.

    returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    # Handle Duplicates.
    df = df[~df.index.duplicated(keep='first')]

    # Handle NaN Values.
    df = df.dropna(inplace=true)

    # Handle holes in data
    df.ffill().bfill()

    return df

def main()-> None:
    tickers = "AAPL"
    start_date = "2020-01-01"
    end_date = "2023-01-01"

    df = load_data(tickers, start_date, end_date)
    df = preprocess_data(df)

if __name__ == "__main__":  
    main()
