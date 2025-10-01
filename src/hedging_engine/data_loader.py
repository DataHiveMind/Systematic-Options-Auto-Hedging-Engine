"""
======================================================================================
Title: Data Loader
Author: Kenneth LeGare
Date: 2023-10-05
Updated: 2025-01-01

Description: 
    This module is responsible for loading and preprocessing data for the hedging engine.
    Updated to use modern OpenBB Platform API with proper error handling and fallback providers.

Dependencies:
    - pandas
    - numpy
    - openbb
    - warnings
======================================================================================
"""

# Standard Library
import warnings
from typing import Optional

# Third Party Libraries
import numpy as np
import pandas as pd
from openbb import obb
from openbb_core.provider.utils.errors import EmptyDataError

def load_data(
    tickers: str, 
    start_date: str, 
    end_date: str, 
    provider: str = "yfinance"
) -> pd.DataFrame:
    """
    Load historical stock data from OpenBB API with fallback providers.

    Args:
        tickers (str): A comma-separated string of stock tickers.
        start_date (str): The start date for the historical data (YYYY-MM-DD).
        end_date (str): The end date for the historical data (YYYY-MM-DD).
        provider (str): The data provider to use. Defaults to "yfinance".

    Returns:
        pd.DataFrame: A DataFrame containing the historical stock data.
        
    Raises:
        EmptyDataError: If no data is found for the given parameters.
    """
    # List of providers to try in order
    providers_to_try = [provider, "yfinance", "fmp", "alpha_vantage"]
    
    # Remove duplicates while preserving order
    providers_to_try = list(dict.fromkeys(providers_to_try))
    
    for current_provider in providers_to_try:
        try:
            print(f"Attempting to load data with provider: {current_provider}")
            
            # Use the modern OpenBB API pattern with provider specification
            output = obb.equity.price.historical(
                symbol=tickers, 
                start_date=start_date, 
                end_date=end_date,
                provider=current_provider
            )
            
            # Convert to DataFrame using the correct method
            df = output.to_df()
            
            if df is not None and not df.empty:
                print(f"Successfully loaded data with provider: {current_provider}")
                return df
            else:
                print(f"No data returned from provider: {current_provider}")
                continue
                
        except EmptyDataError as e:
            print(f"EmptyDataError with provider {current_provider}: {e}")
            continue
        except Exception as e:
            print(f"Error with provider {current_provider}: {type(e).__name__}: {e}")
            continue
    
    # If all providers fail, raise an error
    raise EmptyDataError(
        f"No data found for ticker(s) {tickers} between {start_date} and {end_date} "
        f"with any of the tried providers: {providers_to_try}"
    )

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
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
    df = df[~df.index.duplicated(keep='first')]
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

def save_raw_data(df: pd.DataFrame, filename: str) -> None:
    """
    Save the raw DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The name of the file to save the data to.
    """
    if df is None or df.empty:
        print("No data to save.")
        return

    df.to_csv(filename)
    print(f"Raw data saved to {filename}")

def save_interim_data(df: pd.DataFrame, filename: str) -> None:
    """
    Save the interim DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The name of the file to save the data to.
    """
    if df is None or df.empty:
        print("No data to save.")
        return

    df.to_csv(filename)
    print(f"Interim data saved to {filename}")

def save_processed_data(df : pd.DataFrame, filename : str) -> None:
    """
    Save the processed DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The name of the file to save the data to.
    """
    if df is None or df.empty:
        print("No data to save.")
        return

    df.to_csv(filename)
    print(f"Processed data saved to {filename}")


def main() -> None:
    """
    Main function to demonstrate data loading and preprocessing.
    """
    tickers = "GOOG"
    start_date = "2020-01-01"
    end_date = "2023-01-01"

    try:
        print(f"Loading data for {tickers} from {start_date} to {end_date}")
        df = load_data(tickers, start_date, end_date)
        
        print("\nRaw data info:")
        print(df.head(5))
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        print("\nPreprocessing data...")
        df = preprocess_data(df)
        
        print("\nFinal data sample:")
        print(df.head())
        print("\nData loading completed successfully!")
        
    except Exception as e:
        print(f"Error in main: {type(e).__name__}: {e}")
        raise

if __name__ == "__main__":  
    main()
