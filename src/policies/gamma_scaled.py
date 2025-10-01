"""
====================================================================================
Title: Gamma Scaled Hedging Policy
Author: Kenneth LeGare
Date: 2023-10-01

Description: 
    This policy outlines the approach for maintaining a gamma scaled position in options trading by 
    dynamically adjusting the hedge ratio based on market conditions.

dependencies:
    - numpy
    - pandas
    - scipy
    - scikit-learn
    - statsmodels
"""

import numpy as np, pandas as pd

import scipy.stats as stats
import scipy.sparse as sp
import scipy.sparse.linalg as la

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as lm

import statsmodels.stats as st

def calculate_gamma(option_data: pd.DataFrame) -> float:
    """
    Calculate the gamma for a given options portfolio.

    args:
        option_data (pd.DataFrame): A DataFrame containing option data with columns for option_price,
                                      strike_price, time_to_expiration, and implied_volatility.

    returns:
        float: The calculated gamma.
    """
    # Fit a linear regression model to estimate the gamma
    X = option_data[['implied_volatility', 'time_to_expiration']]
    y = option_data['gamma']
    model = lm()
    model.fit(X, y)

    # The gamma is the model's coefficient for implied volatility
    gamma = model.coef_[0]
    return gamma

def calculate_beta(option_data: pd.DataFrame) -> float:
    """
    Calculate the beta for a given options portfolio.

    args:
        option_data (pd.DataFrame): A DataFrame containing option data with columns for option_price,
                                      strike_price, time_to_expiration, and implied_volatility.

    returns:
        float: The calculated beta.
    """
    # Fit a linear regression model to estimate the beta
    X = option_data[['implied_volatility', 'time_to_expiration']]
    y = option_data['beta']
    model = lm()
    model.fit(X, y)

    # The beta is the model's coefficient for implied volatility
    beta = model.coef_[0]
    return beta

def calculate_gamma_score(option_data: pd.DataFrame) -> float:
    """
    Calculate the gamma score for a given options portfolio.

    args:
        option_data (pd.DataFrame): A DataFrame containing option data with columns for option_price,
                                      strike_price, time_to_expiration, and implied_volatility.

    returns:
        float: The calculated gamma score.
    """
    gamma = calculate_gamma(option_data)
    beta = calculate_beta(option_data)
    if beta != 0:
        gamma_score = gamma / beta
    else:
        gamma_score = 0
    return gamma_score

def main():
    # create sample data
    option_data = pd.DataFrame({
        'option_price': [10, 12, 14],
        'strike_price': [11, 13, 15],
        'time_to_expiration': [30, 60, 90],
        'implied_volatility': [0.2, 0.25, 0.3],
        'gamma': [0.1, 0.15, 0.2],
        "beta": [0.05, 0.1, 0.15]
    })

    # Calculate gamma score
    gamma_score = calculate_gamma_score(option_data)

    # Print results
    print(f"Gamma Score: {gamma_score}")

if __name__ == "__main__":
    main()
