"""
======================================================================================================
Title: Cost Models Module
Author: Kenneth LeGare
Date: 2024-06-10

Description:
    This module defines various cost models for transaction cost analysis in hedging strategies.

Dependencies:
    - pandas
    - numpy
    - scipy
    - statsmodels
    - sklearn
======================================================================================================
"""

# Standard Library
import warnings
from typing import Optional, Tuple
import logging  
import time
from datetime import datetime
from functools import lru_cache

# Third Party Libraries
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CostModel:
    """
    Base class for cost models.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series):
        raise NotImplementedError("Fit method not implemented.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("Predict method not implemented.")
    
class LinearCostModel(CostModel):
    """
    Linear regression cost model.
    """

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        logger.info("LinearCostModel fitted successfully.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        predictions = self.model.predict(X)
        return pd.Series(predictions, index=X.index)

class RandomForestCostModel(CostModel):
    """
    Random forest regression cost model.
    """

    def __init__(self, n_estimators: int = 100, random_state: Optional[int] = None):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        logger.info("RandomForestCostModel fitted successfully.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        predictions = self.model.predict(X)
        return pd.Series(predictions, index=X.index)
    
def main() -> None:
    """
    Main function to demonstrate cost model fitting and prediction.
    """
    # Example data
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    X = pd.DataFrame({
        "feature1": np.random.rand(100),
        "feature2": np.random.rand(100)
    }, index=dates)
    y = pd.Series(0.5 * X["feature1"] + 0.3 * X["feature2"] + np.random.normal(0, 0.05, 100), index=dates)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit and evaluate LinearCostModel
    linear_model = LinearCostModel()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    logger.info(f"LinearCostModel MSE: {mean_squared_error(y_test, y_pred_linear)}")
    logger.info(f"LinearCostModel R2: {r2_score(y_test, y_pred_linear)}")

    # Fit and evaluate RandomForestCostModel
    rf_model = RandomForestCostModel(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    logger.info(f"RandomForestCostModel MSE: {mean_squared_error(y_test, y_pred_rf)}")
    logger.info(f"RandomForestCostModel R2: {r2_score(y_test, y_pred_rf)}")

if __name__ == "__main__":
    main()