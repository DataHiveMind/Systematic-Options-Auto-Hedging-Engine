"""
============================================================================================
Title: Simulated Limit Order Book (LOB) Module
Author: Kenneth LeGare
Date: 2024-06-18

Description:
    This module provides a simulated limit order book (LOB) for testing and evaluating
    hedging strategies in a controlled environment.

Dependencies:
    - pandas
    - numpy
    - scipy
    - statsmodels
    - sklearn
============================================================================================
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimulatedLOB:
    """
    Simulated Limit Order Book (LOB) for testing hedging strategies.
    """

    def __init__(self, initial_price: float, spread: float = 0.01):
        self.initial_price = initial_price
        self.spread = spread
        self.order_book = pd.DataFrame(columns=["bid_price", "ask_price", "bid_size", "ask_size"])
        self.current_time = datetime.now()
        self._initialize_order_book()

    def _initialize_order_book(self):
        """
        Initialize the order book with initial bid and ask prices.
        """
        bid_price = self.initial_price - self.spread / 2
        ask_price = self.initial_price + self.spread / 2
        bid_size = 1000  # Initial bid size
        ask_size = 1000  # Initial ask size
        new_row = pd.DataFrame([{
            "bid_price": bid_price,
            "ask_price": ask_price,
            "bid_size": bid_size,
            "ask_size": ask_size,
        }])
        self.order_book = pd.concat([self.order_book, new_row], ignore_index=True)
        logger.info("Initialized order book.")

    def update_order_book(self, new_price: float):
        """
        Update the order book with a new mid price.

        Args:
            new_price (float): New mid price to update the order book.
        """
        bid_price = new_price - self.spread / 2
        ask_price = new_price + self.spread / 2
        bid_size = np.random.randint(500, 1500)  # Randomize bid size
        ask_size = np.random.randint(500, 1500)  # Randomize ask size
        new_row = pd.DataFrame([{
            "bid_price": bid_price,
            "ask_price": ask_price,
            "bid_size": bid_size,
            "ask_size": ask_size,
        }])
        self.order_book = pd.concat([self.order_book, new_row], ignore_index=True)
        logger.info(f"Updated order book with new price: {new_price}")

    def get_best_bid_ask(self) -> Tuple[float, float]:
        """
        Get the best bid and ask prices from the order book.

        Returns:
            Tuple[float, float]: Best bid and ask prices.
        """
        if self.order_book.empty:
            # Return a default tuple if order book is empty
            return (0.0, 0.0)
        
        best_bid = self.order_book["bid_price"].max()
        best_ask = self.order_book["ask_price"].min()
        return best_bid, best_ask
    
def main() -> None:
    """
    Main function to demonstrate the SimulatedLOB functionality.
    """
    lob = SimulatedLOB(initial_price=100.0, spread=0.05)
    print("Initial Best Bid and Ask:", lob.get_best_bid_ask())

    # Simulate price updates
    for price in [100.5, 101.0, 100.8, 101.2]:
        time.sleep(1)  # Simulate time delay
        lob.update_order_book(new_price=price)
        print("Updated Best Bid and Ask:", lob.get_best_bid_ask())
        
if __name__ == "__main__":
    main()