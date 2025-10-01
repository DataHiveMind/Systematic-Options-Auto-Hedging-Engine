"""
===========================================================
Title: black_scholes.py
Author: Kenneth LeGare
Data: 2025-01-01

Description:
    This module implements the Black-Scholes option pricing model.

Dependencies:
    - numpy
    - scipy
"""

import numpy as np
import scipy.stats as si

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the Black-Scholes call option price.

    Args:
        S (float): Current stock price
        K (float): Option strike price
        T (float): Time to expiration (in years)
        r (float): Risk-free interest rate (annualized)
        sigma (float): Volatility of the underlying stock (annualized)

    Returns:
        float: Black-Scholes call option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2))
    return call_price

def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the Black-Scholes put option price.

    Args:
        S (float): Current stock price
        K (float): Option strike price
        T (float): Time to expiration (in years)
        r (float): Risk-free interest rate (annualized)
        sigma (float): Volatility of the underlying stock (annualized)

    Returns:
        float: Black-Scholes put option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = (K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1))
    return put_price

def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the Black-Scholes Gamma for a European option.

    Args:
        S (float): Current stock price
        K (float): Option strike price
        T (float): Time to expiration (in years)
        r (float): Risk-free interest rate (annualized)
        sigma (float): Volatility of the underlying stock (annualized)

    Returns:
        float: Black-Scholes Gamma
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the Black-Scholes Vega for a European option.

    Args:
        S (float): Current stock price
        K (float): Option strike price
        T (float): Time to expiration (in years)
        r (float): Risk-free interest rate (annualized)
        sigma (float): Volatility of the underlying stock (annualized)

    Returns:
        float: Black-Scholes Vega
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    vega = S * si.norm.pdf(d1) * np.sqrt(T)
    return vega

def theta(S: float, K: float, T: float, r: float, sigma: float, position: str) -> float:
    """
    Calculate the Black-Scholes Theta for a European option.

    Args:
        S (float): Current stock price
        K (float): Option strike price
        T (float): Time to expiration (in years)
        r (float): Risk-free interest rate (annualized)
        sigma (float): Volatility of the underlying stock (annualized)

    Returns:
        float: Black-Scholes Theta
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if position == "call":
        theta = (-S * si.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
                  r * K * np.exp(-r * T) * si.norm.cdf(d2))
    elif position == "put":
        theta = (-S * si.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) +
                 r * K * np.exp(-r * T) * si.norm.cdf(-d2))
    else:
        raise ValueError("position must be either 'call' or 'put'")
    return theta

def rho(S: float, K: float, T: float, r: float, sigma: float, position: str) -> float:
    """
    Calculate the Black-Scholes Rho for a European option.

    Args:
        S (float): Current stock price
        K (float): Option strike price
        T (float): Time to expiration (in years)
        r (float): Risk-free interest rate (annualized)
        sigma (float): Volatility of the underlying stock (annualized)
        position (str): Position type ("call" or "put")

    Returns:
        float: Black-Scholes Rho
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if position == "call":
        rho = K * T * np.exp(-r * T) * si.norm.cdf(d2)
    elif position == "put":
        rho = -K * T * np.exp(-r * T) * si.norm.cdf(-d2)
    else:
        raise ValueError("position must be either 'call' or 'put'")
    return rho

def Summary(S: float, K: float, T: float, r: float, sigma: float) -> dict:
    """
    Calculate the Black-Scholes Greeks (Delta, Gamma, Vega, Theta, rho) for a European option.

    Args:
        S (float): Current stock price
        K (float): Option strike price
        T (float): Time to expiration (in years)
        r (float): Risk-free interest rate (annualized)
        sigma (float): Volatility of the underlying stock (annualized)

    Returns:
        dict: A dictionary containing the values of Delta, Gamma, Vega, and Theta
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = black_scholes_call(S, K, T, r, sigma)
    put_price = black_scholes_put(S, K, T, r, sigma)
    gamma_val = gamma(S, K, T, r, sigma)
    vega_val = vega(S, K, T, r, sigma)
    theta_call = theta(S, K, T, r, sigma, "call")
    theta_put = theta(S, K, T, r, sigma, "put")
    rho_call = rho(S, K, T, r, sigma, "call")
    rho_put = rho(S, K, T, r, sigma, "put")
    return {
        "call_price": call_price,
        "put_price": put_price,
        "gamma": gamma_val,
        "vega": vega_val,
        "theta_call": theta_call,
        "theta_put": theta_put,
        "rho_call": rho_call,
        "rho_put": rho_put,
        "d1": d1,
        "d2": d2
    }

def main():
    S = 100  # Current stock price
    K = 100  # Option strike price
    T = 1    # Time to expiration (in years)
    r = 0.05 # Risk-free interest rate (annualized)
    sigma = 0.2 # Volatility of the underlying stock (annualized)

    call_price = black_scholes_call(S, K, T, r, sigma)
    put_price = black_scholes_put(S, K, T, r, sigma)
    greeks = Summary(S, K, T, r, sigma)

    print(f"Call Price: {call_price}")
    print(f"Put Price: {put_price}")
    print("Greeks:")
    for greek, value in greeks.items():
        print(f"  {greek}: {value}")


if __name__ == "__main__":
    main()