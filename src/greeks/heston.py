"""
===========================================================================================
Title: heston.py
Author: Kenneth LeGare
Date: 2025-01-01

Description:
    This module implements the Heston stochastic volatility model.

Dependencies:
    - numpy
    - scipy

===========================================================================================
"""

# Third Party libraries
import numpy as np
import warnings


def heston_characteristic_function(
    u: np.ndarray,
    S: float,
    T: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
) -> np.ndarray:
    """
    Heston characteristic function with numerical stability improvements.

    args:
        u: Fourier transform variable
        S: Spot price of the underlying asset
        T: Time to expiration
        r: Risk-free interest rate
        q: Dividend yield
        v0: Initial volatility
        kappa: Mean reversion speed
        theta: Long-run average volatility
        sigma: Volatility of volatility
        rho: Correlation between asset and volatility

    returns:
        Value of the characteristic function
    """
    # Parameter validation
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if v0 <= 0:
        raise ValueError("v0 must be positive")
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    if theta <= 0:
        raise ValueError("theta must be positive")
    if abs(rho) >= 1:
        raise ValueError("rho must be in (-1, 1)")
    if T <= 0:
        raise ValueError("T must be positive")

    x = np.log(S)
    a = kappa * theta
    b = kappa

    # Compute discriminant with numerical stability
    discriminant = (rho * sigma * u * 1j - b) ** 2 + (sigma**2) * (u * 1j + u**2)
    # Use the principal branch of sqrt for complex numbers
    d = np.sqrt(discriminant)

    # Avoid division by zero and numerical instability
    denominator = b - rho * sigma * u * 1j - d
    numerator = b - rho * sigma * u * 1j + d

    # Use small epsilon to avoid exact zeros
    eps = 1e-15
    denominator = np.where(np.abs(denominator) < eps, eps, denominator)
    g = numerator / denominator

    # Numerical stability for exponentials
    dT = d * T
    # Clamp extreme values
    dT_real = np.real(dT)
    dT_real = np.clip(dT_real, -700, 700)  # Avoid overflow/underflow
    dT = dT_real + 1j * np.imag(dT)

    exp_dT = np.exp(dT)

    # Avoid log(0) and division by zero
    term1 = 1 - g * exp_dT
    term2 = 1 - g

    # Handle near-zero denominators
    term1 = np.where(np.abs(term1) < eps, eps, term1)
    term2 = np.where(np.abs(term2) < eps, eps, term2)

    # Safe logarithm
    log_term = np.log(term1 / term2)

    # Calculate components
    exp1 = 1j * u * (x + (r - q) * T)
    exp2 = (a / sigma**2) * (numerator * T - 2 * log_term)

    # Handle exp3 with numerical stability
    exp_term = 1 - exp_dT
    denom_exp3 = 1 - g * exp_dT
    denom_exp3 = np.where(np.abs(denom_exp3) < eps, eps, denom_exp3)
    exp3 = (v0 / sigma**2) * numerator * exp_term / denom_exp3

    # Final result with overflow protection
    exponent = exp1 + exp2 - exp3
    exponent_real = np.real(exponent)
    exponent_real = np.clip(exponent_real, -700, 700)
    exponent = exponent_real + 1j * np.imag(exponent)

    return np.exp(exponent)


def heston_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    alpha=1.5,
    N=4096,
    eta=0.25,
) -> float:
    """
    Heston option pricing using the original Heston (1993) semi-analytical formula.

    args:
        S: Spot price of the underlying asset
        K: Strike price of the option
        T: Time to expiration
        r: Risk-free interest rate
        q: Dividend yield
        v0: Initial volatility
        kappa: Mean reversion speed
        theta: Long-run average volatility
        sigma: Volatility of volatility
        rho: Correlation between asset and volatility
        alpha: Fourier transform parameter (unused in this implementation)
        N: Number of integration points
        eta: Integration step size (unused)

    returns:
        Price of the option
    """
    # Basic parameter validation
    if S <= 0 or K <= 0 or T <= 0:
        raise ValueError("S, K, and T must be positive")
    if T > 10:  # More than 10 years is likely an error
        raise ValueError(
            "T seems unreasonably large (>10 years). Check if using correct time units."
        )

    # Validate Heston parameters
    if v0 <= 0 or theta <= 0 or sigma <= 0 or kappa <= 0:
        raise ValueError("Heston parameters v0, theta, sigma, kappa must be positive")
    if abs(rho) >= 1:
        raise ValueError("Correlation rho must be in (-1, 1)")

    # Check Feller condition
    if 2 * kappa * theta <= sigma**2:
        warnings.warn(
            "Feller condition not satisfied: 2*kappa*theta <= sigma^2. This may cause issues."
        )

    try:
        from scipy import integrate

        def P1_integrand(phi):
            """Integrand for probability P1"""
            cf = heston_cf_p1(phi, S, T, r, q, v0, kappa, theta, sigma, rho)
            return np.real(np.exp(-1j * phi * np.log(K)) * cf / (1j * phi))

        def P2_integrand(phi):
            """Integrand for probability P2"""
            cf = heston_cf_p2(phi, S, T, r, q, v0, kappa, theta, sigma, rho)
            return np.real(np.exp(-1j * phi * np.log(K)) * cf / (1j * phi))

        # Integrate to get probabilities
        P1_integral, _ = integrate.quad(P1_integrand, 0.001, 100, limit=1000)
        P2_integral, _ = integrate.quad(P2_integrand, 0.001, 100, limit=1000)

        P1 = 0.5 + P1_integral / np.pi
        P2 = 0.5 + P2_integral / np.pi

        # Heston call price formula
        price = S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2

        # Ensure non-negative
        price = max(0, price)

        # Sanity check
        if np.isnan(price) or np.isinf(price) or price > 2 * S:
            raise ValueError(f"Unrealistic price: {price}")

        return price

    except Exception as e:
        # Fallback to Black-Scholes
        warnings.warn(f"Heston pricing failed ({e}), using Black-Scholes approximation")
        vol_approx = np.sqrt(v0)
        try:
            from .black_scholes import black_scholes_call

            S_adj = S * np.exp(-q * T)
            return black_scholes_call(S_adj, K, T, r, vol_approx)
        except Exception:
            return max(0, S * np.exp(-q * T) - K * np.exp(-r * T))


def heston_cf_p1(
    phi: float,
    S: float,
    T: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
) -> complex:
    """Characteristic function for P1 probability"""
    return heston_cf(phi, S, T, r, q, v0, kappa, theta, sigma, rho, 1)


def heston_cf_p2(
    phi: float,
    S: float,
    T: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
) -> complex:
    """Characteristic function for P2 probability"""
    return heston_cf(phi, S, T, r, q, v0, kappa, theta, sigma, rho, 0)


def heston_cf(
    phi: float,
    S: float,
    T: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    j: int,
) -> complex:
    """Heston characteristic function for probabilities P1 (j=1) and P2 (j=0)"""
    if j == 1:
        u = 0.5
        b = kappa - rho * sigma
    else:
        u = -0.5
        b = kappa

    a = kappa * theta

    d = np.sqrt(
        (rho * sigma * phi * 1j - b) ** 2 - sigma**2 * (2 * u * phi * 1j - phi**2)
    )
    g = (b - rho * sigma * phi * 1j + d) / (b - rho * sigma * phi * 1j - d)

    # Numerical stability improvements
    if np.abs(g) > 1e10:
        g = np.sign(g) * 1e10

    exp_term = np.exp(d * T)
    if np.abs(exp_term) > 1e10:
        exp_term = np.sign(exp_term) * 1e10

    C = (r - q) * phi * 1j * T + (a / sigma**2) * (
        (b - rho * sigma * phi * 1j + d) * T - 2 * np.log((1 - g * exp_term) / (1 - g))
    )
    D = (
        (b - rho * sigma * phi * 1j + d)
        / sigma**2
        * ((1 - exp_term) / (1 - g * exp_term))
    )

    return np.exp(C + D * v0 + 1j * phi * np.log(S))


def heston_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    h=1e-4,
) -> float:
    """
    Calculate the Delta of a Heston option.

    args:
        S: Spot price of the underlying asset
        K: Strike price of the option
        T: Time to expiration
        r: Risk-free interest rate
        q: Dividend yield
        v0: Initial volatility
        kappa: Mean reversion speed
        theta: Long-run average volatility
        sigma: Volatility of volatility
        rho: Correlation between asset and volatility
        h: Finite difference step size

    returns:
        Delta of the option
    """
    price_up = heston_price(S + h, K, T, r, q, v0, kappa, theta, sigma, rho)
    price_down = heston_price(S - h, K, T, r, q, v0, kappa, theta, sigma, rho)
    return (price_up - price_down) / (2 * h)


def heston_gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    h=1e-4,
) -> float:
    price_up = heston_price(S + h, K, T, r, q, v0, kappa, theta, sigma, rho)
    price_mid = heston_price(S, K, T, r, q, v0, kappa, theta, sigma, rho)
    price_down = heston_price(S - h, K, T, r, q, v0, kappa, theta, sigma, rho)
    return (price_up - 2 * price_mid + price_down) / (h**2)


def heston_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    h=1e-4,
) -> float:
    """
    Calculate the Vega of a Heston option.

    args:
        S: Spot price of the underlying asset
        K: Strike price of the option
        T: Time to expiration
        r: Risk-free interest rate
        q: Dividend yield
        v0: Initial volatility
        kappa: Mean reversion speed
        theta: Long-run average volatility
        sigma: Volatility of volatility
        rho: Correlation between asset and volatility
        h: Finite difference step size

    returns:
        Vega of the option
    """
    price_up = heston_price(S, K, T, r, q, v0 + h, kappa, theta, sigma, rho)
    price_down = heston_price(S, K, T, r, q, v0 - h, kappa, theta, sigma, rho)
    return (price_up - price_down) / (2 * h)


def heston_theta(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    h=1e-4,
) -> float:
    """
    Calculate the Theta of a Heston option.

    args:
        S: Spot price of the underlying asset
        K: Strike price of the option
        T: Time to expiration
        r: Risk-free interest rate
        q: Dividend yield
        v0: Initial volatility
        kappa: Mean reversion speed
        theta: Long-run average volatility
        sigma: Volatility of volatility
        rho: Correlation between asset and volatility
        h: Finite difference step size

    returns:
        Theta of the option
    """
    price_up = heston_price(S, K, T + h, r, q, v0, kappa, theta, sigma, rho)
    price_down = heston_price(S, K, T - h, r, q, v0, kappa, theta, sigma, rho)
    return -(price_up - price_down) / (2 * h)


def main():
    # Use realistic parameters for option pricing
    S = 100.0  # Spot price
    K = 100.0  # Strike price
    T = 0.25  # 3 months to expiration
    r = 0.05  # Risk-free rate
    q = 0.02  # Dividend yield
    v0 = 0.04  # Initial variance (2% volatility)
    kappa = 2.0  # Mean reversion speed
    theta_param = 0.04  # Long-run variance (renamed to avoid conflict)
    sigma = 0.3  # Vol of vol
    rho = -0.5  # Correlation

    print("Heston Model Greek Calculations")
    print(f"Parameters: S={S}, K={K}, T={T}, r={r}, q={q}")
    print(
        f"Heston: v0={v0}, kappa={kappa}, theta={theta_param}, sigma={sigma}, rho={rho}"
    )
    print("-" * 50)

    try:
        # Calculate Greeks with error handling
        delta = heston_delta(S, K, T, r, q, v0, kappa, theta_param, sigma, rho)
        gamma = heston_gamma(S, K, T, r, q, v0, kappa, theta_param, sigma, rho)
        vega = heston_vega(S, K, T, r, q, v0, kappa, theta_param, sigma, rho)
        theta_greek = heston_theta(S, K, T, r, q, v0, kappa, theta_param, sigma, rho)

        # Also calculate option price
        price = heston_price(S, K, T, r, q, v0, kappa, theta_param, sigma, rho)

        summary = {
            "Option Price": price,
            "Delta": delta,
            "Gamma": gamma,
            "Vega": vega,
            "Theta": theta_greek,
        }

        for key, value in summary.items():
            if np.isnan(value) or np.isinf(value):
                print(f"{key}: ERROR - {value}")
            else:
                print(f"{key}: {value:.6f}")

    except Exception as e:
        print(f"Error in calculations: {e}")
        print("This may indicate parameter issues or numerical instability.")


if __name__ == "__main__":
    main()
