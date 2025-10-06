# tools/financial_analytics.py
"""Financial analytics and valuation tools.

Provides functions for financial calculations including:
- Net Present Value (NPV)
- Internal Rate of Return (IRR)
- Volatility calculations
- Black-Scholes option pricing
"""

import numpy as np
from typing import List


def calculate_npv(rate: float, cashflows: List[float]) -> float:
    """Calculate Net Present Value.
    
    Args:
        rate: Discount rate (e.g., 0.1 for 10%)
        cashflows: List of cash flows (t=0, t=1, ...)
    
    Returns:
        NPV value
    """
    return sum(cf / (1 + rate)**t for t, cf in enumerate(cashflows))


def calculate_irr(cashflows: List[float]) -> float:
    """Calculate Internal Rate of Return.
    
    Args:
        cashflows: List of cash flows (t=0, t=1, ...)
    
    Returns:
        IRR value
    """
    from scipy.optimize import fsolve
    
    def npv_func(rate):
        return sum(cf / (1 + rate)**t for t, cf in enumerate(cashflows))
    
    try:
        irr = fsolve(npv_func, 0.1)[0]
        return float(irr)
    except Exception:
        return 0.0


def calculate_volatility(returns: List[float]) -> float:
    """Calculate volatility (standard deviation of returns).
    
    Args:
        returns: List of returns
    
    Returns:
        Volatility (std dev)
    """
    return float(np.std(returns))


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate Black-Scholes call option price.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Volatility
    
    Returns:
        Call option price
    """
    from scipy.stats import norm
    
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return float(call)


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate Black-Scholes put option price.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Volatility
    
    Returns:
        Put option price
    """
    from scipy.stats import norm
    
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return float(put)
