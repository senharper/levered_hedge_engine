"""
Metrics Module

Calculates portfolio performance metrics including CAGR, Sharpe ratio,
and maximum drawdown.
"""

import numpy as np
import pandas as pd
from typing import Optional


def compute_cagr(values: pd.Series, periods_per_year: int) -> float:
    """
    Calculate Compound Annual Growth Rate.
    
    Args:
        values: Time series of portfolio values
        periods_per_year: Number of periods per year (e.g., 12 for monthly)
        
    Returns:
        CAGR as a decimal (e.g., 0.15 for 15%)
    """
    if len(values) == 0:
        return np.nan
    
    start = values.iloc[0]
    end = values.iloc[-1]
    n_periods = len(values)
    years = n_periods / periods_per_year
    
    if start <= 0 or end <= 0 or years <= 0:
        return np.nan
    
    return (end / start) ** (1 / years) - 1


def compute_max_drawdown(values: pd.Series) -> float:
    """
    Calculate maximum drawdown (peak-to-trough decline).
    
    Args:
        values: Time series of portfolio values
        
    Returns:
        Maximum drawdown as a negative decimal (e.g., -0.25 for 25% drawdown)
    """
    if len(values) == 0:
        return np.nan
    
    # Calculate running maximum
    running_max = values.cummax()
    
    # Calculate drawdowns from running max
    drawdowns = (values - running_max) / running_max
    
    # Return the minimum (most negative) drawdown
    return drawdowns.min()


def compute_sharpe(returns: pd.Series, periods_per_year: int, 
                   rf: float = 0.0) -> float:
    """
    Calculate annualised Sharpe ratio.
    
    Args:
        returns: Time series of period returns (not cumulative)
        periods_per_year: Number of periods per year (e.g., 12 for monthly)
        rf: Risk-free rate as annual decimal (default: 0.0)
        
    Returns:
        Sharpe ratio (annualised)
    """
    if len(returns) == 0:
        return np.nan
    
    # Convert annual risk-free rate to period rate
    rf_period = rf / periods_per_year
    
    # Calculate excess returns
    excess = returns - rf_period
    
    # Calculate mean and standard deviation
    mu = excess.mean()
    sigma = excess.std(ddof=1)
    
    # Handle zero volatility case
    if sigma == 0 or np.isnan(sigma):
        return np.nan
    
    # Annualise the Sharpe ratio
    sharpe = (mu * periods_per_year) / (sigma * np.sqrt(periods_per_year))
    
    return sharpe


def compute_volatility(returns: pd.Series, periods_per_year: int) -> float:
    """
    Calculate annualised volatility.
    
    Args:
        returns: Time series of period returns
        periods_per_year: Number of periods per year
        
    Returns:
        Annualised volatility as a decimal
    """
    if len(returns) == 0:
        return np.nan
    
    period_vol = returns.std(ddof=1)
    annual_vol = period_vol * np.sqrt(periods_per_year)
    
    return annual_vol


def compute_sortino(returns: pd.Series, periods_per_year: int,
                    rf: float = 0.0, target: float = 0.0) -> float:
    """
    Calculate annualised Sortino ratio (downside deviation).
    
    Args:
        returns: Time series of period returns
        periods_per_year: Number of periods per year
        rf: Risk-free rate as annual decimal
        target: Target return for downside calculation (default: 0.0)
        
    Returns:
        Sortino ratio (annualised)
    """
    if len(returns) == 0:
        return np.nan
    
    rf_period = rf / periods_per_year
    excess = returns - rf_period
    
    # Calculate downside deviation (only negative excess returns)
    downside_returns = excess[excess < target]
    
    if len(downside_returns) == 0:
        return np.nan
    
    downside_std = np.sqrt(np.mean(downside_returns ** 2))
    
    if downside_std == 0:
        return np.nan
    
    mu = excess.mean()
    sortino = (mu * periods_per_year) / (downside_std * np.sqrt(periods_per_year))
    
    return sortino


def compute_calmar(cagr: float, max_drawdown: float) -> float:
    """
    Calculate Calmar ratio (CAGR / absolute max drawdown).
    
    Args:
        cagr: Compound annual growth rate
        max_drawdown: Maximum drawdown (as negative number)
        
    Returns:
        Calmar ratio
    """
    if max_drawdown >= 0 or np.isnan(max_drawdown):
        return np.nan
    
    return cagr / abs(max_drawdown)


def compute_all_metrics(values: pd.Series, periods_per_year: int,
                       rf: float = 0.0) -> dict:
    """
    Calculate all available metrics for a portfolio time series.
    
    Args:
        values: Time series of portfolio values
        periods_per_year: Number of periods per year
        rf: Risk-free rate
        
    Returns:
        Dictionary containing all metrics
    """
    # Calculate returns
    returns = values.pct_change().dropna()
    
    # Calculate all metrics
    cagr = compute_cagr(values, periods_per_year)
    max_dd = compute_max_drawdown(values)
    sharpe = compute_sharpe(returns, periods_per_year, rf)
    volatility = compute_volatility(returns, periods_per_year)
    sortino = compute_sortino(returns, periods_per_year, rf)
    calmar = compute_calmar(cagr, max_dd)
    
    return {
        'cagr': cagr,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
        'volatility': volatility,
        'sortino': sortino,
        'calmar': calmar,
        'final_value': values.iloc[-1],
        'total_return': (values.iloc[-1] / values.iloc[0]) - 1,
    }
