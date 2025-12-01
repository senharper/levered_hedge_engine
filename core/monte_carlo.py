"""
Monte Carlo Simulation Module

Runs bootstrap-based Monte Carlo simulations of the hedged portfolio strategy.
Uses historical returns to generate synthetic paths without Gaussian assumptions.
"""

import numpy as np
import pandas as pd
from typing import Dict
from dataclasses import dataclass

from config.strategy_config import StrategyConfig
from .portfolio import Portfolio
from .metrics import compute_max_drawdown


@dataclass
class MonteCarloSimulator:
    """
    Monte Carlo simulator for the leveraged hedge strategy.
    
    Uses bootstrap sampling of historical returns to generate
    synthetic market paths and simulate strategy performance.
    Supports both hedged and unhedged strategy modes.
    """
    
    ndx_returns: pd.Series
    strategy_config: StrategyConfig
    
    def run_simulation(self, n_paths: int, n_days: int, hedged: bool = True) -> pd.DataFrame:
        """
        Run Monte Carlo simulation with hedged or unhedged strategy.
        
        For each path:
        - Bootstrap-sample ndx_returns for n_days
        - Instantiate fresh Portfolio with appropriate config
        - Simulate day-by-day using portfolio.run_path()
        - Track final equity and max drawdown
        
        Args:
            n_paths: Number of simulation paths
            n_days: Number of days per simulation
            hedged: If True, use hedged strategy; if False, use unhedged
            
        Returns:
            DataFrame with columns:
                - path_id: Path identifier (0 to n_paths-1)
                - final_equity: Final portfolio value
                - total_return: Total return as percentage
                - max_drawdown: Maximum drawdown during path
        """
        # Create appropriate config based on hedged flag
        if hedged:
            # Use current config (default: 70% hedged, 30% unhedged)
            sim_config = self.strategy_config
        else:
            # Create unhedged version: 0% hedged, 100% unhedged
            sim_config = StrategyConfig(
                initial_capital=self.strategy_config.initial_capital,
                hedged_weight=0.0,
                unhedged_weight=1.0,
                unhedged_leverage=self.strategy_config.unhedged_leverage,
                hedged_up_beta=self.strategy_config.hedged_up_beta,
                hedged_down_beta=self.strategy_config.hedged_down_beta,
                crash_floor=self.strategy_config.crash_floor,
                annual_hedge_cost=0.0,  # No hedge cost for unhedged
                periods_per_year=self.strategy_config.periods_per_year,
            )
        
        results = []
        
        for path_id in range(n_paths):
            # Bootstrap sample returns with replacement
            sampled_indices = np.random.choice(len(self.ndx_returns), size=n_days, replace=True)
            sampled_returns = self.ndx_returns.iloc[sampled_indices].reset_index(drop=True)
            
            # Create fresh portfolio for this path
            portfolio = Portfolio(sim_config)
            
            # Simulate path using run_path (reuses existing daily P&L logic)
            path_df = portfolio.run_path(sampled_returns)
            
            # Extract metrics
            final_equity = path_df['total_value'].iloc[-1]
            initial_equity = sim_config.initial_capital
            total_return = (final_equity / initial_equity) - 1
            max_dd = compute_max_drawdown(path_df['total_value'])
            
            results.append({
                'path_id': path_id,
                'final_equity': final_equity,
                'total_return': total_return,
                'max_drawdown': max_dd,
            })
        
        return pd.DataFrame(results)
    
    def summary_stats(self, results: pd.DataFrame) -> Dict:
        """
        Compute summary statistics from simulation results.
        
        Args:
            results: DataFrame from run_simulation()
            
        Returns:
            Dictionary with summary statistics:
                - median_final_equity
                - p5_final_equity
                - p95_final_equity
                - mean_final_equity
                - std_final_equity
                - median_total_return
                - p5_total_return
                - p95_total_return
                - mean_total_return
                - median_max_drawdown
                - p5_max_drawdown (least negative, i.e., best case)
                - p95_max_drawdown (most negative, i.e., worst case)
                - mean_max_drawdown
        """
        return {
            'median_final_equity': results['final_equity'].median(),
            'p5_final_equity': results['final_equity'].quantile(0.05),
            'p95_final_equity': results['final_equity'].quantile(0.95),
            'mean_final_equity': results['final_equity'].mean(),
            'std_final_equity': results['final_equity'].std(),
            
            'median_total_return': results['total_return'].median(),
            'p5_total_return': results['total_return'].quantile(0.05),
            'p95_total_return': results['total_return'].quantile(0.95),
            'mean_total_return': results['total_return'].mean(),
            
            'median_max_drawdown': results['max_drawdown'].median(),
            'p5_max_drawdown': results['max_drawdown'].quantile(0.05),
            'p95_max_drawdown': results['max_drawdown'].quantile(0.95),
            'mean_max_drawdown': results['max_drawdown'].mean(),
        }
