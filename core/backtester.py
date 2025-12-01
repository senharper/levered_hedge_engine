"""
Backtester Module

Orchestrates the complete backtesting workflow:
- Load data
- Run portfolio simulation
- Calculate metrics
- Generate reports
"""

from pathlib import Path
from typing import Optional, Dict
import pandas as pd

from config.strategy_config import StrategyConfig
from .data_loader import DataLoader
from .portfolio import Portfolio
from .metrics import compute_cagr, compute_max_drawdown, compute_sharpe, compute_all_metrics


class Backtester:
    """
    Orchestrates the backtesting workflow for the Levered Hedge Engine.
    
    Handles data loading, portfolio simulation, metric calculation,
    and result reporting.
    """
    
    def __init__(self, config: StrategyConfig, data_path: Path):
        """
        Initialize the backtester.
        
        Args:
            config: Strategy configuration
            data_path: Path to index returns CSV file
        """
        self.config = config
        self.data_path = data_path
        self.loader = DataLoader(data_path)
        self.portfolio = Portfolio(config)
        
        # Results storage
        self.results_df: Optional[pd.DataFrame] = None
        self.metrics: Optional[Dict] = None
    
    def run(self, rebalance: bool = False, 
            rebalance_frequency: int = 12) -> pd.DataFrame:
        """
        Run the complete backtest.
        
        Args:
            rebalance: Whether to use periodic rebalancing (default: False)
            rebalance_frequency: Rebalance every N periods if rebalancing
            
        Returns:
            DataFrame with portfolio time series
        """
        # Load index returns
        index_returns = self.loader.load_index_returns()
        
        # Run portfolio simulation
        if rebalance:
            self.results_df = self.portfolio.run_path_with_rebalancing(
                index_returns, rebalance_frequency
            )
        else:
            self.results_df = self.portfolio.run_path(index_returns)
        
        # Calculate metrics
        self.metrics = self._calculate_metrics()
        
        return self.results_df
    
    def _calculate_metrics(self) -> Dict:
        """
        Calculate performance metrics for both portfolio and benchmark.
        
        Returns:
            Dictionary with portfolio and index metrics
        """
        if self.results_df is None:
            raise ValueError("Must run backtest before calculating metrics")
        
        # Portfolio metrics
        portfolio_metrics = compute_all_metrics(
            self.results_df['total_value'],
            self.config.periods_per_year
        )
        
        # Index metrics
        index_metrics = compute_all_metrics(
            self.results_df['index_value'],
            self.config.periods_per_year
        )
        
        return {
            'portfolio': portfolio_metrics,
            'index': index_metrics,
        }
    
    def print_summary(self) -> None:
        """Print a formatted summary of backtest results."""
        if self.metrics is None:
            raise ValueError("Must run backtest before printing summary")
        
        port = self.metrics['portfolio']
        idx = self.metrics['index']
        
        print("\n" + "="*60)
        print("LEVERED HEDGE ENGINE - BACKTEST SUMMARY")
        print("="*60)
        
        print("\n--- Portfolio Performance ---")
        print(f"Final Value:      ${port['final_value']:,.2f}")
        print(f"Total Return:     {port['total_return']:.2%}")
        print(f"CAGR:             {port['cagr']:.2%}")
        print(f"Max Drawdown:     {port['max_drawdown']:.2%}")
        print(f"Volatility:       {port['volatility']:.2%}")
        print(f"Sharpe Ratio:     {port['sharpe']:.2f}")
        print(f"Sortino Ratio:    {port['sortino']:.2f}")
        print(f"Calmar Ratio:     {port['calmar']:.2f}")
        
        print("\n--- Index Performance ---")
        print(f"Final Value:      {idx['final_value']:.4f}")
        print(f"Total Return:     {idx['total_return']:.2%}")
        print(f"CAGR:             {idx['cagr']:.2%}")
        print(f"Max Drawdown:     {idx['max_drawdown']:.2%}")
        print(f"Volatility:       {idx['volatility']:.2%}")
        print(f"Sharpe Ratio:     {idx['sharpe']:.2f}")
        print(f"Sortino Ratio:    {idx['sortino']:.2f}")
        print(f"Calmar Ratio:     {idx['calmar']:.2f}")
        
        print("\n--- Outperformance ---")
        print(f"Alpha (CAGR):     {port['cagr'] - idx['cagr']:.2%}")
        print(f"Sharpe Advantage: {port['sharpe'] - idx['sharpe']:.2f}")
        
        print("\n--- Configuration ---")
        print(f"Initial Capital:  ${self.config.initial_capital:,.2f}")
        print(f"Hedged Weight:    {self.config.hedged_weight:.1%}")
        print(f"Unhedged Weight:  {self.config.unhedged_weight:.1%}")
        print(f"Hedge Cost:       {self.config.annual_hedge_cost:.2%} p.a.")
        
        print("\n" + "="*60 + "\n")
    
    def save_results(self, output_dir: Path) -> None:
        """
        Save backtest results to files.
        
        Args:
            output_dir: Directory to save output files
        """
        if self.results_df is None or self.metrics is None:
            raise ValueError("Must run backtest before saving results")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save time series data
        timeseries_path = output_dir / "portfolio_timeseries.csv"
        self.results_df.to_csv(timeseries_path)
        print(f"Saved time series to: {timeseries_path}")
        
        # Save summary metrics
        summary_path = output_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            # Redirect print to file
            import sys
            original_stdout = sys.stdout
            sys.stdout = f
            self.print_summary()
            sys.stdout = original_stdout
        
        print(f"Saved summary to: {summary_path}")
    
    def get_metrics(self) -> Optional[Dict]:
        """
        Get calculated metrics.
        
        Returns:
            Dictionary with metrics or None if backtest hasn't run
        """
        return self.metrics
    
    def get_results(self) -> Optional[pd.DataFrame]:
        """
        Get results DataFrame.
        
        Returns:
            DataFrame with time series or None if backtest hasn't run
        """
        return self.results_df
