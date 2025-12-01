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
from .reporting import MetricsReport


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
        self.report: Optional[MetricsReport] = None
    
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
        
        # Generate comprehensive report
        self.report = MetricsReport(self.results_df, self.config)
        
        return self.results_df
    
    def print_summary(self) -> None:
        """Print a formatted summary of backtest results."""
        if self.report is None:
            raise ValueError("Must run backtest before printing summary")
        
        print(self.report.to_text())
    
    def save_results(self, output_dir: Path) -> None:
        """
        Save backtest results to files.
        
        Args:
            output_dir: Directory to save output files
        """
        if self.results_df is None or self.report is None:
            raise ValueError("Must run backtest before saving results")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save time series data
        timeseries_path = output_dir / "portfolio_timeseries.csv"
        self.results_df.to_csv(timeseries_path)
        print(f"Saved time series to: {timeseries_path}")
        
        # Save all reports (summary.txt, summary.md, metrics.csv)
        self.report.save_reports(output_dir)
    
    def get_report(self) -> Optional[MetricsReport]:
        """
        Get the metrics report.
        
        Returns:
            MetricsReport object or None if backtest hasn't run
        """
        return self.report
    
    def get_results(self) -> Optional[pd.DataFrame]:
        """
        Get results DataFrame.
        
        Returns:
            DataFrame with time series or None if backtest hasn't run
        """
        return self.results_df
