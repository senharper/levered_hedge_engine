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
import csv
from datetime import datetime

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
    
    def update_realtime(self, current_ndx_price: float, previous_ndx_price: float, 
                        current_date) -> Dict:
        """
        Update portfolio based on current NDX price (real-time).
        
        Computes ndx_return = (current_ndx_price / previous_ndx_price) - 1.
        Uses internal step logic to update equity, futures notional, and cash.
        
        Args:
            current_ndx_price: Current NDX price
            previous_ndx_price: Previous NDX price
            current_date: Current date
            
        Returns:
            Dictionary with: date, equity, futures_notional, cash, ndx_return
        """
        # Compute return
        ndx_return = (current_ndx_price / previous_ndx_price) - 1
        
        # Create a minimal series with this single return
        index_returns = pd.Series({current_date: ndx_return})
        
        # Run one step of simulation
        results_df = self.portfolio.run_path(index_returns)
        
        # Extract the final state
        final_row = results_df.iloc[-1]
        
        return {
            'date': current_date,
            'equity': final_row['total_value'],
            'hedged_value': final_row['hedged_value'],
            'unhedged_value': final_row['unhedged_value'],
            'hedged_weight': final_row['hedged_weight'],
            'unhedged_weight': final_row['unhedged_weight'],
            'ndx_price': current_ndx_price,
            'ndx_return': ndx_return,
        }
    
    def log_state_to_csv(self, path: str, record: Dict) -> None:
        """
        Append the provided record to a CSV file.
        Creates the file with headers if it does not exist.
        
        Args:
            path: Path to CSV file
            record: Dictionary with state to log
        """
        path = Path(path)
        file_exists = path.exists()
        
        with open(path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(record)
