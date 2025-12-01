"""
Reporting Module

Generates comprehensive metrics reports for backtest results,
including metrics for index, hedged sleeve, unhedged sleeve, and total portfolio.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np

from config.strategy_config import StrategyConfig
from .metrics import (
    compute_cagr, compute_max_drawdown, compute_sharpe,
    compute_volatility, compute_sortino, compute_calmar, compute_all_metrics
)


@dataclass
class MetricsReport:
    """
    Comprehensive metrics report for backtest results.
    
    Computes and stores metrics for:
    - Index (benchmark)
    - Hedged sleeve
    - Unhedged sleeve
    - Total portfolio
    
    Metrics include: final value, total return, CAGR, annual volatility,
    max drawdown, Sharpe ratio, Sortino ratio, and Calmar ratio.
    """
    
    results_df: pd.DataFrame
    config: StrategyConfig
    metrics: Dict[str, Dict] = None
    
    def __post_init__(self):
        """Compute all metrics upon initialization."""
        if self.metrics is None:
            self.metrics = self._compute_all_metrics()
    
    def _compute_all_metrics(self) -> Dict[str, Dict]:
        """
        Compute metrics for all portfolio components.
        
        Returns:
            Dictionary with metrics keyed by component name
            (index, hedged, unhedged, total)
        """
        rf = 0.0  # Risk-free rate (default)
        periods_per_year = self.config.periods_per_year
        
        metrics = {}
        
        # Index metrics
        metrics['index'] = compute_all_metrics(
            self.results_df['index_value'],
            periods_per_year,
            rf
        )
        
        # Hedged sleeve metrics
        metrics['hedged'] = compute_all_metrics(
            self.results_df['hedged_value'],
            periods_per_year,
            rf
        )
        
        # Unhedged sleeve metrics
        metrics['unhedged'] = compute_all_metrics(
            self.results_df['unhedged_value'],
            periods_per_year,
            rf
        )
        
        # Total portfolio metrics
        metrics['total'] = compute_all_metrics(
            self.results_df['total_value'],
            periods_per_year,
            rf
        )
        
        return metrics
    
    def get_metrics(self, component: str = 'total') -> Dict:
        """
        Retrieve metrics for a specific component.
        
        Args:
            component: One of 'index', 'hedged', 'unhedged', or 'total'
            
        Returns:
            Dictionary with metrics for the component
        """
        if component not in self.metrics:
            raise ValueError(f"Unknown component: {component}. "
                           f"Must be one of {list(self.metrics.keys())}")
        return self.metrics[component]
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert metrics to a DataFrame format for export.
        
        Returns:
            DataFrame with metrics as rows, components as columns
        """
        data = {}
        
        for component, component_metrics in self.metrics.items():
            data[component] = component_metrics
        
        df = pd.DataFrame(data)
        return df
    
    def to_markdown(self) -> str:
        """
        Generate a formatted markdown report.
        
        Returns:
            Formatted markdown string
        """
        lines = []
        lines.append("# Backtest Summary Report\n")
        
        # Portfolio Performance
        lines.append("## Portfolio Performance\n")
        port = self.metrics['total']
        lines.append(f"- **Final Value**: ${port['final_value']:,.2f}")
        lines.append(f"- **Total Return**: {port['total_return']:.2%}")
        lines.append(f"- **CAGR**: {port['cagr']:.2%}")
        lines.append(f"- **Annual Volatility**: {port['volatility']:.2%}")
        lines.append(f"- **Max Drawdown**: {port['max_drawdown']:.2%}")
        lines.append(f"- **Sharpe Ratio**: {port['sharpe']:.2f}")
        lines.append(f"- **Sortino Ratio**: {port['sortino']:.2f}")
        lines.append(f"- **Calmar Ratio**: {port['calmar']:.2f}\n")
        
        # Index Performance
        lines.append("## Index Performance\n")
        idx = self.metrics['index']
        lines.append(f"- **Final Value**: {idx['final_value']:.4f}")
        lines.append(f"- **Total Return**: {idx['total_return']:.2%}")
        lines.append(f"- **CAGR**: {idx['cagr']:.2%}")
        lines.append(f"- **Annual Volatility**: {idx['volatility']:.2%}")
        lines.append(f"- **Max Drawdown**: {idx['max_drawdown']:.2%}")
        lines.append(f"- **Sharpe Ratio**: {idx['sharpe']:.2f}")
        lines.append(f"- **Sortino Ratio**: {idx['sortino']:.2f}")
        lines.append(f"- **Calmar Ratio**: {idx['calmar']:.2f}\n")
        
        # Outperformance
        lines.append("## Outperformance\n")
        lines.append(f"- **Alpha (CAGR)**: {port['cagr'] - idx['cagr']:.2%}")
        lines.append(f"- **Sharpe Advantage**: {port['sharpe'] - idx['sharpe']:.2f}")
        lines.append(f"- **Volatility Difference**: {port['volatility'] - idx['volatility']:.2%}\n")
        
        # Sleeve Performance
        lines.append("## Sleeve Performance\n")
        hedged = self.metrics['hedged']
        unhedged = self.metrics['unhedged']
        
        lines.append("### Hedged Sleeve")
        lines.append(f"- **Final Value**: ${hedged['final_value']:,.2f}")
        lines.append(f"- **CAGR**: {hedged['cagr']:.2%}")
        lines.append(f"- **Max Drawdown**: {hedged['max_drawdown']:.2%}")
        lines.append(f"- **Sharpe Ratio**: {hedged['sharpe']:.2f}\n")
        
        lines.append("### Unhedged Sleeve")
        lines.append(f"- **Final Value**: ${unhedged['final_value']:,.2f}")
        lines.append(f"- **CAGR**: {unhedged['cagr']:.2%}")
        lines.append(f"- **Max Drawdown**: {unhedged['max_drawdown']:.2%}")
        lines.append(f"- **Sharpe Ratio**: {unhedged['sharpe']:.2f}\n")
        
        # Configuration
        lines.append("## Configuration\n")
        lines.append(f"- **Initial Capital**: ${self.config.initial_capital:,.2f}")
        lines.append(f"- **Hedged Weight**: {self.config.hedged_weight:.1%}")
        lines.append(f"- **Unhedged Weight**: {self.config.unhedged_weight:.1%}")
        lines.append(f"- **Hedge Cost**: {self.config.annual_hedge_cost:.2%} p.a.")
        lines.append(f"- **Periods Per Year**: {self.config.periods_per_year}")
        
        return "\n".join(lines)
    
    def to_text(self) -> str:
        """
        Generate a formatted text report.
        
        Returns:
            Formatted text string
        """
        lines = []
        
        lines.append("=" * 70)
        lines.append("LEVERED HEDGE ENGINE - BACKTEST SUMMARY")
        lines.append("=" * 70)
        
        # Portfolio Performance
        lines.append("\n--- Portfolio Performance ---\n")
        port = self.metrics['total']
        lines.append(f"Final Value:          ${port['final_value']:>15,.2f}")
        lines.append(f"Total Return:         {port['total_return']:>15.2%}")
        lines.append(f"CAGR:                 {port['cagr']:>15.2%}")
        lines.append(f"Annual Volatility:    {port['volatility']:>15.2%}")
        lines.append(f"Max Drawdown:         {port['max_drawdown']:>15.2%}")
        lines.append(f"Sharpe Ratio:         {port['sharpe']:>15.2f}")
        lines.append(f"Sortino Ratio:        {port['sortino']:>15.2f}")
        lines.append(f"Calmar Ratio:         {port['calmar']:>15.2f}")
        
        # Index Performance
        lines.append("\n--- Index Performance ---\n")
        idx = self.metrics['index']
        lines.append(f"Final Value:          {idx['final_value']:>15.4f}")
        lines.append(f"Total Return:         {idx['total_return']:>15.2%}")
        lines.append(f"CAGR:                 {idx['cagr']:>15.2%}")
        lines.append(f"Annual Volatility:    {idx['volatility']:>15.2%}")
        lines.append(f"Max Drawdown:         {idx['max_drawdown']:>15.2%}")
        lines.append(f"Sharpe Ratio:         {idx['sharpe']:>15.2f}")
        lines.append(f"Sortino Ratio:        {idx['sortino']:>15.2f}")
        lines.append(f"Calmar Ratio:         {idx['calmar']:>15.2f}")
        
        # Outperformance
        lines.append("\n--- Outperformance vs Index ---\n")
        lines.append(f"Alpha (CAGR):         {port['cagr'] - idx['cagr']:>15.2%}")
        lines.append(f"Sharpe Advantage:     {port['sharpe'] - idx['sharpe']:>15.2f}")
        lines.append(f"Volatility Diff:      {port['volatility'] - idx['volatility']:>15.2%}")
        
        # Sleeve Performance
        lines.append("\n--- Hedged Sleeve ---\n")
        hedged = self.metrics['hedged']
        lines.append(f"Final Value:          ${hedged['final_value']:>15,.2f}")
        lines.append(f"CAGR:                 {hedged['cagr']:>15.2%}")
        lines.append(f"Max Drawdown:         {hedged['max_drawdown']:>15.2%}")
        lines.append(f"Sharpe Ratio:         {hedged['sharpe']:>15.2f}")
        
        lines.append("\n--- Unhedged Sleeve ---\n")
        unhedged = self.metrics['unhedged']
        lines.append(f"Final Value:          ${unhedged['final_value']:>15,.2f}")
        lines.append(f"CAGR:                 {unhedged['cagr']:>15.2%}")
        lines.append(f"Max Drawdown:         {unhedged['max_drawdown']:>15.2%}")
        lines.append(f"Sharpe Ratio:         {unhedged['sharpe']:>15.2f}")
        
        # Configuration
        lines.append("\n--- Configuration ---\n")
        lines.append(f"Initial Capital:      ${self.config.initial_capital:>15,.2f}")
        lines.append(f"Hedged Weight:        {self.config.hedged_weight:>15.1%}")
        lines.append(f"Unhedged Weight:      {self.config.unhedged_weight:>15.1%}")
        lines.append(f"Annual Hedge Cost:    {self.config.annual_hedge_cost:>15.2%}")
        lines.append(f"Periods Per Year:     {self.config.periods_per_year:>15d}")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)
    
    def save_reports(self, output_dir: Path) -> None:
        """
        Save all reports to output directory.
        
        Generates:
        - summary.txt (formatted text report)
        - summary.md (formatted markdown report)
        - metrics.csv (metrics table export)
        
        Args:
            output_dir: Directory to save output files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save text report
        text_path = output_dir / "summary.txt"
        with open(text_path, 'w') as f:
            f.write(self.to_text())
        print(f"Saved text report to: {text_path}")
        
        # Save markdown report
        md_path = output_dir / "summary.md"
        with open(md_path, 'w') as f:
            f.write(self.to_markdown())
        print(f"Saved markdown report to: {md_path}")
        
        # Save metrics CSV
        metrics_path = output_dir / "metrics.csv"
        metrics_df = self.to_dataframe()
        metrics_df.to_csv(metrics_path)
        print(f"Saved metrics CSV to: {metrics_path}")
