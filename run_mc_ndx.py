"""
Monte Carlo Analysis: Hedged vs Unhedged Strategy Comparison

Runs 1000 simulations over 252 days (1 year) comparing:
- Hedged strategy (70% hedged, 30% unhedged, 3% annual hedge cost)
- Unhedged strategy (0% hedged, 100% unhedged, no hedge cost)

Computes outcome probabilities and downside protection metrics.
"""

import pandas as pd
from pathlib import Path
import numpy as np

from config.strategy_config import StrategyConfig
from core.data_loader import DataLoader
from core.monte_carlo import MonteCarloSimulator


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)


def print_stats(label: str, stats: dict):
    """Pretty-print simulation statistics."""
    print(f"\n{label}:")
    print(f"  Median Final Equity:     ${stats['median_final_equity']:>12,.2f}")
    print(f"  Mean Final Equity:       ${stats['mean_final_equity']:>12,.2f}")
    print(f"  Std Dev:                 ${stats['std_final_equity']:>12,.2f}")
    print(f"\n  5th Percentile (P5):     ${stats['p5_final_equity']:>12,.2f}")
    print(f"  95th Percentile (P95):   ${stats['p95_final_equity']:>12,.2f}")
    print(f"\n  Median Total Return:     {stats['median_total_return']:>12.2%}")
    print(f"  Mean Total Return:       {stats['mean_total_return']:>12.2%}")
    print(f"  P5 Total Return:         {stats['p5_total_return']:>12.2%}")
    print(f"  P95 Total Return:        {stats['p95_total_return']:>12.2%}")
    print(f"\n  Median Max Drawdown:     {stats['median_max_drawdown']:>12.2%}")
    print(f"  Mean Max Drawdown:       {stats['mean_max_drawdown']:>12.2%}")
    print(f"  P5 Max Drawdown (best):  {stats['p5_max_drawdown']:>12.2%}")
    print(f"  P95 Max Drawdown (worst):{stats['p95_max_drawdown']:>12.2%}")


def main():
    """
    Main Monte Carlo analysis comparing hedged vs unhedged strategies.
    """
    
    # Configuration
    data_path = Path("data/ndx_returns_sample.csv")
    n_paths = 1000
    n_days = 252
    
    print_section("MONTE CARLO ANALYSIS: HEDGED VS UNHEDGED")
    print(f"\nSimulation Parameters:")
    print(f"  Number of Paths:  {n_paths}")
    print(f"  Days per Path:    {n_days} (1 year)")
    print(f"  Data Source:      {data_path}")
    
    # Load returns
    print(f"\nLoading NDX returns from {data_path}...")
    loader = DataLoader(data_path)
    ndx_returns = loader.load_index_returns()
    print(f"  Loaded {len(ndx_returns)} daily returns")
    print(f"  Date range: {ndx_returns.index.min()} to {ndx_returns.index.max()}")
    print(f"  Mean return: {ndx_returns.mean():.4%}")
    print(f"  Std dev: {ndx_returns.std():.4%}")
    
    # Create base configuration (hedged strategy)
    print_section("STRATEGY CONFIGURATIONS")
    
    cfg_hedged = StrategyConfig(
        hedged_weight=0.7,
        unhedged_weight=0.3,
        annual_hedge_cost=0.03
    )
    
    print("\nHedged Strategy:")
    print(f"  Initial Capital:    ${cfg_hedged.initial_capital:,.2f}")
    print(f"  Hedged Weight:      {cfg_hedged.hedged_weight:.1%}")
    print(f"  Unhedged Weight:    {cfg_hedged.unhedged_weight:.1%}")
    print(f"  Annual Hedge Cost:  {cfg_hedged.annual_hedge_cost:.2%}")
    print(f"  Unhedged Leverage:  {cfg_hedged.unhedged_leverage:.1f}x")
    print(f"  Hedged Up Beta:     {cfg_hedged.hedged_up_beta:.2f}")
    print(f"  Hedged Down Beta:   {cfg_hedged.hedged_down_beta:.2f}")
    print(f"  Crash Floor:        {cfg_hedged.crash_floor:.2%}")
    
    print("\nUnhedged Strategy:")
    print(f"  Initial Capital:    ${cfg_hedged.initial_capital:,.2f}")
    print(f"  Hedged Weight:      0.0%")
    print(f"  Unhedged Weight:    100.0%")
    print(f"  Annual Hedge Cost:  0.0%")
    print(f"  Unhedged Leverage:  {cfg_hedged.unhedged_leverage:.1f}x")
    
    # Run simulations
    print_section("RUNNING MONTE CARLO SIMULATIONS")
    
    print(f"\nHedged Strategy: Running {n_paths} paths...")
    mc_hedged = MonteCarloSimulator(ndx_returns, cfg_hedged)
    results_hedged = mc_hedged.run_simulation(n_paths, n_days, hedged=True)
    print(f"  [OK] Completed")
    
    print(f"\nUnhedged Strategy: Running {n_paths} paths...")
    results_unhedged = mc_hedged.run_simulation(n_paths, n_days, hedged=False)
    print(f"  [OK] Completed")
    
    # Compute stats
    print_section("COMPUTING STATISTICS")
    
    print("\nHedged strategy...")
    stats_hedged = mc_hedged.summary_stats(results_hedged)
    
    print("Unhedged strategy...")
    stats_unhedged = mc_hedged.summary_stats(results_unhedged)
    
    # Display results
    print_section("RESULTS: HEDGED STRATEGY")
    print_stats("Hedged Strategy", stats_hedged)
    
    print_section("RESULTS: UNHEDGED STRATEGY")
    print_stats("Unhedged Strategy", stats_unhedged)
    
    # Comparative analysis
    print_section("COMPARATIVE ANALYSIS")
    
    # Count outcomes
    hedged_better_equity = (results_hedged['final_equity'] > results_unhedged['final_equity']).sum()
    hedged_better_drawdown = (results_hedged['max_drawdown'] > results_unhedged['max_drawdown']).sum()
    
    pct_hedged_better_equity = (hedged_better_equity / n_paths) * 100
    pct_hedged_better_drawdown = (hedged_better_drawdown / n_paths) * 100
    
    print(f"\nHedged > Unhedged (Final Equity):")
    print(f"  {hedged_better_equity}/{n_paths} paths ({pct_hedged_better_equity:.1f}%)")
    
    print(f"\nHedged > Unhedged (Max Drawdown - less negative is better):")
    print(f"  {hedged_better_drawdown}/{n_paths} paths ({pct_hedged_better_drawdown:.1f}%)")
    
    # Equity advantage
    equity_diff = stats_hedged['median_final_equity'] - stats_unhedged['median_final_equity']
    equity_diff_pct = (equity_diff / stats_unhedged['median_final_equity']) * 100
    
    print(f"\nMedian Final Equity Difference:")
    print(f"  Hedged - Unhedged: ${equity_diff:,.2f} ({equity_diff_pct:+.2f}%)")
    
    # Drawdown advantage
    dd_diff = stats_hedged['median_max_drawdown'] - stats_unhedged['median_max_drawdown']
    
    print(f"\nMedian Max Drawdown Difference (less negative = better):")
    print(f"  Hedged - Unhedged: {dd_diff:+.2%}")
    print(f"  Downside Protection: {abs(dd_diff):.2%}")
    
    # Efficiency ratio (return vs drawdown)
    hedged_efficiency = stats_hedged['median_total_return'] / abs(stats_hedged['median_max_drawdown']) if stats_hedged['median_max_drawdown'] != 0 else np.inf
    unhedged_efficiency = stats_unhedged['median_total_return'] / abs(stats_unhedged['median_max_drawdown']) if stats_unhedged['median_max_drawdown'] != 0 else np.inf
    
    print(f"\nReturn/Risk Efficiency (Median Return / |Max Drawdown|):")
    print(f"  Hedged:    {hedged_efficiency:.2f}x")
    print(f"  Unhedged:  {unhedged_efficiency:.2f}x")
    
    # Save results
    print_section("SAVING RESULTS")
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Save simulation results
    hedged_path = output_dir / "mc_results_hedged.csv"
    unhedged_path = output_dir / "mc_results_unhedged.csv"
    
    results_hedged.to_csv(hedged_path, index=False)
    results_unhedged.to_csv(unhedged_path, index=False)
    
    print(f"\nHedged results:   {hedged_path}")
    print(f"Unhedged results: {unhedged_path}")
    
    # Save summary
    summary_path = output_dir / "mc_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("MONTE CARLO ANALYSIS: HEDGED VS UNHEDGED\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Simulation Parameters:\n")
        f.write(f"  Paths: {n_paths}\n")
        f.write(f"  Days per path: {n_days}\n")
        f.write(f"  Initial capital: ${cfg_hedged.initial_capital:,.2f}\n\n")
        
        f.write("HEDGED STRATEGY STATS:\n")
        f.write("-" * 70 + "\n")
        for key, val in stats_hedged.items():
            if 'equity' in key:
                f.write(f"  {key}: ${val:,.2f}\n")
            elif 'drawdown' in key:
                f.write(f"  {key}: {val:.2%}\n")
            else:
                f.write(f"  {key}: {val:.2%}\n")
        
        f.write("\nUNHEDGED STRATEGY STATS:\n")
        f.write("-" * 70 + "\n")
        for key, val in stats_unhedged.items():
            if 'equity' in key:
                f.write(f"  {key}: ${val:,.2f}\n")
            elif 'drawdown' in key:
                f.write(f"  {key}: {val:.2%}\n")
            else:
                f.write(f"  {key}: {val:.2%}\n")
        
        f.write("\nCOMPARATIVE ANALYSIS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Hedged > Unhedged (Equity): {hedged_better_equity}/{n_paths} ({pct_hedged_better_equity:.1f}%)\n")
        f.write(f"  Hedged > Unhedged (Drawdown): {hedged_better_drawdown}/{n_paths} ({pct_hedged_better_drawdown:.1f}%)\n")
        f.write(f"  Median Equity Difference: ${equity_diff:,.2f} ({equity_diff_pct:+.2f}%)\n")
        f.write(f"  Downside Protection: {abs(dd_diff):.2%}\n")
        f.write(f"  Return/Risk Efficiency (Hedged): {hedged_efficiency:.2f}x\n")
        f.write(f"  Return/Risk Efficiency (Unhedged): {unhedged_efficiency:.2f}x\n")
    
    print(f"Summary:          {summary_path}")
    
    print_section("ANALYSIS COMPLETE")
    print("\n")


if __name__ == "__main__":
    main()


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)


def print_stats(label: str, stats: dict):
    """Pretty-print statistics."""
    print(f"\n{label}:")
    print(f"  Median Final Equity:     ${stats['median_final_equity']:>12,.2f}")
    print(f"  Mean Final Equity:       ${stats['mean_final_equity']:>12,.2f}")
    print(f"  Std Dev:                 ${stats['std_final_equity']:>12,.2f}")
    print(f"\n  5th Percentile (P5):     ${stats['p5_final_equity']:>12,.2f}")
    print(f"  95th Percentile (P95):   ${stats['p95_final_equity']:>12,.2f}")
    print(f"\n  Median Total Return:     {stats['median_total_return']:>12.2%}")
    print(f"  Mean Total Return:       {stats['mean_total_return']:>12.2%}")
    print(f"  P5 Total Return:         {stats['p5_total_return']:>12.2%}")
    print(f"  P95 Total Return:        {stats['p95_total_return']:>12.2%}")
    print(f"\n  Median Max Drawdown:     {stats['median_max_drawdown']:>12.2%}")
    print(f"  Mean Max Drawdown:       {stats['mean_max_drawdown']:>12.2%}")
    print(f"  P5 Max Drawdown (best):  {stats['p5_max_drawdown']:>12.2%}")
    print(f"  P95 Max Drawdown (worst):{stats['p95_max_drawdown']:>12.2%}")


def main():
    """
    Main Monte Carlo analysis.
    """
    
    # Configuration
    data_path = Path("data/ndx_returns_sample.csv")
    n_paths = 1000
    n_days = 252
    
    print_section("MONTE CARLO ANALYSIS: HEDGED VS UNHEDGED")
    print(f"\nSimulation Parameters:")
    print(f"  Number of Paths:  {n_paths}")
    print(f"  Days per Path:    {n_days} (1 year)")
    print(f"  Data Source:      {data_path}")
    
    # Load returns
    print(f"\nLoading NDX returns from {data_path}...")
    loader = DataLoader(data_path)
    ndx_returns = loader.load_index_returns()
    print(f"  Loaded {len(ndx_returns)} daily returns")
    print(f"  Date range: {ndx_returns.index.min()} to {ndx_returns.index.max()}")
    print(f"  Mean return: {ndx_returns.mean():.4%}")
    print(f"  Std dev: {ndx_returns.std():.4%}")
    
    # Create configurations
    print_section("STRATEGY CONFIGURATIONS")
    
    cfg_hedged = StrategyConfig(
        hedged_weight=0.7,
        unhedged_weight=0.3,
        annual_hedge_cost=0.04
    )
    
    cfg_unhedged = StrategyConfig(
        hedged_weight=0.0,
        unhedged_weight=1.0,
        annual_hedge_cost=0.0
    )
    
    print("\nHedged Strategy:")
    print(f"  Hedged Weight:      {cfg_hedged.hedged_weight:.1%}")
    print(f"  Unhedged Weight:    {cfg_hedged.unhedged_weight:.1%}")
    print(f"  Annual Hedge Cost:  {cfg_hedged.annual_hedge_cost:.2%}")
    print(f"  Unhedged Leverage:  {cfg_hedged.unhedged_leverage:.1f}x")
    print(f"  Hedged Up Beta:     {cfg_hedged.hedged_up_beta:.2f}")
    print(f"  Hedged Down Beta:   {cfg_hedged.hedged_down_beta:.2f}")
    print(f"  Crash Floor:        {cfg_hedged.crash_floor:.2%}")
    
    print("\nUnhedged Strategy:")
    print(f"  Hedged Weight:      {cfg_unhedged.hedged_weight:.1%}")
    print(f"  Unhedged Weight:    {cfg_unhedged.unhedged_weight:.1%}")
    print(f"  Annual Hedge Cost:  {cfg_unhedged.annual_hedge_cost:.2%}")
    print(f"  Unhedged Leverage:  {cfg_unhedged.unhedged_leverage:.1f}x")
    
    # Run simulations
    print_section("RUNNING MONTE CARLO SIMULATIONS")
    
    print(f"\nHedged Strategy: Running {n_paths} paths...")
    mc_hedged = MonteCarloSimulator(ndx_returns, cfg_hedged)
    results_hedged = mc_hedged.run_simulation(n_paths, n_days)
    print(f"  ✓ Completed")
    
    print(f"\nUnhedged Strategy: Running {n_paths} paths...")
    mc_unhedged = MonteCarloSimulator(ndx_returns, cfg_unhedged)
    results_unhedged = mc_unhedged.run_simulation(n_paths, n_days)
    print(f"  ✓ Completed")
    
    # Compute stats
    print_section("COMPUTING STATISTICS")
    
    print("\nHedged strategy...")
    stats_hedged = mc_hedged.summary_stats(results_hedged)
    
    print("Unhedged strategy...")
    stats_unhedged = mc_unhedged.summary_stats(results_unhedged)
    
    # Display results
    print_section("RESULTS: HEDGED STRATEGY")
    print_stats("Hedged Strategy", stats_hedged)
    
    print_section("RESULTS: UNHEDGED STRATEGY")
    print_stats("Unhedged Strategy", stats_unhedged)
    
    # Comparative analysis
    print_section("COMPARATIVE ANALYSIS")
    
    # Count outcomes
    hedged_better_equity = (results_hedged['final_equity'] > results_unhedged['final_equity']).sum()
    hedged_better_drawdown = (results_hedged['max_drawdown'] > results_unhedged['max_drawdown']).sum()
    
    pct_hedged_better_equity = (hedged_better_equity / n_paths) * 100
    pct_hedged_better_drawdown = (hedged_better_drawdown / n_paths) * 100
    
    print(f"\nHedged > Unhedged (Final Equity):")
    print(f"  {hedged_better_equity}/{n_paths} paths ({pct_hedged_better_equity:.1f}%)")
    
    print(f"\nHedged > Unhedged (Max Drawdown - less negative is better):")
    print(f"  {hedged_better_drawdown}/{n_paths} paths ({pct_hedged_better_drawdown:.1f}%)")
    
    # Equity advantage
    equity_diff = stats_hedged['median_final_equity'] - stats_unhedged['median_final_equity']
    equity_diff_pct = (equity_diff / stats_unhedged['median_final_equity']) * 100
    
    print(f"\nMedian Final Equity Difference:")
    print(f"  Hedged - Unhedged: ${equity_diff:,.2f} ({equity_diff_pct:+.2f}%)")
    
    # Drawdown advantage
    dd_diff = stats_hedged['median_max_drawdown'] - stats_unhedged['median_max_drawdown']
    
    print(f"\nMedian Max Drawdown Difference (less negative = better):")
    print(f"  Hedged - Unhedged: {dd_diff:+.2%}")
    print(f"  Downside Protection: {abs(dd_diff):.2%}")
    
    # Efficiency ratio (return vs drawdown)
    hedged_efficiency = stats_hedged['median_total_return'] / abs(stats_hedged['median_max_drawdown']) if stats_hedged['median_max_drawdown'] != 0 else np.inf
    unhedged_efficiency = stats_unhedged['median_total_return'] / abs(stats_unhedged['median_max_drawdown']) if stats_unhedged['median_max_drawdown'] != 0 else np.inf
    
    print(f"\nReturn/Risk Efficiency (Median Return / |Max Drawdown|):")
    print(f"  Hedged:    {hedged_efficiency:.2f}x")
    print(f"  Unhedged:  {unhedged_efficiency:.2f}x")
    
    # Save results
    print_section("SAVING RESULTS")
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Save simulation results
    hedged_path = output_dir / "mc_results_hedged.csv"
    unhedged_path = output_dir / "mc_results_unhedged.csv"
    
    results_hedged.to_csv(hedged_path, index=False)
    results_unhedged.to_csv(unhedged_path, index=False)
    
    print(f"\nHedged results:   {hedged_path}")
    print(f"Unhedged results: {unhedged_path}")
    
    # Save summary
    summary_path = output_dir / "mc_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("MONTE CARLO ANALYSIS: HEDGED VS UNHEDGED\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Simulation Parameters:\n")
        f.write(f"  Paths: {n_paths}\n")
        f.write(f"  Days per path: {n_days}\n\n")
        
        f.write("HEDGED STRATEGY STATS:\n")
        f.write("-" * 70 + "\n")
        for key, val in stats_hedged.items():
            if 'equity' in key:
                f.write(f"  {key}: ${val:,.2f}\n")
            elif 'drawdown' in key:
                f.write(f"  {key}: {val:.2%}\n")
            else:
                f.write(f"  {key}: {val:.2%}\n")
        
        f.write("\nUNHEDGED STRATEGY STATS:\n")
        f.write("-" * 70 + "\n")
        for key, val in stats_unhedged.items():
            if 'equity' in key:
                f.write(f"  {key}: ${val:,.2f}\n")
            elif 'drawdown' in key:
                f.write(f"  {key}: {val:.2%}\n")
            else:
                f.write(f"  {key}: {val:.2%}\n")
        
        f.write("\nCOMPARATIVE ANALYSIS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Hedged > Unhedged (Equity): {hedged_better_equity}/{n_paths} ({pct_hedged_better_equity:.1f}%)\n")
        f.write(f"  Hedged > Unhedged (Drawdown): {hedged_better_drawdown}/{n_paths} ({pct_hedged_better_drawdown:.1f}%)\n")
        f.write(f"  Median Equity Difference: ${equity_diff:,.2f} ({equity_diff_pct:+.2f}%)\n")
        f.write(f"  Downside Protection: {abs(dd_diff):.2%}\n")
    
    print(f"Summary:          {summary_path}")
    
    print_section("ANALYSIS COMPLETE")
    print("\n")


if __name__ == "__main__":
    main()
