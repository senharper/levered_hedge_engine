"""
Main Entry Point for Levered Hedge Engine

Runs a backtest of the leveraged hedged portfolio strategy.
"""

from pathlib import Path
from config.strategy_config import StrategyConfig
from core.data_loader import DataLoader
from core.portfolio import Portfolio
from core.metrics import compute_cagr, compute_max_drawdown, compute_sharpe


def main():
    """
    Main execution function.
    
    Loads data, runs portfolio simulation, calculates metrics,
    and saves results.
    """
    # Configuration
    data_path = Path("data/ndx_returns_sample.csv")
    output_dir = Path("outputs")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading index return data...")
    loader = DataLoader(data_path)
    index_returns = loader.load_index_returns()
    
    print(f"Loaded {len(index_returns)} periods of data")
    print(f"Date range: {index_returns.index.min()} to {index_returns.index.max()}\n")
    
    # Initialize strategy
    config = StrategyConfig()
    portfolio = Portfolio(config)
    
    # Run simulation
    print("Running portfolio simulation...")
    df = portfolio.run_path(index_returns)
    
    # Calculate metrics
    print("Calculating performance metrics...\n")
    
    # Portfolio metrics
    cagr_port = compute_cagr(df["total_value"], config.periods_per_year)
    dd_port = compute_max_drawdown(df["total_value"])
    port_returns = df["total_value"].pct_change().dropna()
    sharpe_port = compute_sharpe(port_returns, config.periods_per_year)
    
    # Index metrics
    cagr_idx = compute_cagr(df["index_value"], config.periods_per_year)
    dd_idx = compute_max_drawdown(df["index_value"])
    idx_returns = df["index_value"].pct_change().dropna()
    sharpe_idx = compute_sharpe(idx_returns, config.periods_per_year)
    
    # Print summary
    print("="*60)
    print("BACKTEST SUMMARY")
    print("="*60)
    print("\n--- Portfolio Results ---")
    print(f"Final Portfolio Value:  ${df['total_value'].iloc[-1]:,.2f}")
    print(f"Final Index Value:      {df['index_value'].iloc[-1]:.4f}")
    print(f"\nPortfolio CAGR:         {cagr_port:.2%}")
    print(f"Index CAGR:             {cagr_idx:.2%}")
    print(f"Alpha (CAGR):           {cagr_port - cagr_idx:.2%}")
    print(f"\nPortfolio Max DD:       {dd_port:.2%}")
    print(f"Index Max DD:           {dd_idx:.2%}")
    print(f"\nPortfolio Sharpe:       {sharpe_port:.2f}")
    print(f"Index Sharpe:           {sharpe_idx:.2f}")
    print(f"Sharpe Advantage:       {sharpe_port - sharpe_idx:.2f}")
    
    print("\n--- Sleeve Analysis ---")
    print(f"Final Hedged Weight:    {df['hedged_weight'].iloc[-1]:.2%}")
    print(f"Final Unhedged Weight:  {df['unhedged_weight'].iloc[-1]:.2%}")
    print(f"(Initial: {config.hedged_weight:.0%} / {config.unhedged_weight:.0%})")
    
    print("\n" + "="*60)
    
    # Save results
    timeseries_path = output_dir / "portfolio_timeseries.csv"
    df.to_csv(timeseries_path)
    print(f"\nTime series saved to: {timeseries_path}")
    
    # Save summary to text file
    summary_path = output_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("LEVERED HEDGE ENGINE - BACKTEST SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write("Portfolio Results\n")
        f.write("-"*60 + "\n")
        f.write(f"Final Portfolio Value:  ${df['total_value'].iloc[-1]:,.2f}\n")
        f.write(f"Final Index Value:      {df['index_value'].iloc[-1]:.4f}\n\n")
        f.write(f"Portfolio CAGR:         {cagr_port:.2%}\n")
        f.write(f"Index CAGR:             {cagr_idx:.2%}\n")
        f.write(f"Alpha (CAGR):           {cagr_port - cagr_idx:.2%}\n\n")
        f.write(f"Portfolio Max DD:       {dd_port:.2%}\n")
        f.write(f"Index Max DD:           {dd_idx:.2%}\n\n")
        f.write(f"Portfolio Sharpe:       {sharpe_port:.2f}\n")
        f.write(f"Index Sharpe:           {sharpe_idx:.2f}\n")
        f.write(f"Sharpe Advantage:       {sharpe_port - sharpe_idx:.2f}\n\n")
        f.write("Sleeve Analysis\n")
        f.write("-"*60 + "\n")
        f.write(f"Final Hedged Weight:    {df['hedged_weight'].iloc[-1]:.2%}\n")
        f.write(f"Final Unhedged Weight:  {df['unhedged_weight'].iloc[-1]:.2%}\n")
        f.write(f"Initial Allocation:     {config.hedged_weight:.0%} / {config.unhedged_weight:.0%}\n")
    
    print(f"Summary saved to: {summary_path}\n")
    print("Backtest complete!")


if __name__ == "__main__":
    main()
