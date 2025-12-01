"""
Main Entry Point for Levered Hedge Engine

Runs a backtest of the leveraged hedged portfolio strategy.
"""

from pathlib import Path
from config.strategy_config import StrategyConfig
from core.backtester import Backtester


def main():
    """
    Main execution function.
    
    Loads data, runs portfolio simulation, calculates metrics,
    and saves results to multiple output formats.
    """
    # Configuration
    data_path = Path("data/ndx_returns_sample.csv")
    output_dir = Path("outputs")
    
    # Initialize and run backtester
    config = StrategyConfig()
    backtester = Backtester(config, data_path)
    
    print("Running backtest...\n")
    backtester.run()
    
    # Print summary to console
    backtester.print_summary()
    
    # Save all results
    print("Saving results...\n")
    backtester.save_results(output_dir)
    
    print("Backtest complete!")


if __name__ == "__main__":
    main()
