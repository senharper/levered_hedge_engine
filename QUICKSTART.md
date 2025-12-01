# Quick Start Guide

## Get Running in 3 Steps

### 1. Install Dependencies (Windows)

Open Command Prompt or PowerShell in the project directory:

```bash
pip install -r requirements.txt
```

### 2. Run the Backtest

```bash
python main.py
```

### 3. Check Your Results

Look in the `outputs/` folder:
- `portfolio_timeseries.csv` - Complete time series data
- `summary.txt` - Performance summary

## What You'll See

The backtest will print results showing:
- Portfolio final value vs. index
- CAGR (annualised return)
- Maximum drawdown
- Sharpe ratio
- How the sleeve weights drifted over time

## Next Steps

### Modify Strategy Parameters

Edit `config/strategy_config.py` to change:
- Initial capital amount
- Hedged/unhedged allocation
- Leverage levels
- Hedge cost
- Crash floor protection level

### Use Your Own Data

Replace `data/ndx_returns_sample.csv` with your data:
- Must have `date` and `return` columns
- Returns as decimals (0.05 = 5%)
- Monthly frequency recommended

### Explore the Code

Start with these files:
1. `main.py` - See the workflow
2. `core/sleeves.py` - Understand the strategy logic
3. `core/portfolio.py` - See how sleeves combine
4. `core/metrics.py` - Performance calculations

## Common Tasks

### Run with Rebalancing

Modify `main.py` to use the rebalancing feature:

```python
# Change this line in main.py:
df = portfolio.run_path(index_returns)

# To this:
df = portfolio.run_path_with_rebalancing(index_returns, rebalance_frequency=12)
```

### Use the Backtester Class

For more control, use the `Backtester` class:

```python
from pathlib import Path
from config.strategy_config import StrategyConfig
from core.backtester import Backtester

config = StrategyConfig()
backtester = Backtester(config, Path("data/ndx_returns_sample.csv"))
backtester.run()
backtester.print_summary()
backtester.save_results(Path("outputs"))
```

## Need Help?

- Check `README.md` for full documentation
- Review code comments in each module
- All classes have docstrings explaining their purpose

Happy backtesting!
