# Levered Hedge Engine

A modular Python backtesting framework for simulating a leveraged portfolio strategy with asymmetric hedging.

## Overview

The Levered Hedge Engine combines two portfolio "sleeves":

1. **Hedged Sleeve** (70% allocation by default)
   - Provides asymmetric exposure: higher beta in up markets, lower beta in down markets
   - Includes a crash floor protection at -30% index drawdown
   - Pays an ongoing hedge cost

2. **Unhedged Sleeve** (30% allocation by default)
   - Simple leveraged exposure to the index
   - No downside protection

## Strategy Logic

### Hedged Sleeve Returns
- **If index return > 0%**: Return = 1.3 × index_return - hedge_cost
- **If -30% ≤ index return ≤ 0%**: Return = 0.9 × index_return - hedge_cost
- **If index return < -30%**: Return = crash_floor (-30%)

### Unhedged Sleeve Returns
- **All scenarios**: Return = 1.3 × index_return

## Project Structure

```
levered_hedge_engine/
├─ README.md                      # This file
├─ requirements.txt               # Python dependencies
├─ main.py                        # Entry point - runs backtest
├─ config/
│  └─ strategy_config.py         # Strategy parameters
├─ core/
│  ├─ data_loader.py             # Loads index return data
│  ├─ sleeves.py                 # Hedged and Unhedged sleeve logic
│  ├─ portfolio.py               # Combines sleeves, runs simulation
│  ├─ metrics.py                 # Performance metrics (CAGR, Sharpe, Max DD)
│  └─ backtester.py              # Orchestrates the backtest workflow
├─ data/
│  └─ ndx_returns_sample.csv     # Sample monthly index returns
└─ outputs/
   ├─ portfolio_timeseries.csv   # Generated time series data
   └─ summary.txt                # Performance summary
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository

2. Navigate to the project directory:
   ```bash
   cd levered_hedge_engine
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Backtest

Simply run the main script:

```bash
python main.py
```

This will:
1. Load index returns from `data/ndx_returns_sample.csv`
2. Run the portfolio simulation
3. Calculate performance metrics
4. Save results to `outputs/portfolio_timeseries.csv`
5. Print a summary to the console

### Customising Strategy Parameters

Edit `config/strategy_config.py` to modify:

- `initial_capital`: Starting portfolio value (default: $100,000)
- `hedged_weight`: Allocation to hedged sleeve (default: 0.7)
- `unhedged_weight`: Allocation to unhedged sleeve (default: 0.3)
- `unhedged_leverage`: Leverage multiplier for unhedged sleeve (default: 1.3)
- `hedged_up_beta`: Beta in up markets for hedged sleeve (default: 1.3)
- `hedged_down_beta`: Beta in down markets for hedged sleeve (default: 0.9)
- `crash_floor`: Maximum loss in crash scenario (default: -0.30)
- `annual_hedge_cost`: Annual cost of hedging (default: 0.03 or 3%)
- `periods_per_year`: Number of periods per year (default: 12 for monthly)

### Using Your Own Data

Replace `data/ndx_returns_sample.csv` with your own data file. The CSV should have two columns:
- `date`: Date in YYYY-MM-DD format
- `return`: Decimal return (e.g., 0.05 for 5%)

Example:
```csv
date,return
2020-01-31,0.0245
2020-02-29,-0.0812
2020-03-31,-0.1275
```

## Performance Metrics

The engine calculates:

- **CAGR (Compound Annual Growth Rate)**: Annualised return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return metric

Metrics are calculated for both the portfolio and the benchmark index.

## Architecture

The codebase follows a building-block methodology:

- **Separation of Concerns**: Each class has a single, well-defined responsibility
- **Modularity**: Components can be tested, modified, or replaced independently
- **Extensibility**: Easy to add new sleeves, metrics, or strategy logic
- **Type Safety**: Uses Python type hints throughout
- **Clarity**: Readable code with descriptive names and clear logic flow

## Example Output

```
=== Summary ===
Portfolio value: 245678.45
Index value: 187234.12
Portfolio CAGR: 15.32%
Index CAGR: 11.87%
Portfolio Max DD: -23.45%
Index Max DD: -35.67%
Portfolio Sharpe: 1.23
Index Sharpe: 0.89
```

## License

This project is provided as-is for educational and research purposes.

## Author

Built following modular best practices for quantitative portfolio analysis.
