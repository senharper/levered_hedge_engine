"""
Real-Time NDX Portfolio Tracker

Tracks portfolio performance in real-time against NDX prices.
Loads previous state from CSV and updates incrementally.

Features:
  - Retry logic with sanity checks for price fetches
  - Weekend/holiday detection
  - Duplicate daily update prevention
  - Portfolio safety alerts
  - Excess return tracking
  - Optional dry-run mode
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import pandas as pd
from pathlib import Path
from datetime import datetime
import csv
import time
import argparse
from typing import Dict, Optional
import yfinance as yf

from config.strategy_config import StrategyConfig
from core.portfolio import Portfolio


# Define CSV columns in exact order matching existing CSV format
LOG_COLUMNS = [
    "date",
    "equity",
    "hedged_value",
    "unhedged_value",
    "hedged_weight",
    "unhedged_weight",
    "ndx_price",
    "ndx_return",
    "excess_return",
]


def get_latest_ndx_price() -> float:
    """
    Fetch the most recent NDX (Nasdaq-100) index price using yfinance.
    
    Includes retry logic (3 attempts with 2-second sleep between retries).
    
    Returns:
        Latest NDX closing price (float)
        
    Raises:
        RuntimeError: If all retry attempts fail or data validation fails
    """
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(1, max_retries + 1):
        try:
            # Fetch with auto_adjust=False to get actual prices
            data = yf.download(
                "^NDX",
                period="1d",
                interval="1d",
                progress=False,
                auto_adjust=False
            )
            
            if data.empty:
                raise RuntimeError("No data returned from yfinance")
            
            # Extract the close price (safe handling for both DataFrame and Series)
            close_series = data["Close"]
            latest_price = close_series.iloc[-1] if hasattr(close_series, 'iloc') else close_series
            latest_price = float(latest_price)
            
            if latest_price <= 0:
                raise RuntimeError("Invalid price: must be positive")
            
            return latest_price
        
        except Exception as e:
            if attempt < max_retries:
                print(f"  Attempt {attempt}/{max_retries} failed: {e}")
                print(f"  Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"Failed to fetch NDX price after {max_retries} attempts: {e}")


def load_previous_state(log_path: str) -> Dict:
    """
    Load the last logged state from CSV.
    
    Args:
        log_path: Path to ndx_realtime_log.csv
        
    Returns:
        Dictionary with keys:
            - last_date: str (date from previous row, YYYY-MM-DD)
            - last_ndx_price: float (NDX price from previous row)
            - equity: float (total portfolio value)
            - hedged_value: float (hedged sleeve value)
            - unhedged_value: float (unhedged sleeve value)
            - hedged_weight: float (hedged sleeve weight)
            - unhedged_weight: float (unhedged sleeve weight)
        Returns empty dict if file doesn't exist or is empty.
    """
    csv_path = Path(log_path)
    
    if not csv_path.exists():
        return {}
    
    try:
        df = pd.read_csv(csv_path)
        
        if df.empty:
            return {}
        
        last_row = df.iloc[-1]
        
        return {
            'last_date': str(last_row['date']),
            'last_ndx_price': float(last_row['ndx_price']),
            'equity': float(last_row['equity']),
            'hedged_value': float(last_row['hedged_value']),
            'unhedged_value': float(last_row['unhedged_value']),
            'hedged_weight': float(last_row['hedged_weight']),
            'unhedged_weight': float(last_row['unhedged_weight']),
        }
    except Exception as e:
        print(f"⚠️  Error reading CSV: {e}")
        return {}


def log_state_to_csv(log_path: str, record: Dict) -> None:
    """
    Append a record to the real-time log CSV.
    
    Auto-creates the file with LOG_COLUMNS header on first call.
    
    Args:
        log_path: Path to ndx_realtime_log.csv
        record: Dictionary with keys matching LOG_COLUMNS
    """
    csv_path = Path(log_path)
    file_exists = csv_path.exists()
    
    # Ensure directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build row with only columns in LOG_COLUMNS (preserves column order)
    row = {col: record.get(col, '') for col in LOG_COLUMNS}
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row)


def build_portfolio_from_previous(previous_state: Dict) -> Portfolio:
    """
    Reconstruct Portfolio instance with state restored from previous session.
    
    Args:
        previous_state: Dictionary returned from load_previous_state()
        
    Returns:
        Portfolio with internal state set to continue from previous values
    """
    config = StrategyConfig()
    portfolio = Portfolio(config)
    
    # Restore the sleeve values from previous session
    portfolio._current_hedged_value = previous_state['hedged_value']
    portfolio._current_unhedged_value = previous_state['unhedged_value']
    
    return portfolio


def main(log_path: str = "outputs/ndx_realtime_log.csv", dry_run: bool = False) -> None:
    """
    Main real-time update loop.
    
    1. Check for weekend/holiday
    2. Load previous state from CSV (if exists)
    3. Check if already updated today
    4. Get latest NDX price (with retry logic and sanity checks)
    5. Update portfolio using Portfolio.update_realtime()
    6. Calculate safety metrics and excess return
    7. Optionally append record to CSV (or just print in dry-run mode)
    8. Print summary
    
    Args:
        log_path: Path to CSV log file (default: outputs/ndx_realtime_log.csv)
        dry_run: If True, don't write to CSV, only print values
    """
    
    print("=" * 70)
    print("REAL-TIME NDX PORTFOLIO TRACKER")
    print("=" * 70)
    
    # ===== WEEKEND / HOLIDAY GUARD =====
    today_weekday = datetime.now().weekday()
    if today_weekday >= 5:  # 5=Saturday, 6=Sunday
        print(f"\n⏰ Weekend detected (weekday={today_weekday}) — skipping update.")
        return
    
    today_date_str = datetime.now().strftime('%Y-%m-%d')
    
    # Initialize config
    config = StrategyConfig()
    print(f"\nConfiguration:")
    print(f"  Initial Capital:    ${config.initial_capital:,.2f}")
    print(f"  Hedged Weight:      {config.hedged_weight:.1%}")
    print(f"  Unhedged Weight:    {config.unhedged_weight:.1%}")
    print(f"  Annual Hedge Cost:  {config.annual_hedge_cost:.2%}")
    
    # Load previous state or initialize fresh
    previous_state = load_previous_state(log_path)
    
    # ===== DUPLICATE DAILY UPDATE CHECK =====
    if previous_state and previous_state['last_date'] == today_date_str:
        print(f"\n✓ Already updated today ({today_date_str}) — skipping update.")
        return
    
    if previous_state:
        print(f"\nLoaded previous state from {log_path}:")
        print(f"  Last Date:         {previous_state['last_date']}")
        print(f"  Total Equity:      ${previous_state['equity']:,.2f}")
        print(f"  Hedged Value:      ${previous_state['hedged_value']:,.2f}")
        print(f"  Unhedged Value:    ${previous_state['unhedged_value']:,.2f}")
        print(f"  NDX Price:         ${previous_state['last_ndx_price']:,.2f}")
        
        portfolio = build_portfolio_from_previous(previous_state)
        ndx_price_yesterday = previous_state['last_ndx_price']
        equity_yesterday = previous_state['equity']
        current_date = today_date_str
    else:
        print(f"\nNo previous state found. Initializing fresh portfolio.")
        print(f"  Initial Equity:    ${config.initial_capital:,.2f}")
        
        portfolio = Portfolio(config)
        ndx_price_yesterday = 10000.0  # Placeholder for first run
        equity_yesterday = config.initial_capital
        current_date = today_date_str
    
    # ===== GET LATEST NDX PRICE WITH RETRY & SANITY CHECKS =====
    print(f"\nFetching latest NDX price...")
    try:
        ndx_price_today = get_latest_ndx_price()
    except RuntimeError as e:
        print(f"\n❌ Error fetching NDX price: {e}")
        print("   Not updating portfolio. Will retry on next run.")
        return
    
    # Price sanity check: ±10% deviation
    price_change_pct = (ndx_price_today / ndx_price_yesterday - 1) if ndx_price_yesterday > 0 else 0
    if abs(price_change_pct) > 0.10:
        print(f"\n⚠️  PRICE SANITY CHECK FAILED")
        print(f"   Price changed {price_change_pct:.2%} (threshold: ±10%)")
        print(f"   Yesterday: ${ndx_price_yesterday:,.2f}")
        print(f"   Today:     ${ndx_price_today:,.2f}")
        print(f"   NOT updating portfolio. Check for market data error.")
        return
    
    print(f"  Current NDX Price: ${ndx_price_today:,.2f}")
    print(f"  NDX Price Change:  {price_change_pct:.4%}")
    
    # ===== UPDATE PORTFOLIO =====
    print(f"\nUpdating portfolio...")
    update_record = portfolio.update_realtime(ndx_price_today, ndx_price_yesterday, current_date)
    
    # Add ndx_price to record (Portfolio.update_realtime doesn't include it)
    update_record['ndx_price'] = ndx_price_today
    
    # ===== CALCULATE EXCESS RETURN =====
    equity_today = update_record['equity']
    portfolio_daily_return = (equity_today / equity_yesterday - 1) if equity_yesterday > 0 else 0
    ndx_return = update_record['ndx_return']
    excess_return = portfolio_daily_return - ndx_return
    update_record['excess_return'] = excess_return
    
    print(f"  Total Equity:      ${equity_today:,.2f}")
    print(f"  Hedged Value:      ${update_record['hedged_value']:,.2f}")
    print(f"  Unhedged Value:    ${update_record['unhedged_value']:,.2f}")
    print(f"  Hedged Weight:     {update_record['hedged_weight']:.2%}")
    print(f"  Unhedged Weight:   {update_record['unhedged_weight']:.2%}")
    print(f"  NDX Return:        {ndx_return:.4%}")
    print(f"  Portfolio Return:  {portfolio_daily_return:.4%}")
    print(f"  Excess Return:     {excess_return:.4%}")
    
    # ===== PORTFOLIO SAFETY CHECKS =====
    if abs(portfolio_daily_return) > 0.05:
        print(f"\n🚨 ALERT: Daily return exceeds ±5%")
        print(f"   Portfolio Daily Return: {portfolio_daily_return:.4%}")
    
    if equity_today < config.initial_capital * 0.80:
        print(f"\n⚠️  WARNING: Portfolio equity below 80% of initial capital")
        print(f"   Current:  ${equity_today:,.2f}")
        print(f"   Threshold: ${config.initial_capital * 0.80:,.2f}")
    
    # ===== LOG TO CSV (or dry-run) =====
    if dry_run:
        print(f"\n[DRY-RUN MODE] Skipping CSV write. To write, run without --dry-run.")
    else:
        print(f"\nLogging to {log_path}...")
        log_state_to_csv(log_path, update_record)
        print(f"✓ Logged successfully")
    
    print("\n" + "=" * 70)
    print(f"PORTFOLIO EQUITY: ${equity_today:,.2f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time NDX portfolio tracker with hedge overlay"
    )
    parser.add_argument(
        "--log",
        type=str,
        default="outputs/ndx_realtime_log.csv",
        help="Path to CSV log file (default: outputs/ndx_realtime_log.csv)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print values without writing to CSV"
    )
    
    args = parser.parse_args()
    main(log_path=args.log, dry_run=args.dry_run)
