"""
Portfolio Module

Combines the hedged and unhedged sleeves to simulate the complete portfolio.
Tracks value evolution and weight drift over time.
"""

from dataclasses import dataclass
from typing import Dict
import pandas as pd
from config.strategy_config import StrategyConfig
from .sleeves import HedgedSleeve, UnhedgedSleeve


@dataclass
class Portfolio:
    """
    Portfolio that combines hedged and unhedged sleeves.
    
    Simulates the evolution of portfolio values over time based on
    index returns and sleeve-specific return mappings.
    """
    
    config: StrategyConfig
    _current_hedged_value: float = None
    _current_unhedged_value: float = None

    def __post_init__(self):
        """Initialize the two portfolio sleeves and current state."""
        self.hedged_sleeve = HedgedSleeve(self.config)
        self.unhedged_sleeve = UnhedgedSleeve(self.config)
        
        # Initialize current state for real-time updates
        if self._current_hedged_value is None:
            self._current_hedged_value = self.config.initial_capital * self.config.hedged_weight
        if self._current_unhedged_value is None:
            self._current_unhedged_value = self.config.initial_capital * self.config.unhedged_weight

    def run_path(self, index_returns: pd.Series) -> pd.DataFrame:
        """
        Simulate the portfolio path given a series of index returns.
        
        This is a buy-and-hold simulation with no rebalancing.
        Sleeves drift from their initial weights over time.
        
        Args:
            index_returns: Series of index returns with date index
            
        Returns:
            DataFrame with columns:
                - index_value: Cumulative index value (starts at 1.0)
                - hedged_value: Value of hedged sleeve
                - unhedged_value: Value of unhedged sleeve
                - total_value: Total portfolio value
                - hedged_weight: Current weight of hedged sleeve
                - unhedged_weight: Current weight of unhedged sleeve
        """
        c = self.config

        # Initialize values
        index_value = 1.0
        hedged_value = c.initial_capital * c.hedged_weight
        unhedged_value = c.initial_capital * c.unhedged_weight

        records = []

        # Simulate each period
        for date, r_index in index_returns.items():
            # Update index value
            index_value *= (1 + r_index)

            # Map index return to sleeve returns
            r_hedged = self.hedged_sleeve.map_index_return(r_index)
            r_unhedged = self.unhedged_sleeve.map_index_return(r_index)

            # Update sleeve values
            hedged_value *= (1 + r_hedged)
            unhedged_value *= (1 + r_unhedged)
            total_value = hedged_value + unhedged_value

            # Calculate current weights (allowing drift from initial allocation)
            records.append({
                "date": date,
                "index_value": index_value,
                "hedged_value": hedged_value,
                "unhedged_value": unhedged_value,
                "total_value": total_value,
                "hedged_weight": hedged_value / total_value if total_value > 0 else 0,
                "unhedged_weight": unhedged_value / total_value if total_value > 0 else 0,
            })

        # Convert to DataFrame with date index
        return pd.DataFrame.from_records(records).set_index("date")
    
    def run_path_with_rebalancing(self, index_returns: pd.Series, 
                                   rebalance_frequency: int = 12) -> pd.DataFrame:
        """
        Simulate the portfolio path with periodic rebalancing.
        
        Args:
            index_returns: Series of index returns with date index
            rebalance_frequency: Rebalance every N periods (default: 12 = annually)
            
        Returns:
            DataFrame with the same structure as run_path()
        """
        c = self.config

        index_value = 1.0
        hedged_value = c.initial_capital * c.hedged_weight
        unhedged_value = c.initial_capital * c.unhedged_weight

        records = []
        period_counter = 0

        for date, r_index in index_returns.items():
            index_value *= (1 + r_index)

            r_hedged = self.hedged_sleeve.map_index_return(r_index)
            r_unhedged = self.unhedged_sleeve.map_index_return(r_index)

            hedged_value *= (1 + r_hedged)
            unhedged_value *= (1 + r_unhedged)
            total_value = hedged_value + unhedged_value

            # Rebalance if it's time
            period_counter += 1
            if period_counter % rebalance_frequency == 0:
                hedged_value = total_value * c.hedged_weight
                unhedged_value = total_value * c.unhedged_weight

            records.append({
                "date": date,
                "index_value": index_value,
                "hedged_value": hedged_value,
                "unhedged_value": unhedged_value,
                "total_value": total_value,
                "hedged_weight": hedged_value / total_value if total_value > 0 else 0,
                "unhedged_weight": unhedged_value / total_value if total_value > 0 else 0,
            })

        return pd.DataFrame.from_records(records).set_index("date")
    
    def update_realtime(self, ndx_price_today: float, ndx_price_yesterday: float, 
                       current_date) -> Dict:
        """
        Update portfolio state for a single day of real-time performance.
        
        Computes the NDX return and applies the same daily P&L logic as run_path.
        Maintains internal state (_current_hedged_value, _current_unhedged_value)
        for subsequent real-time calls.
        
        Args:
            ndx_price_today: NDX closing price today
            ndx_price_yesterday: NDX closing price yesterday
            current_date: Current date (for logging)
            
        Returns:
            Dictionary with:
                - date: Current date
                - equity_total: Total portfolio equity
                - equity_hedged: Hedged sleeve equity
                - equity_unhedged: Unhedged sleeve equity
                - ndx_return: NDX daily return
                - notional_hedged: Notional value (weight) of hedged portion
                - notional_unhedged: Notional value (weight) of unhedged portion
        """
        # Compute NDX return
        ndx_return = (ndx_price_today / ndx_price_yesterday) - 1
        
        # Map index return to sleeve returns
        r_hedged = self.hedged_sleeve.map_index_return(ndx_return)
        r_unhedged = self.unhedged_sleeve.map_index_return(ndx_return)
        
        # Update sleeve values
        self._current_hedged_value *= (1 + r_hedged)
        self._current_unhedged_value *= (1 + r_unhedged)
        total_value = self._current_hedged_value + self._current_unhedged_value
        
        # Calculate current weights
        hedged_weight = self._current_hedged_value / total_value if total_value > 0 else 0
        unhedged_weight = self._current_unhedged_value / total_value if total_value > 0 else 0
        
        return {
            'date': current_date,
            'equity': total_value,
            'hedged_value': self._current_hedged_value,
            'unhedged_value': self._current_unhedged_value,
            'ndx_return': ndx_return,
            'hedged_weight': hedged_weight,
            'unhedged_weight': unhedged_weight,
        }
