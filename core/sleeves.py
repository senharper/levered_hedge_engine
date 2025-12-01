"""
Sleeves Module

Defines the two portfolio sleeves: Hedged and Unhedged.
Each sleeve maps index returns to sleeve-specific returns based on its strategy.
"""

from dataclasses import dataclass
from config.strategy_config import StrategyConfig


@dataclass
class HedgedSleeve:
    """
    Hedged portfolio sleeve with asymmetric exposure and crash protection.
    
    Return mapping:
    - Up market (R > 0): hedged_up_beta × R - hedge_cost
    - Down market (0 ≥ R ≥ -30%): hedged_down_beta × R - hedge_cost
    - Crash (R < -30%): crash_floor
    """
    
    config: StrategyConfig

    def map_index_return(self, r_index: float) -> float:
        """
        Map an index return to the hedged sleeve return.
        
        Args:
            r_index: Index return for the period (as decimal)
            
        Returns:
            Hedged sleeve return for the period (as decimal)
        """
        c = self.config
        hc = c.period_hedge_cost

        # Up market: amplified exposure minus hedge cost
        if r_index > 0:
            return c.hedged_up_beta * r_index - hc
        
        # Down market (but not crash): reduced exposure minus hedge cost
        elif r_index >= -0.30:
            return c.hedged_down_beta * r_index - hc
        
        # Crash scenario: floor protection kicks in
        else:
            return c.crash_floor


@dataclass
class UnhedgedSleeve:
    """
    Unhedged portfolio sleeve with simple leveraged exposure.
    
    Return mapping:
    - All scenarios: unhedged_leverage × R
    """
    
    config: StrategyConfig

    def map_index_return(self, r_index: float) -> float:
        """
        Map an index return to the unhedged sleeve return.
        
        Args:
            r_index: Index return for the period (as decimal)
            
        Returns:
            Unhedged sleeve return for the period (as decimal)
        """
        # Simple leveraged exposure
        return self.config.unhedged_leverage * r_index
