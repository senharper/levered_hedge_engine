"""
Strategy Configuration Module

Defines all configurable parameters for the Levered Hedge Engine strategy.
"""

from dataclasses import dataclass


@dataclass
class StrategyConfig:
    """
    Configuration parameters for the leveraged hedged portfolio strategy.
    
    Attributes:
        initial_capital: Starting portfolio value in dollars
        hedged_weight: Allocation percentage to hedged sleeve (0.0 to 1.0)
        unhedged_weight: Allocation percentage to unhedged sleeve (0.0 to 1.0)
        unhedged_leverage: Leverage multiplier for unhedged sleeve
        hedged_up_beta: Beta multiplier for hedged sleeve in up markets
        hedged_down_beta: Beta multiplier for hedged sleeve in down markets
        crash_floor: Maximum loss in severe crash scenario (as negative decimal)
        annual_hedge_cost: Annual cost of hedging as decimal (e.g., 0.03 = 3%)
        periods_per_year: Number of return periods per year (12 for monthly)
    """
    initial_capital: float = 100_000.0
    hedged_weight: float = 0.7
    unhedged_weight: float = 0.3
    unhedged_leverage: float = 1.3
    hedged_up_beta: float = 1.3
    hedged_down_beta: float = 0.9
    crash_floor: float = -0.30
    annual_hedge_cost: float = 0.03
    periods_per_year: int = 12

    @property
    def period_hedge_cost(self) -> float:
        """
        Calculate the hedge cost per period.
        
        Returns:
            Decimal hedge cost for a single period
        """
        return self.annual_hedge_cost / self.periods_per_year
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.hedged_weight + self.unhedged_weight == 1.0, \
            "Hedged and unhedged weights must sum to 1.0"
        assert 0 <= self.hedged_weight <= 1.0, \
            "Hedged weight must be between 0 and 1"
        assert 0 <= self.unhedged_weight <= 1.0, \
            "Unhedged weight must be between 0 and 1"
        assert self.initial_capital > 0, \
            "Initial capital must be positive"
