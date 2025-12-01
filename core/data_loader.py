"""
Data Loader Module

Handles loading and preparing index return data from CSV files.
"""

import pandas as pd
from pathlib import Path
from typing import Union


class DataLoader:
    """
    Loads benchmark index returns from CSV files.
    
    The CSV file should contain columns:
    - date: Date in parseable format (e.g., YYYY-MM-DD)
    - return: Decimal return (e.g., 0.05 for 5%)
    """
    
    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize the DataLoader.
        
        Args:
            data_path: Path to the CSV file containing index returns
        """
        self.data_path = Path(data_path)
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
    
    def load_index_returns(self) -> pd.Series:
        """
        Load index returns from CSV file.
        
        Returns:
            pandas Series with date index and return values
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            KeyError: If required columns are missing
            ValueError: If data cannot be parsed
        """
        try:
            # Read CSV with date parsing
            df = pd.read_csv(self.data_path, parse_dates=["date"])
            
            # Validate required columns
            if "date" not in df.columns or "return" not in df.columns:
                raise KeyError("CSV must contain 'date' and 'return' columns")
            
            # Sort by date to ensure chronological order
            df = df.sort_values("date")
            
            # Remove any duplicate dates (keep first occurrence)
            df = df.drop_duplicates(subset=["date"], keep="first")
            
            # Create and return the Series with date index
            return df.set_index("date")["return"]
            
        except Exception as e:
            raise ValueError(f"Error loading data from {self.data_path}: {str(e)}")
    
    def get_data_summary(self) -> dict:
        """
        Get summary statistics of the loaded data.
        
        Returns:
            Dictionary with data summary statistics
        """
        returns = self.load_index_returns()
        
        return {
            "start_date": returns.index.min(),
            "end_date": returns.index.max(),
            "num_periods": len(returns),
            "mean_return": returns.mean(),
            "std_return": returns.std(),
            "min_return": returns.min(),
            "max_return": returns.max(),
        }
