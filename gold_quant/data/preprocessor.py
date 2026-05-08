"""
Data Preprocessor Module

Aligns time series, computes basic transformations, handles volume availability,
winsorizes returns, and outputs merged DataFrame with metadata.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import mstats

logger = logging.getLogger(__name__)


class Preprocessor:
    """Preprocesses raw market data for feature engineering."""
    
    def __init__(self, config: dict):
        """
        Initialize the preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.primary_ticker = config['primary_ticker']
        self.macro_tickers = config['macro_tickers']
        self.processed_data_path = Path(config['paths']['data_processed'])
        
        # Ensure directory exists
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
    
    def check_volume_availability(self, df: pd.DataFrame) -> bool:
        """
        Check if volume data is available (non-zero).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            True if volume is available, False otherwise
        """
        if 'Volume' not in df.columns:
            return False
        
        # Check if all volumes are zero (common for forex pairs in yfinance)
        volume_sum = df['Volume'].sum()
        volume_available = volume_sum > 0
        
        if not volume_available:
            logger.info("Volume data unavailable (all zeros). Skipping volume features.")
        else:
            logger.info("Volume data available.")
        
        return volume_available
    
    def compute_gold_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute return-based features for XAUUSD.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with return columns added
        """
        df = df.copy()
        
        # Simple returns
        df['returns'] = df['Close'].pct_change()
        
        # Log returns
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Rolling volatility (20-day)
        df['volatility_20d'] = df['log_returns'].rolling(window=20, min_periods=20).std()
        
        # High-Low spread (intraday range)
        df['hl_spread'] = (df['High'] - df['Low']) / df['Close']
        
        return df
    
    def winsorize_returns(self, df: pd.DataFrame, limits: tuple = (0.01, 0.99)) -> pd.DataFrame:
        """
        Winsorize returns to remove extreme outliers.
        
        Args:
            df: DataFrame with returns column
            limits: Tuple of (lower, upper) percentiles
            
        Returns:
            DataFrame with winsorized returns
        """
        df = df.copy()
        
        if 'returns' in df.columns:
            # Winsorize at 1% and 99%
            df['returns_winsorized'] = mstats.winsorize(
                df['returns'].dropna(), 
                limits=limits
            )
            # Reindex to match original
            df['returns_winsorized'] = pd.Series(
                df['returns_winsorized'], 
                index=df['returns'].dropna().index
            ).reindex(df.index)
        
        return df
    
    def transform_macro_series(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Apply transformations to macro time series.
        
        Args:
            data: Dictionary mapping ticker symbols to DataFrames
            
        Returns:
            Dictionary with transformed DataFrames
        """
        transformed = {}
        
        for ticker, df in data.items():
            if ticker == self.primary_ticker:
                continue
            
            df = df.copy()
            
            # Compute daily returns
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            # Rolling volatility
            df['volatility_20d'] = df['log_returns'].rolling(window=20, min_periods=20).std()
            
            # Rolling momentum (20-day)
            df['momentum_20d'] = df['Close'].pct_change(periods=20)
            
            transformed[ticker] = df
        
        return transformed
    
    def align_time_series(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Align all time series to the same date index.
        
        Args:
            data: Dictionary mapping ticker symbols to DataFrames
            
        Returns:
            Merged DataFrame with aligned dates
        """
        logger.info("Aligning time series...")
        
        # Start with primary ticker
        primary_df = data[self.primary_ticker].copy()
        
        # Extract Close prices for macro tickers
        merged = primary_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        merged = merged.rename(columns={
            'Open': 'XAUUSD_Open',
            'High': 'XAUUSD_High',
            'Low': 'XAUUSD_Low',
            'Close': 'XAUUSD_Close',
            'Volume': 'XAUUSD_Volume'
        })
        
        for ticker, df in data.items():
            if ticker == self.primary_ticker:
                continue
            
            # Forward-fill gaps in individual series first
            df = df.ffill()
            
            # Add Close price with ticker prefix
            col_name = f'{ticker}_Close'
            merged[col_name] = df['Close']
            
            # Add returns if available
            if 'returns' in df.columns:
                merged[f'{ticker}_returns'] = df['returns']
        
        # Forward-fill missing values across all series
        merged = merged.ffill()
        
        # Backfill any remaining NaNs at the start
        merged = merged.bfill()
        
        # Drop rows that are still NaN
        initial_rows = len(merged)
        merged = merged.dropna(how='all')
        dropped = initial_rows - len(merged)
        
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows with all NaN values")
        
        logger.info(f"Aligned dataset: {len(merged)} rows, {len(merged.columns)} columns")
        
        return merged
    
    def process(self, raw_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Main preprocessing pipeline.
        
        Args:
            raw_data: Dictionary mapping ticker symbols to raw DataFrames
            
        Returns:
            Tuple of (processed DataFrame, metadata dict)
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Check volume availability for XAUUSD
        volume_available = self.check_volume_availability(
            raw_data[self.primary_ticker]
        )
        
        # Compute gold returns
        gold_df = self.compute_gold_returns(raw_data[self.primary_ticker])
        gold_df = self.winsorize_returns(gold_df)
        
        # Update raw_data with processed gold data
        raw_data[self.primary_ticker] = gold_df
        
        # Transform macro series
        transformed_macros = self.transform_macro_series(raw_data)
        
        # Combine all data
        all_data = {
            self.primary_ticker: gold_df,
            **transformed_macros
        }
        
        # Align time series
        merged_df = self.align_time_series(all_data)
        
        # Create metadata
        metadata = {
            'volume_available': volume_available,
            'date_range': {
                'start': str(merged_df.index[0].date()) if hasattr(merged_df.index[0], 'date') else str(merged_df.index[0]),
                'end': str(merged_df.index[-1].date()) if hasattr(merged_df.index[-1], 'date') else str(merged_df.index[-1])
            },
            'total_rows': len(merged_df),
            'total_columns': len(merged_df.columns),
            'tickers': list(raw_data.keys())
        }
        
        logger.info(f"Preprocessing complete. Metadata: {metadata}")
        
        return merged_df, metadata
    
    def save_processed_data(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> None:
        """
        Save processed data to parquet.
        
        Args:
            df: Processed DataFrame
            metadata: Metadata dictionary
        """
        # Save DataFrame
        filepath = self.processed_data_path / 'processed_data.parquet'
        df.reset_index().to_parquet(filepath, index=False)
        logger.info(f"Saved processed data to {filepath}")
        
        # Save metadata
        import json
        metadata_path = self.processed_data_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata to {metadata_path}")
