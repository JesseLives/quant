"""
Data Fetcher Module

Downloads daily OHLCV data for XAUUSD and macro tickers from Yahoo Finance.
Handles data alignment, freshness checks, and saves to parquet format.
"""

import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches market data from Yahoo Finance."""
    
    def __init__(self, config: dict):
        """
        Initialize the data fetcher.
        
        Args:
            config: Configuration dictionary with ticker symbols and paths
        """
        self.config = config
        self.primary_ticker = config['primary_ticker']
        self.macro_tickers = config['macro_tickers']
        self.lookback_days = config['lookback_days']
        self.raw_data_path = Path(config['paths']['data_raw'])
        
        # Ensure directory exists
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
    
    def fetch_ticker(self, ticker: str, period: str = "max") -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a single ticker.
        
        Args:
            ticker: Yahoo Finance ticker symbol
            period: Data period ('max', '1y', '2y', etc.)
            
        Returns:
            DataFrame with OHLCV data or None if fetch fails
        """
        logger.info(f"Fetching data for {ticker}...")
        
        try:
            # Download data
            df = yf.download(ticker, period=period, progress=False)
            
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return None
            
            # Handle multi-level columns if present (yfinance v0.23+)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Keep only relevant columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_cols = [c for c in required_cols if c in df.columns]
            
            if len(available_cols) < 3:
                logger.warning(f"Insufficient columns for {ticker}: {available_cols}")
                return None
            
            df = df[available_cols]
            
            # Forward fill missing values within the series
            df = df.ffill()
            
            logger.info(f"Fetched {len(df)} rows for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return None
    
    def fetch_all(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all tickers (primary + macro).
        
        Returns:
            Dictionary mapping ticker symbols to DataFrames
        """
        logger.info("Fetching all market data...")
        
        data = {}
        
        # Fetch primary ticker (XAUUSD)
        primary_df = self.fetch_ticker(self.primary_ticker)
        if primary_df is not None:
            data[self.primary_ticker] = primary_df
        else:
            raise ValueError(f"Failed to fetch primary ticker {self.primary_ticker}")
        
        # Fetch macro tickers
        for ticker in self.macro_tickers:
            df = self.fetch_ticker(ticker)
            if df is not None:
                data[ticker] = df
            else:
                logger.warning(f"Skipping {ticker} due to fetch failure")
        
        logger.info(f"Successfully fetched {len(data)} tickers")
        return data
    
    def check_freshness(self, df: pd.DataFrame, max_age_days: int = 2) -> bool:
        """
        Check if data is fresh (within max_age_days of current date).
        
        Args:
            df: DataFrame with DateTimeIndex
            max_age_days: Maximum allowed age in days
            
        Returns:
            True if data is fresh, False otherwise
        """
        if df.empty:
            return False
        
        last_date = df.index[-1]
        current_date = datetime.now()
        
        # Convert to timezone-naive if needed
        if hasattr(last_date, 'tz') and last_date.tz is not None:
            last_date = last_date.tz_localize(None)
        
        age = (current_date - last_date).days
        
        is_fresh = age <= max_age_days
        
        if not is_fresh:
            logger.warning(
                f"Data freshness check failed: last date={last_date.date()}, "
                f"age={age} days, max_allowed={max_age_days} days"
            )
        else:
            logger.info(f"Data freshness OK: last date={last_date.date()}, age={age} days")
        
        return is_fresh
    
    def save_to_parquet(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Save fetched data to parquet files.
        
        Args:
            data: Dictionary mapping ticker symbols to DataFrames
        """
        logger.info("Saving data to parquet files...")
        
        for ticker, df in data.items():
            # Sanitize ticker name for filename
            safe_name = ticker.replace('=', '_').replace('-', '_')
            filepath = self.raw_data_path / f"{safe_name}.parquet"
            
            # Reset index to store date as column
            df_to_save = df.reset_index()
            df_to_save.to_parquet(filepath, index=False)
            
            logger.info(f"Saved {ticker} to {filepath}")
    
    def load_from_parquet(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load data from parquet file.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            DataFrame with OHLCV data or None if file doesn't exist
        """
        safe_name = ticker.replace('=', '_').replace('-', '_')
        filepath = self.raw_data_path / f"{safe_name}.parquet"
        
        if not filepath.exists():
            logger.warning(f"Parquet file not found: {filepath}")
            return None
        
        df = pd.read_parquet(filepath)
        
        # Set date as index
        if 'Date' in df.columns:
            df = df.set_index('Date')
        elif 'date' in df.columns:
            df = df.set_index('date')
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        return df
    
    def log_data_quality(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Log data quality statistics.
        
        Args:
            data: Dictionary mapping ticker symbols to DataFrames
        """
        logger.info("=== Data Quality Report ===")
        
        for ticker, df in data.items():
            total_rows = len(df)
            missing_pct = df.isnull().sum().sum() / (total_rows * len(df.columns)) * 100
            
            if hasattr(df.index[0], 'date'):
                start_date = df.index[0].date()
                end_date = df.index[-1].date()
            else:
                start_date = df.index[0]
                end_date = df.index[-1]
            
            logger.info(
                f"{ticker}: {total_rows} rows, "
                f"date range={start_date} to {end_date}, "
                f"missing={missing_pct:.2f}%"
            )
        
        logger.info("===========================")
