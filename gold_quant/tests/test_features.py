"""
Unit tests for feature engineering module.

Tests cover:
- Feature shape consistency
- No NaN in feature columns after dropping warm-up rows
- Look-ahead check function catching leakage
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.engineer import FeatureEngineer


def create_sample_data(n_rows: int = 500) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range(start='2020-01-01', periods=n_rows, freq='D')
    
    # Generate realistic price series
    returns = np.random.randn(n_rows) * 0.01
    prices = 2000 * np.exp(np.cumsum(returns))
    
    # Create OHLCV
    df = pd.DataFrame({
        'XAUUSD_Open': prices * (1 + np.random.randn(n_rows) * 0.001),
        'XAUUSD_High': prices * (1 + np.abs(np.random.randn(n_rows) * 0.01)),
        'XAUUSD_Low': prices * (1 - np.abs(np.random.randn(n_rows) * 0.01)),
        'XAUUSD_Close': prices,
        'XAUUSD_Volume': np.random.randint(1000, 10000, n_rows),
    }, index=dates)
    
    # Add macro data
    for ticker in ['DX-Y.NYB', '^TNX', 'TIP', 'SPY', 'GLD', 'TLT', '^VIX']:
        macro_prices = 100 * np.exp(np.cumsum(np.random.randn(n_rows) * 0.01))
        df[f'{ticker}_Close'] = macro_prices
    
    return df


class TestFeatureShapeConsistency:
    """Test that feature generation produces consistent shapes."""
    
    def test_feature_count(self):
        """Test that we generate at least 30 features."""
        config = {
            'primary_ticker': 'XAUUSD=X',
            'macro_tickers': ['DX-Y.NYB', '^TNX', 'TIP', 'SPY', 'GLD', 'TLT', '^VIX'],
            'paths': {'data_raw': 'data/raw', 'data_processed': 'data/processed'}
        }
        
        engineer = FeatureEngineer(config)
        sample_data = create_sample_data(500)
        
        # Add metadata
        metadata = {'volume_available': True}
        
        features_df, _ = engineer.generate_features((sample_data, metadata))
        
        # Count features (excluding target)
        feature_cols = [c for c in features_df.columns if c != 'target']
        
        assert len(feature_cols) >= 30, f"Expected at least 30 features, got {len(feature_cols)}"
    
    def test_feature_columns_present(self):
        """Test that key feature categories are present."""
        config = {
            'primary_ticker': 'XAUUSD=X',
            'macro_tickers': ['DX-Y.NYB', '^TNX', 'TIP', 'SPY', 'GLD', 'TLT', '^VIX'],
            'paths': {'data_raw': 'data/raw', 'data_processed': 'data/processed'}
        }
        
        engineer = FeatureEngineer(config)
        sample_data = create_sample_data(500)
        metadata = {'volume_available': True}
        
        features_df, meta = engineer.generate_features((sample_data, metadata))
        
        # Check for key features from each category
        expected_features = [
            'return_5d',  # Momentum
            'rsi_14',  # Momentum
            'zscore_20',  # Mean reversion
            'atr_14',  # Volatility
            'volume_flag',  # Volume
        ]
        
        for feat in expected_features:
            assert feat in features_df.columns, f"Missing feature: {feat}"


class TestNoNaNAfterWarmup:
    """Test that there are no NaN values after dropping warm-up rows."""
    
    def test_no_nan_after_drop(self):
        """Test that dropping warm-up rows removes all NaN values."""
        config = {
            'primary_ticker': 'XAUUSD=X',
            'macro_tickers': ['DX-Y.NYB', '^TNX', 'TIP', 'SPY', 'GLD', 'TLT', '^VIX'],
            'paths': {'data_raw': 'data/raw', 'data_processed': 'data/processed'}
        }
        
        engineer = FeatureEngineer(config)
        sample_data = create_sample_data(600)
        metadata = {'volume_available': True}
        
        features_df, _ = engineer.generate_features((sample_data, metadata))
        
        # Drop warm-up rows
        clean_df = engineer.drop_warmup_rows(features_df, warmup_period=252)
        
        # Check for NaN in features (not target)
        feature_cols = [c for c in clean_df.columns if c != 'target']
        
        nan_count = clean_df[feature_cols].isnull().sum().sum()
        
        assert nan_count == 0, f"Found {nan_count} NaN values after warmup drop"
    
    def test_sufficient_rows_remaining(self):
        """Test that enough rows remain after dropping warm-up."""
        config = {
            'primary_ticker': 'XAUUSD=X',
            'macro_tickers': ['DX-Y.NYB', '^TNX', 'TIP', 'SPY', 'GLD', 'TLT', '^VIX'],
            'paths': {'data_raw': 'data/raw', 'data_processed': 'data/processed'}
        }
        
        engineer = FeatureEngineer(config)
        sample_data = create_sample_data(600)
        metadata = {'volume_available': True}
        
        features_df, _ = engineer.generate_features((sample_data, metadata))
        clean_df = engineer.drop_warmup_rows(features_df, warmup_period=252)
        
        assert len(clean_df) > 100, f"Only {len(clean_df)} rows remaining after warmup drop"


class TestLookAheadCheck:
    """Test the look-ahead bias detection function."""
    
    def test_no_leakage_clean_data(self):
        """Test that clean data passes the look-ahead check."""
        config = {
            'primary_ticker': 'XAUUSD=X',
            'macro_tickers': ['DX-Y.NYB', '^TNX', 'TIP', 'SPY', 'GLD', 'TLT', '^VIX'],
            'paths': {'data_raw': 'data/raw', 'data_processed': 'data/processed'}
        }
        
        engineer = FeatureEngineer(config)
        sample_data = create_sample_data(600)
        metadata = {'volume_available': True}
        
        features_df, _ = engineer.generate_features((sample_data, metadata))
        clean_df = engineer.drop_warmup_rows(features_df, warmup_period=252)
        
        # Should pass (no leakage)
        passed = engineer.look_ahead_check(clean_df, max_correlation=0.05)
        
        assert passed, "Clean data should pass look-ahead check"
    
    def test_leakage_detection(self):
        """Test that the function detects leaked features."""
        config = {
            'primary_ticker': 'XAUUSD=X',
            'macro_tickers': ['DX-Y.NYB', '^TNX', 'TIP', 'SPY', 'GLD', 'TLT', '^VIX'],
            'paths': {'data_raw': 'data/raw', 'data_processed': 'data/processed'}
        }
        
        engineer = FeatureEngineer(config)
        sample_data = create_sample_data(600)
        metadata = {'volume_available': True}
        
        features_df, _ = engineer.generate_features((sample_data, metadata))
        clean_df = engineer.drop_warmup_rows(features_df, warmup_period=252)
        
        # Introduce leakage: add a feature that is the shifted target
        clean_df['leaked_feature'] = clean_df['target'].shift(-1)
        
        # Should fail (leakage detected)
        passed = engineer.look_ahead_check(clean_df, max_correlation=0.05)
        
        assert not passed, "Data with leakage should fail look-ahead check"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
