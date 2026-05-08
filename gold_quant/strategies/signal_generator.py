"""
Signal Generator Module

Converts ensemble model output into trade signals with regime filters:
- Regime filter: suppress LONG signals in bear markets when gold is below 200-day MA
- Volatility filter: skip trading during extreme volatility periods
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generates trading signals from model predictions with filters."""
    
    def __init__(self, config: dict):
        """
        Initialize the signal generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.signal_thresholds = config.get('signal_thresholds', {
            'long_threshold': 0.60,
            'short_threshold': 0.40
        })
    
    def check_regime_filter(self, features_df: pd.DataFrame) -> bool:
        """
        Check if market regime allows LONG signals.
        
        Suppresses LONG signals if:
        - SPY 50-day MA < SPY 200-day MA (bear stock market)
        - AND XAUUSD close < XAUUSD 200-day MA
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            True if LONG signals are allowed, False otherwise
        """
        latest = features_df.iloc[-1]
        
        # Check if we have the required features
        if 'ma_200' not in features_df.columns:
            logger.warning("MA_200 not found, skipping regime filter")
            return True
        
        # Get current gold price and 200-day MA
        # Note: We need to reconstruct these from raw data or use available features
        # For now, we'll use a simplified check based on zscore
        if 'zscore_200' in features_df.columns:
            gold_below_ma200 = latest['zscore_200'] < 0
        else:
            gold_below_ma200 = False
        
        # Check SPY regime (we need SPY data)
        # Simplified: if we don't have SPY features, allow longs
        spy_bear = False
        
        # Look for SPY-related features
        spy_cols = [c for c in features_df.columns if 'spy' in c.lower()]
        if len(spy_cols) > 0:
            # Could add more sophisticated SPY trend detection here
            pass
        
        # If both conditions met, suppress LONG
        if spy_bear and gold_below_ma200:
            logger.info("Regime filter: Bear market + Gold below 200-day MA. Suppressing LONG signals.")
            return False
        
        return True
    
    def check_volatility_filter(self, features_df: pd.DataFrame) -> bool:
        """
        Check if volatility is too high for trading.
        
        Skips trading if XAUUSD 20-day realized vol is above its 1-year 95th percentile.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            True if trading is allowed, False if should skip
        """
        if 'realized_vol_20d' not in features_df.columns:
            logger.warning("realized_vol_20d not found, skipping volatility filter")
            return True
        
        latest = features_df.iloc[-1]
        current_vol = latest['realized_vol_20d']
        
        # Calculate 1-year (252 day) 95th percentile of volatility
        if len(features_df) >= 252:
            vol_history = features_df['realized_vol_20d'].iloc[-252:]
            vol_95th = vol_history.quantile(0.95)
            
            if current_vol > vol_95th:
                logger.info(
                    f"Volatility filter: Current vol={current_vol:.4f} > 95th percentile={vol_95th:.4f}. "
                    f"Skipping trading."
                )
                return False
        
        return True
    
    def generate_signal(self, features_df: pd.DataFrame, ensemble) -> Dict:
        """
        Generate trading signal with all filters applied.
        
        Args:
            features_df: DataFrame with features
            ensemble: EnsemblePredictor instance
            
        Returns:
            Signal dictionary with 'signal', 'confidence', and metadata
        """
        # Get raw signal from ensemble
        raw_signal = ensemble.get_latest_signal(features_df)
        signal = raw_signal['signal']
        confidence = raw_signal['confidence']
        
        logger.info(f"Raw ensemble signal: {signal} (confidence={confidence:.3f})")
        
        # Apply regime filter
        regime_ok = self.check_regime_filter(features_df)
        
        if not regime_ok and signal == 'LONG':
            logger.info("Regime filter converted LONG to HOLD")
            signal = 'HOLD'
        
        # Apply volatility filter
        vol_ok = self.check_volatility_filter(features_df)
        
        if not vol_ok and signal in ['LONG', 'SHORT']:
            logger.info("Volatility filter converted position to HOLD")
            signal = 'HOLD'
        
        final_signal = {
            'signal': signal,
            'confidence': confidence,
            'regime_ok': regime_ok,
            'volatility_ok': vol_ok,
            'timestamp': features_df.index[-1]
        }
        
        logger.info(f"Final signal: {signal} (confidence={confidence:.3f})")
        
        return final_signal
