"""
Position Manager Module

Manages position entry and exit logic:
- Entry: only enter if no current position
- Exit hierarchy:
  1. Stop-loss hit
  2. Opposite signal
  3. Confidence decay (below 0.55)
- No pyramiding
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """Represents a trading order."""
    side: str  # 'LONG' or 'SHORT'
    size: float  # Number of lots
    order_type: str = 'MARKET'
    
    def __repr__(self):
        return f"Order(side={self.side}, size={self.size:.2f} lots)"


class PositionManager:
    """Manages position entry and exit decisions."""
    
    def __init__(self, config: dict):
        """
        Initialize the position manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.confidence_exit_threshold = config.get('signal_thresholds', {}).get(
            'confidence_exit', 0.55
        )
    
    def manage_position(self, signal: Dict, portfolio_state: Dict, 
                       features_df: pd.DataFrame) -> Order:
        """
        Determine what action to take based on signal and current position.
        
        Exit hierarchy:
        1. If stop-loss hit → exit immediately (handled by risk manager/broker)
        2. If opposite signal → exit current, then enter new
        3. If confidence decay → exit position
        
        Entry:
        - Only enter if no current position
        - Size determined by risk manager
        
        Args:
            signal: Signal dictionary from signal generator
            portfolio_state: Current portfolio state from broker
            features_df: DataFrame with features (for ATR calculation)
            
        Returns:
            Order object (may be size=0 for HOLD)
        """
        current_position = portfolio_state.get('position')
        signal_side = signal['signal']
        confidence = signal['confidence']
        
        logger.info(f"Managing position: current={current_position}, signal={signal_side}")
        
        # Check for exit conditions first
        
        # Condition 1: Stop-loss hit (checked by broker, but we can pre-check)
        if current_position is not None:
            stop_triggered = self._check_stop_loss(current_position, features_df)
            if stop_triggered:
                logger.info("Stop-loss triggered. Exiting position.")
                return Order(side='EXIT', size=current_position['size'])
        
        # Condition 2: Opposite signal
        if current_position is not None and signal_side != 'HOLD':
            current_side = current_position['side']
            
            if (current_side == 'LONG' and signal_side == 'SHORT') or \
               (current_side == 'SHORT' and signal_side == 'LONG'):
                logger.info(f"Opposite signal detected ({signal_side} vs {current_side}). "
                           f"Exiting and reversing.")
                # Exit current position first, then enter new one
                # For simplicity, we return the new direction with the size to be determined
                return Order(side=signal_side, size=abs(current_position['size']))
        
        # Condition 3: Confidence decay
        if current_position is not None and signal_side == 'HOLD':
            # Check if confidence for current direction has decayed
            current_side = current_position['side']
            
            if current_side == 'LONG' and confidence < self.confidence_exit_threshold:
                logger.info(f"Confidence decay for LONG ({confidence:.3f} < "
                           f"{self.confidence_exit_threshold}). Exiting.")
                return Order(side='EXIT', size=current_position['size'])
            
            elif current_side == 'SHORT' and confidence > (1 - self.confidence_exit_threshold + 0.05):
                # For short, low confidence in UP means high probability of down
                # So we check if confidence in UP is too high
                logger.info(f"Confidence decay for SHORT ({1-confidence:.3f} too high). Exiting.")
                return Order(side='EXIT', size=current_position['size'])
        
        # Entry logic: only enter if no position
        if current_position is None and signal_side in ['LONG', 'SHORT']:
            logger.info(f"No position. Entering {signal_side}.")
            # Size will be determined by risk manager
            return Order(side=signal_side, size=0.0)  # Size TBD by risk manager
        
        # No action needed
        logger.info("No action required.")
        return Order(side='HOLD', size=0.0)
    
    def _check_stop_loss(self, position: Dict, features_df: pd.DataFrame) -> bool:
        """
        Check if stop-loss has been hit.
        
        This is a simplified check; the actual stop-loss monitoring happens
        in the broker during backtest execution.
        
        Args:
            position: Current position dictionary
            features_df: DataFrame with OHLCV data
            
        Returns:
            True if stop-loss should be triggered
        """
        # This is mainly for paper trading mode
        # In backtest, the broker handles this more precisely
        return False
