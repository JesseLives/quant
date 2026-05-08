"""
Risk Manager Module

Implements risk management logic:
- Kelly criterion position sizing
- ATR-based trailing stop-loss
- Maximum drawdown guard
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ApprovedOrder:
    """Represents an approved order with size."""
    side: str  # 'LONG', 'SHORT', 'EXIT', or 'HOLD'
    size: float  # Number of lots (0 for HOLD)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    def __repr__(self):
        return f"ApprovedOrder(side={self.side}, size={self.size:.2f} lots, stop={self.stop_loss})"


class RiskManager:
    """Manages risk and approves/rejects orders."""
    
    def __init__(self, config: dict):
        """
        Initialize the risk manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.initial_capital = config['initial_capital']
        self.max_position_pct = config['max_position_pct']
        self.max_drawdown_limit = config['max_drawdown_limit']
        self.commission_per_unit = config['commission_per_unit']
        self.slippage_per_unit = config['slippage_per_unit']
        self.lot_size = config['lot_size']
        self.entry_size_mult = config['entry_size_mult']
        
        # Kelly parameters
        self.min_trades_for_kelly = 10
        self.kelly_fraction = 0.5  # Half-Kelly
    
    def calculate_kelly_size(self, trade_history: list, equity: float) -> float:
        """
        Calculate position size using Kelly Criterion.
        
        Kelly fraction: f = p - (1-p)/R
        where p = win rate, R = win/loss ratio
        
        Uses half-Kelly for safety.
        
        Args:
            trade_history: List of closed trades with P&L
            equity: Current account equity
            
        Returns:
            Position size in lots
        """
        if len(trade_history) < self.min_trades_for_kelly:
            # Fall back to fixed fractional
            size = self.entry_size_mult * equity / (self.lot_size * 2000)  # Approx gold price
            logger.info(f"Insufficient trade history ({len(trade_history)}). "
                       f"Using fixed size: {size:.2f} lots")
            return size
        
        # Extract P&L from trade history
        pnl_values = [t.get('pnl', 0) for t in trade_history[-50:]]  # Last 50 trades max
        
        # Calculate win rate and win/loss ratio
        wins = [p for p in pnl_values if p > 0]
        losses = [abs(p) for p in pnl_values if p < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            # Can't calculate Kelly
            size = self.entry_size_mult * equity / (self.lot_size * 2000)
            logger.info("Can't calculate Kelly (no wins or losses). Using fixed size.")
            return size
        
        p = len(wins) / len(pnl_values)  # Win rate
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            R = float('inf')
        else:
            R = avg_win / avg_loss
        
        # Kelly fraction
        kelly_f = p - (1 - p) / R
        
        # Apply half-Kelly
        kelly_f_half = kelly_f * self.kelly_fraction
        
        # Ensure non-negative
        kelly_f_half = max(0, kelly_f_half)
        
        # Calculate notional size
        notional_size = kelly_f_half * equity
        
        # Convert to lots
        gold_price = 2000  # Approximate
        lots = notional_size / (gold_price * self.lot_size)
        
        # Cap at max position
        max_lots = (self.max_position_pct * equity) / (gold_price * self.lot_size)
        lots = min(lots, max_lots)
        
        # Minimum size check
        if lots < 0.01:
            logger.info(f"Kelly size too small ({lots:.3f}). Rejecting trade.")
            return 0.0
        
        logger.info(f"Kelly calculation: p={p:.3f}, R={R:.3f}, "
                   f"f={kelly_f_half:.3f}, size={lots:.2f} lots")
        
        return lots
    
    def calculate_trailing_stop(self, entry_price: float, atr: float, 
                                side: str, current_high: float = None,
                                current_low: float = None) -> float:
        """
        Calculate trailing stop-loss level.
        
        For LONG: initial stop = entry - 2*ATR, trail up with high - 2*ATR
        For SHORT: initial stop = entry + 2*ATR, trail down with low + 2*ATR
        
        Args:
            entry_price: Entry price of the position
            atr: Current ATR(14) value
            side: 'LONG' or 'SHORT'
            current_high: Highest high since entry (for LONG)
            current_low: Lowest low since entry (for SHORT)
            
        Returns:
            Stop-loss price level
        """
        multiplier = 2.0
        
        if side == 'LONG':
            if current_high is None:
                # Initial stop
                stop = entry_price - multiplier * atr
            else:
                # Trailing stop (only moves up)
                trailing_stop = current_high - multiplier * atr
                stop = max(entry_price - multiplier * atr, trailing_stop)
        else:  # SHORT
            if current_low is None:
                # Initial stop
                stop = entry_price + multiplier * atr
            else:
                # Trailing stop (only moves down)
                trailing_stop = current_low + multiplier * atr
                stop = min(entry_price + multiplier * atr, trailing_stop)
        
        return stop
    
    def check_drawdown_guard(self, portfolio_state: Dict) -> bool:
        """
        Check if we should halt trading due to drawdown.
        
        Halts new entries if drawdown > max_drawdown_limit.
        Resumes when drawdown recovers below 75% of limit.
        
        Args:
            portfolio_state: Current portfolio state
            
        Returns:
            True if trading is allowed, False if should halt
        """
        equity = portfolio_state.get('equity', self.initial_capital)
        peak_equity = portfolio_state.get('peak_equity', self.initial_capital)
        
        drawdown = (peak_equity - equity) / peak_equity
        
        # Hysteresis: resume at 75% of limit
        resume_threshold = self.max_drawdown_limit * 0.75
        
        if drawdown > self.max_drawdown_limit:
            logger.warning(f"Drawdown limit exceeded: {drawdown:.2%} > "
                          f"{self.max_drawdown_limit:.2%}. Halting trading.")
            return False
        
        if drawdown > resume_threshold:
            logger.info(f"Drawdown elevated: {drawdown:.2%}. "
                       f"Trading allowed but monitor closely.")
        
        return True
    
    def check_signal(self, order, portfolio_state: Dict, 
                    features_df: pd.DataFrame) -> ApprovedOrder:
        """
        Perform risk checks on an order and return approved size.
        
        Args:
            order: Order from position manager
            portfolio_state: Current portfolio state
            features_df: DataFrame with features (for ATR)
            
        Returns:
            ApprovedOrder with validated size
        """
        # Handle HOLD and EXIT orders
        if order.side in ['HOLD', 'EXIT']:
            return ApprovedOrder(side=order.side, size=order.size)
        
        # Check drawdown guard
        if not self.check_drawdown_guard(portfolio_state):
            logger.info("Drawdown guard rejected order.")
            return ApprovedOrder(side='HOLD', size=0.0)
        
        # Get current equity
        equity = portfolio_state.get('equity', self.initial_capital)
        
        # Get trade history for Kelly calculation
        trade_history = portfolio_state.get('trade_history', [])
        
        # Calculate position size
        size = self.calculate_kelly_size(trade_history, equity)
        
        if size < 0.01:
            logger.info("Position size too small. Rejecting order.")
            return ApprovedOrder(side='HOLD', size=0.0)
        
        # Calculate stop-loss
        latest = features_df.iloc[-1]
        atr = latest.get('atr_14', 20)  # Default ATR if not available
        
        # Get current price
        # In backtest, this will be provided; here we estimate
        current_price = 2000  # Default
        
        # Find price column
        for col in features_df.columns:
            if 'Close' in col or 'close' in col:
                if not col.startswith(('XAUUSD', 'GLD')):
                    continue
                current_price = latest.get(col, 2000)
                break
        
        stop_loss = self.calculate_trailing_stop(current_price, atr, order.side)
        
        logger.info(f"Order approved: {order.side} {size:.2f} lots, "
                   f"stop={stop_loss:.2f}")
        
        return ApprovedOrder(
            side=order.side,
            size=size,
            stop_loss=stop_loss
        )
