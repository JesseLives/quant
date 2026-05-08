"""
Paper Broker Module

Simulates XAUUSD order execution with:
- Next-day open fills
- Slippage and commission modeling
- Trade history persistence
- Portfolio state management
"""

import logging
from typing import Dict, List, Optional
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PaperBroker:
    """Simulates paper trading execution."""
    
    def __init__(self, config: dict):
        """
        Initialize the paper broker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.initial_capital = config['initial_capital']
        self.commission_per_unit = config['commission_per_unit']
        self.slippage_per_unit = config['slippage_per_unit']
        self.lot_size = config['lot_size']
        
        self.execution_path = Path(config['paths']['execution'])
        self.execution_path.mkdir(parents=True, exist_ok=True)
        
        # Portfolio state
        self.cash = self.initial_capital
        self.position = None  # {'side': 'LONG'/'SHORT', 'size': lots, 'entry_price': price}
        self.peak_equity = self.initial_capital
        self.trade_history = []
        self.equity_curve = []
        
        # Load existing state if available
        self._load_state()
    
    def _load_state(self) -> None:
        """Load portfolio state from disk if exists."""
        state_file = self.execution_path / 'portfolio_state.json'
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                self.cash = state.get('cash', self.initial_capital)
                self.position = state.get('position')
                self.peak_equity = state.get('peak_equity', self.initial_capital)
                self.trade_history = state.get('trade_history', [])
                
                logger.info(f"Loaded portfolio state: cash={self.cash:.2f}, "
                           f"position={self.position}")
            except Exception as e:
                logger.warning(f"Could not load portfolio state: {e}")
    
    def get_portfolio_state(self) -> Dict:
        """
        Get current portfolio state.
        
        Returns:
            Dictionary with portfolio information
        """
        return {
            'cash': self.cash,
            'position': self.position,
            'peak_equity': self.peak_equity,
            'equity': self.get_equity(),
            'trade_history': self.trade_history[-50:],  # Last 50 trades
            'num_trades': len(self.trade_history)
        }
    
    def get_equity(self, current_price: float = None) -> float:
        """
        Calculate current equity including unrealized P&L.
        
        Args:
            current_price: Current market price (optional for mark-to-market)
            
        Returns:
            Total equity
        """
        equity = self.cash
        
        if self.position is not None:
            if current_price is None:
                # Use entry price (no unrealized P&L)
                current_price = self.position['entry_price']
            
            # Calculate unrealized P&L
            size_oz = self.position['size'] * self.lot_size
            
            if self.position['side'] == 'LONG':
                unrealized_pnl = (current_price - self.position['entry_price']) * size_oz
            else:  # SHORT
                unrealized_pnl = (self.position['entry_price'] - current_price) * size_oz
            
            equity += unrealized_pnl
        
        # Update peak equity
        self.peak_equity = max(self.peak_equity, equity)
        
        return equity
    
    def execute_order(self, order, fill_price: float = None, 
                     bar_data: Dict = None) -> Optional[Dict]:
        """
        Execute an order.
        
        Orders fill at next day's open with slippage.
        
        Args:
            order: ApprovedOrder to execute
            fill_price: Price to fill at (if None, uses bar_data)
            bar_data: OHLCV data for the bar
            
        Returns:
            Execution report or None if no execution
        """
        if order.side == 'HOLD':
            return None
        
        logger.info(f"Executing order: {order}")
        
        # Determine fill price
        if fill_price is None and bar_data is not None:
            base_price = bar_data.get('Open', 2000)
        elif fill_price is None:
            base_price = 2000  # Default
        else:
            base_price = fill_price
        
        # Apply slippage
        if order.side in ['LONG', 'EXIT_SHORT']:
            fill_price = base_price * (1 + self.slippage_per_unit)
        else:  # SHORT, EXIT_LONG
            fill_price = base_price * (1 - self.slippage_per_unit)
        
        # Handle EXIT orders
        if order.side == 'EXIT':
            if self.position is None:
                logger.warning("No position to exit")
                return None
            
            # Close current position
            exit_report = self._close_position(fill_price, order.side)
            return exit_report
        
        # Handle position reversal (EXIT then ENTER)
        if self.position is not None:
            # Close existing position first
            self._close_position(fill_price, 'EXIT')
        
        # Open new position
        if order.side in ['LONG', 'SHORT']:
            return self._open_position(order.side, order.size, fill_price, order.stop_loss)
        
        return None
    
    def _open_position(self, side: str, size: float, price: float, 
                      stop_loss: float = None) -> Dict:
        """
        Open a new position.
        
        Args:
            side: 'LONG' or 'SHORT'
            size: Number of lots
            price: Fill price
            stop_loss: Stop-loss level
            
        Returns:
            Execution report
        """
        # Calculate notional and commission
        size_oz = size * self.lot_size
        notional = size_oz * price
        commission = notional * self.commission_per_unit
        
        # Deduct commission from cash
        self.cash -= commission
        
        # Create position
        self.position = {
            'side': side,
            'size': size,
            'entry_price': price,
            'stop_loss': stop_loss,
            'entry_date': datetime.now().strftime('%Y-%m-%d'),
            'highest_high': price,  # For trailing stop
            'lowest_low': price     # For trailing stop
        }
        
        logger.info(f"Opened {side} position: {size:.2f} lots @ {price:.2f}, "
                   f"commission={commission:.2f}")
        
        return {
            'type': 'OPEN',
            'side': side,
            'size': size,
            'price': price,
            'commission': commission,
            'stop_loss': stop_loss
        }
    
    def _close_position(self, price: float, exit_type: str = 'EXIT') -> Dict:
        """
        Close current position.
        
        Args:
            price: Exit price
            exit_type: Type of exit ('EXIT', 'STOP_LOSS', etc.)
            
        Returns:
            Execution report
        """
        if self.position is None:
            return None
        
        pos = self.position
        size_oz = pos['size'] * self.lot_size
        
        # Calculate P&L
        if pos['side'] == 'LONG':
            pnl = (price - pos['entry_price']) * size_oz
        else:  # SHORT
            pnl = (pos['entry_price'] - price) * size_oz
        
        # Calculate commission
        notional = size_oz * price
        commission = notional * self.commission_per_unit
        
        # Net P&L after commission
        net_pnl = pnl - commission
        
        # Add to cash
        self.cash += net_pnl
        
        # Record trade
        trade = {
            'entry_date': pos['entry_date'],
            'exit_date': datetime.now().strftime('%Y-%m-%d'),
            'side': pos['side'],
            'size': pos['size'],
            'entry_price': pos['entry_price'],
            'exit_price': price,
            'pnl': net_pnl,
            'commission': commission,
            'exit_type': exit_type
        }
        self.trade_history.append(trade)
        
        logger.info(f"Closed {pos['side']} position: P&L={net_pnl:.2f}, "
                   f"total trades={len(self.trade_history)}")
        
        # Clear position
        self.position = None
        
        return {
            'type': 'CLOSE',
            'pnl': net_pnl,
            'exit_price': price,
            'commission': commission
        }
    
    def update_trailing_stop(self, bar_data: Dict) -> Optional[Dict]:
        """
        Update trailing stop and check if hit.
        
        Args:
            bar_data: Current bar OHLCV data
            
        Returns:
            Exit report if stop hit, None otherwise
        """
        if self.position is None:
            return None
        
        pos = self.position
        high = bar_data.get('High', pos['entry_price'])
        low = bar_data.get('Low', pos['entry_price'])
        
        # Update highest high / lowest low since entry
        if pos['side'] == 'LONG':
            self.position['highest_high'] = max(pos['highest_high'], high)
            stop_level = self.position.get('stop_loss', pos['entry_price'] - 40)
            
            # Check if stop hit (low <= stop)
            if low <= stop_level:
                logger.info(f"Stop-loss hit for LONG @ {stop_level:.2f}")
                return self._close_position(stop_level, 'STOP_LOSS')
        else:  # SHORT
            self.position['lowest_low'] = min(pos['lowest_low'], low)
            stop_level = self.position.get('stop_loss', pos['entry_price'] + 40)
            
            # Check if stop hit (high >= stop)
            if high >= stop_level:
                logger.info(f"Stop-loss hit for SHORT @ {stop_level:.2f}")
                return self._close_position(stop_level, 'STOP_LOSS')
        
        return None
    
    def mark_to_market(self, close_price: float, date: str) -> None:
        """
        Mark portfolio to market and record equity.
        
        Args:
            close_price: End-of-day close price
            date: Date string
        """
        equity = self.get_equity(close_price)
        
        self.equity_curve.append({
            'date': date,
            'equity': equity,
            'cash': self.cash,
            'position_value': equity - self.cash if self.position else 0
        })
    
    def save_state(self, last_processed_date: str = None) -> None:
        """
        Save portfolio state to disk.
        
        Args:
            last_processed_date: Last processed date
        """
        state = {
            'cash': self.cash,
            'position': self.position,
            'peak_equity': self.peak_equity,
            'trade_history': self.trade_history,
            'equity_curve': self.equity_curve[-100:],  # Last 100 days
            'last_processed_date': last_processed_date
        }
        
        state_file = self.execution_path / 'portfolio_state.json'
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        # Also save full trade history
        history_file = self.execution_path / 'trade_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.trade_history, f, indent=2, default=str)
        
        logger.info(f"Saved portfolio state to {state_file}")
