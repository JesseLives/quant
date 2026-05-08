"""
Walk-Forward Backtester Module

Implements walk-forward validation:
- Train on 24 months, test on 1 month, step 1 month
- Retrains all models at each step
- Simulates execution with realistic fills
- Computes benchmark (buy-and-hold) comparison
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WalkForwardBacktester:
    """Walk-forward backtesting engine."""
    
    def __init__(self, config: dict):
        """
        Initialize the backtester.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.initial_capital = config['initial_capital']
        self.commission_per_unit = config['commission_per_unit']
        self.slippage_per_unit = config['slippage_per_unit']
        self.lot_size = config['lot_size']
        
        self.wf_config = config.get('walkforward', {
            'train_months': 24,
            'test_months': 1,
            'step_months': 1
        })
    
    def generate_walkforward_splits(self, dates: pd.DatetimeIndex) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        Generate walk-forward train/test splits.
        
        Args:
            dates: Full date range
            
        Returns:
            List of (train_dates, test_dates) tuples
        """
        train_months = self.wf_config['train_months']
        test_months = self.wf_config['test_months']
        step_months = self.wf_config['step_months']
        
        splits = []
        
        # Start from when we have enough data
        min_train_days = train_months * 21  # Approx trading days
        
        if len(dates) < min_train_days + test_months * 21:
            logger.warning("Insufficient data for walk-forward. Using simple split.")
            split_idx = int(len(dates) * 0.8)
            return [(dates[:split_idx], dates[split_idx:])]
        
        # Generate splits
        current_idx = min_train_days
        
        while current_idx + test_months * 21 <= len(dates):
            # Training period
            train_start_idx = max(0, current_idx - train_months * 21)
            train_dates = dates[train_start_idx:current_idx]
            
            # Test period
            test_end_idx = min(current_idx + test_months * 21, len(dates))
            test_dates = dates[current_idx:test_end_idx]
            
            if len(train_dates) >= min_train_days and len(test_dates) >= test_months * 15:
                splits.append((train_dates, test_dates))
            
            # Step forward
            current_idx += step_months * 21
        
        logger.info(f"Generated {len(splits)} walk-forward splits")
        
        return splits
    
    def run_backtest_period(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                           initial_capital: float) -> Dict:
        """
        Run backtest for a single test period.
        
        Args:
            train_df: Training data
            test_df: Test data
            initial_capital: Starting capital
            
        Returns:
            Backtest results dictionary
        """
        from features.engineer import FeatureEngineer
        from models.trainer import ModelTrainer
        from models.ensemble import EnsemblePredictor
        from strategies.signal_generator import SignalGenerator
        from strategies.position_manager import PositionManager, Order
        from risk.risk_manager import RiskManager, ApprovedOrder
        from execution.paper_broker import PaperBroker
        
        # Combine train and test for feature engineering (to avoid edge effects)
        combined_df = pd.concat([train_df, test_df])
        
        # Engineer features
        engineer = FeatureEngineer(self.config)
        metadata = {'volume_available': True}
        
        # We need OHLCV format for feature engineering
        # Assume train_df/test_df already have proper columns
        features_full, _ = engineer.generate_features((combined_df, metadata))
        
        # Drop warmup rows
        features_clean = engineer.drop_warmup_rows(features_full, warmup_period=200)
        
        # Split back into train and test
        test_dates = test_df.index
        features_test = features_clean.loc[test_dates.intersection(features_clean.index)]
        features_train = features_clean.loc[train_df.index.intersection(features_clean.index)]
        
        if len(features_train) < 100 or len(features_test) < 10:
            logger.warning("Insufficient data after feature engineering")
            return None
        
        # Train models
        trainer = ModelTrainer(self.config)
        save_date = datetime.now().strftime('%Y%m%d_%H%M%S')
        trainer.train_all_models(features_train, save_date=save_date)
        
        # Load ensemble
        ensemble = EnsemblePredictor(self.config, model_date=save_date)
        
        # Initialize components
        signal_gen = SignalGenerator(self.config)
        position_mgr = PositionManager(self.config)
        risk_mgr = RiskManager(self.config)
        broker = PaperBroker(self.config)
        
        # Backtest loop
        equity_curve = []
        daily_returns = []
        trade_log = []
        
        prev_equity = initial_capital
        
        for date in features_test.index:
            # Get features up to this date
            features_up_to = features_test.loc[:date]
            
            if len(features_up_to) < 2:
                continue
            
            # Generate signal
            try:
                signal = signal_gen.generate_signal(features_up_to, ensemble)
            except Exception as e:
                logger.debug(f"Error generating signal for {date}: {e}")
                continue
            
            # Get portfolio state
            portfolio_state = broker.get_portfolio_state()
            
            # Manage position
            order = position_mgr.manage_position(signal, portfolio_state, features_up_to)
            
            # Risk check
            approved_order = risk_mgr.check_signal(order, portfolio_state, features_up_to)
            
            # Execute order
            if approved_order.size > 0 or approved_order.side == 'EXIT':
                # Get bar data for execution
                bar_data = combined_df.loc[date] if date in combined_df.index else None
                
                exec_report = broker.execute_order(approved_order, bar_data=bar_data)
                
                if exec_report:
                    trade_log.append({
                        'date': date,
                        'type': exec_report.get('type'),
                        'details': exec_report
                    })
            
            # Mark to market
            if date in combined_df.index:
                close_price = combined_df.loc[date, 'XAUUSD_Close'] if 'XAUUSD_Close' in combined_df.columns else combined_df.loc[date, 'Close']
                broker.mark_to_market(close_price, str(date))
            
            # Record equity
            current_equity = broker.get_equity()
            equity_curve.append({
                'date': date,
                'equity': current_equity
            })
            
            # Daily return
            if prev_equity > 0:
                daily_ret = (current_equity - prev_equity) / prev_equity
                daily_returns.append({
                    'date': date,
                    'return': daily_ret
                })
            
            prev_equity = current_equity
        
        # Compile results
        results = {
            'equity_curve': pd.DataFrame(equity_curve).set_index('date'),
            'daily_returns': pd.DataFrame(daily_returns).set_index('date') if daily_returns else pd.DataFrame(),
            'trade_log': trade_log,
            'trade_history': broker.trade_history,
            'final_equity': broker.get_equity(),
            'total_return': (broker.get_equity() - initial_capital) / initial_capital
        }
        
        return results
    
    def compute_benchmark(self, prices: pd.Series, test_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Compute buy-and-hold benchmark returns.
        
        Args:
            prices: Price series
            test_dates: Test period dates
            
        Returns:
            Benchmark equity curve
        """
        # Filter to test period
        test_prices = prices.loc[test_dates.intersection(prices.index)]
        
        if len(test_prices) == 0:
            return pd.DataFrame()
        
        # Buy-and-hold from first price
        initial_price = test_prices.iloc[0]
        units = self.initial_capital / initial_price
        
        benchmark_equity = test_prices * units
        
        return pd.DataFrame({
            'date': test_prices.index,
            'equity': benchmark_equity.values
        }).set_index('date')
    
    def run(self, synthetic_data: pd.DataFrame = None, 
            macro_data: Dict = None) -> Dict:
        """
        Run full walk-forward backtest.
        
        Args:
            synthetic_data: Optional synthetic price data for testing
            macro_data: Optional macro data dictionary
            
        Returns:
            Complete backtest results
        """
        logger.info("Starting walk-forward backtest...")
        
        # Fetch or use provided data
        if synthetic_data is not None:
            logger.info("Using synthetic data")
            data = synthetic_data
        else:
            # Fetch real data
            from data.fetcher import DataFetcher
            from data.preprocessor import Preprocessor
            
            fetcher = DataFetcher(self.config)
            raw_data = fetcher.fetch_all()
            
            preprocessor = Preprocessor(self.config)
            processed, metadata = preprocessor.process(raw_data)
            
            # Extract XAUUSD OHLCV
            data = processed[['XAUUSD_Open', 'XAUUSD_High', 'XAUUSD_Low', 
                             'XAUUSD_Close', 'XAUUSD_Volume']].copy()
            data = data.rename(columns={
                'XAUUSD_Open': 'Open',
                'XAUUSD_High': 'High',
                'XAUUSD_Low': 'Low',
                'XAUUSD_Close': 'Close',
                'XAUUSD_Volume': 'Volume'
            })
        
        # Generate walk-forward splits
        dates = data.index
        splits = self.generate_walkforward_splits(dates)
        
        if len(splits) == 0:
            logger.error("No valid walk-forward splits generated")
            return None
        
        # Run backtest for each split
        all_equity = []
        all_benchmark = []
        all_trades = []
        
        for i, (train_dates, test_dates) in enumerate(splits):
            logger.info(f"Running split {i+1}/{len(splits)}")
            
            train_df = data.loc[train_dates]
            test_df = data.loc[test_dates]
            
            # Run backtest
            results = self.run_backtest_period(train_df, test_df, self.initial_capital)
            
            if results is not None:
                all_equity.append(results['equity_curve'])
                all_trades.extend(results['trade_history'])
                
                # Compute benchmark
                benchmark = self.compute_benchmark(data['Close'], test_dates)
                if len(benchmark) > 0:
                    all_benchmark.append(benchmark)
        
        # Combine results
        if len(all_equity) == 0:
            logger.error("No successful backtest periods")
            return None
        
        combined_equity = pd.concat(all_equity)
        combined_benchmark = pd.concat(all_benchmark) if all_benchmark else None
        
        final_results = {
            'equity_curve': combined_equity,
            'benchmark': combined_benchmark,
            'trade_history': all_trades,
            'num_trades': len(all_trades),
            'num_splits': len(splits),
            'final_equity': combined_equity['equity'].iloc[-1],
            'total_return': (combined_equity['equity'].iloc[-1] - self.initial_capital) / self.initial_capital
        }
        
        logger.info(f"Backtest complete: {len(all_trades)} trades, "
                   f"total return={final_results['total_return']:.2%}")
        
        return final_results
