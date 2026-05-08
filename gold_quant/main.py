"""
Gold Quant Trading System - Main Orchestrator

This module provides the entry point for all system operations:
- backtest: Run walk-forward backtest
- paper: Live paper trading simulation
- train: Retrain models on latest data
- test: Integration smoke test
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import time
import json

import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('gold_quant.log')
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_backtest(config: dict) -> None:
    """Run full walk-forward backtest and generate report."""
    logger.info("Starting walk-forward backtest...")
    
    from backtest.engine import WalkForwardBacktester
    from analytics.reporter import ReportGenerator
    
    backtester = WalkForwardBacktester(config)
    results = backtester.run()
    
    # Generate report
    reporter = ReportGenerator(config)
    reporter.generate_html_report(results)
    
    logger.info("Backtest completed. Report saved to analytics/report.html")


def run_paper_trading(config: dict) -> None:
    """Run live paper trading loop."""
    logger.info("Starting paper trading mode...")
    
    from execution.paper_broker import PaperBroker
    from strategies.signal_generator import SignalGenerator
    from strategies.position_manager import PositionManager
    from risk.risk_manager import RiskManager
    from data.fetcher import DataFetcher
    from data.preprocessor import Preprocessor
    from features.engineer import FeatureEngineer
    from models.ensemble import EnsemblePredictor
    
    # Initialize components
    fetcher = DataFetcher(config)
    preprocessor = Preprocessor(config)
    engineer = FeatureEngineer(config)
    signal_gen = SignalGenerator(config)
    position_mgr = PositionManager(config)
    risk_mgr = RiskManager(config)
    broker = PaperBroker(config)
    ensemble = EnsemblePredictor(config)
    
    last_processed_date = None
    
    # Load last processed date if exists
    state_file = Path(config['paths']['execution']) / 'portfolio_state.json'
    if state_file.exists():
        with open(state_file, 'r') as f:
            state = json.load(f)
            last_processed_date = state.get('last_processed_date')
            logger.info(f"Resuming from last processed date: {last_processed_date}")
    
    while True:
        try:
            # Fetch latest data
            logger.info("Fetching latest market data...")
            raw_data = fetcher.fetch_all()
            
            # Check freshness
            if not fetcher.check_freshness(raw_data['XAUUSD=X']):
                logger.warning("Data may be stale. Waiting for next update...")
                time.sleep(3600)  # Wait 1 hour before retrying
                continue
            
            # Preprocess
            processed = preprocessor.process(raw_data)
            
            # Engineer features
            features_df, metadata = engineer.generate_features(processed)
            
            # Get latest date
            current_date = features_df.index[-1]
            
            if last_processed_date and current_date <= last_processed_date:
                logger.info(f"No new data since {last_processed_date}. Waiting...")
                time.sleep(3600)
                continue
            
            # Generate signals
            signal = signal_gen.generate_signal(features_df, ensemble)
            
            # Get portfolio state
            portfolio_state = broker.get_portfolio_state()
            
            # Manage position
            order = position_mgr.manage_position(signal, portfolio_state, features_df)
            
            # Risk check
            approved_order = risk_mgr.check_signal(order, portfolio_state, features_df)
            
            # Execute if approved
            if approved_order.size > 0:
                broker.execute_order(approved_order)
                logger.info(f"Executed order: {approved_order}")
            
            # Save state
            last_processed_date = current_date
            broker.save_state(last_processed_date)
            
            logger.info(f"Processing complete for {current_date}. Waiting for next day...")
            
            # Sleep until next day (check every hour)
            time.sleep(3600)
            
        except Exception as e:
            logger.error(f"Error in paper trading loop: {e}", exc_info=True)
            # Save partial state
            try:
                broker.save_state(last_processed_date)
            except:
                pass
            time.sleep(300)  # Wait 5 minutes before retrying


def run_training(config: dict) -> None:
    """Retrain all models on latest data."""
    logger.info("Starting model training...")
    
    from data.fetcher import DataFetcher
    from data.preprocessor import Preprocessor
    from features.engineer import FeatureEngineer
    from models.trainer import ModelTrainer
    
    # Fetch and prepare data
    fetcher = DataFetcher(config)
    raw_data = fetcher.fetch_all()
    
    preprocessor = Preprocessor(config)
    processed = preprocessor.process(raw_data)
    
    engineer = FeatureEngineer(config)
    features_df, metadata = engineer.generate_features(processed)
    
    # Train models
    trainer = ModelTrainer(config)
    trainer.train_all_models(features_df)
    
    logger.info("Model training completed. Artifacts saved to models/")


def run_test(config: dict) -> None:
    """Run integration smoke test with synthetic data."""
    logger.info("Running integration smoke test...")
    
    import numpy as np
    import pandas as pd
    from backtest.engine import WalkForwardBacktester
    from backtest.metrics import compute_metrics
    
    # Generate synthetic data (sine wave + noise) - use business days to match real trading data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='B')  # Business days only
    
    # Sine wave with trend and noise
    t = np.arange(len(dates))
    trend = 0.0001 * t
    seasonal = 0.02 * np.sin(2 * np.pi * t / 50)
    noise = 0.01 * np.random.randn(len(t))
    
    prices = 2000 * np.exp(np.cumsum(trend + seasonal + noise))
    
    # Create synthetic OHLCV
    synthetic_data = pd.DataFrame({
        'Open': prices * (1 + 0.001 * np.random.randn(len(t))),
        'High': prices * (1 + 0.01 * np.abs(np.random.randn(len(t)))),
        'Low': prices * (1 - 0.01 * np.abs(np.random.randn(len(t)))),
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, len(t))
    }, index=dates)
    
    # Add macro data (random walk)
    macro_data = {}
    for ticker in config['macro_tickers']:
        macro_prices = 100 * np.exp(np.cumsum(0.01 * np.random.randn(len(t))))
        macro_data[ticker] = pd.DataFrame({
            'Open': macro_prices,
            'High': macro_prices * 1.01,
            'Low': macro_prices * 0.99,
            'Close': macro_prices,
            'Volume': np.random.randint(1000, 10000, len(t))
        }, index=dates)
    
    macro_data['XAUUSD=X'] = synthetic_data
    
    logger.info(f"Generated synthetic data: {len(dates)} days")
    
    # Test feature engineering
    from features.engineer import FeatureEngineer
    engineer = FeatureEngineer(config)
    
    # Merge all data into proper format expected by feature engineer
    merged = synthetic_data.copy()
    merged = merged.rename(columns={
        'Open': 'XAUUSD_Open',
        'High': 'XAUUSD_High',
        'Low': 'XAUUSD_Low',
        'Close': 'XAUUSD_Close',
        'Volume': 'XAUUSD_Volume'
    })
    
    for ticker, df in macro_data.items():
        if ticker != 'XAUUSD=X':
            merged = merged.join(df['Close'].rename(f'{ticker}_Close'), how='left')
    
    merged = merged.ffill().bfill()
    
    features_df, metadata = engineer.generate_features((merged, {'volume_available': True}))
    
    # Drop warm-up rows using the proper method
    features_df = engineer.drop_warmup_rows(features_df, warmup_period=200)
    
    logger.info(f"Features generated: {features_df.shape[1]} columns, {features_df.shape[0]} rows")
    
    assert features_df.shape[1] > 20, "Feature count too low"
    assert features_df.shape[0] > 10, f"Row count too low after dropping NaNs: {features_df.shape[0]}"
    
    # Test model training
    from models.trainer import ModelTrainer
    trainer = ModelTrainer(config)
    trainer.train_all_models(features_df)
    
    logger.info("Models trained successfully")
    
    # Test backtest (short version)
    from backtest.engine import WalkForwardBacktester
    
    # Use shorter walk-forward for test
    test_config = config.copy()
    test_config['walkforward'] = {
        'train_months': 6,
        'test_months': 1,
        'step_months': 1
    }
    
    backtester = WalkForwardBacktester(test_config)
    results = backtester.run(synthetic_data=synthetic_data, macro_data=macro_data)
    
    logger.info(f"Backtest completed: {len(results['equity_curve'])} days")
    
    # Compute metrics
    metrics = compute_metrics(results['equity_curve'], results['benchmark'])
    
    logger.info(f"Strategy Sharpe: {metrics['strategy_sharpe']:.2f}")
    logger.info(f"Total trades: {metrics['num_trades']}")
    
    # Basic sanity checks
    assert metrics['num_trades'] > 0, "No trades executed"
    assert len(results['equity_curve']) > 0, "Empty equity curve"
    
    logger.info("✅ All integration tests passed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Gold Quant Trading System')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['backtest', 'paper', 'train', 'test'],
                       help='Operation mode')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    logger.info(f"Starting Gold Quant Trading System in {args.mode} mode")
    logger.info(f"Configuration loaded from {args.config}")
    
    try:
        if args.mode == 'backtest':
            run_backtest(config)
        elif args.mode == 'paper':
            run_paper_trading(config)
        elif args.mode == 'train':
            run_training(config)
        elif args.mode == 'test':
            run_test(config)
        
        logger.info(f"{args.mode.upper()} mode completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
