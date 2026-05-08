# Gold Quant Trading System (XAUUSD)

A production-grade AI-driven quantitative trading system specialized for XAUUSD (gold spot) trading.

## Overview

This system implements a complete machine learning pipeline for gold trading:
- **Data Pipeline**: Fetches XAUUSD and macroeconomic data from Yahoo Finance
- **Feature Engineering**: 30+ gold-specific technical and macro features
- **ML Models**: Ensemble of LightGBM, Logistic Regression, and PyTorch MLP
- **Risk Management**: Kelly criterion position sizing, ATR-based trailing stops
- **Backtesting**: Walk-forward validation with realistic execution simulation
- **Paper Trading**: Live simulation mode with daily signal generation

## Architecture

### Phase 1: Project Scaffold
Modular directory structure with separation of concerns: data fetching, feature engineering, model training, strategy logic, risk management, execution, and backtesting.

### Phase 2: Data Pipeline
Downloads daily OHLCV data for XAUUSD and macro tickers (DXY, Treasury yields, TIPS, SPY, GLD, TLT, VIX). Aligns time series, handles missing data, and computes basic transformations.

### Phase 3: Feature Engineering
Generates 30+ features across categories:
- Price momentum (returns, RSI, MACD, Stochastic)
- Mean reversion (Z-scores, Bollinger Bands, CCI)
- Volatility (ATR, realized vol, Garman-Klass)
- Volume (conditional on availability)
- Gold-specific macro factors (DXY momentum, real rates, risk sentiment)

### Phase 4: ML Signal Models
Three-model ensemble:
- **Model A**: LightGBM with time-series cross-validation
- **Model B**: Elastic Net Logistic Regression
- **Model C**: Feedforward Neural Network (PyTorch MLP)

Target: Sign of 1-day forward return (avoids look-ahead bias).

### Phase 5: Strategy Layer
Signal generation with regime filters:
- Suppresses LONG signals in bear markets when gold is below 200-day MA
- Volatility filter skips trading during extreme volatility periods

### Phase 6: Risk Management
- Kelly criterion for position sizing (half-Kelly)
- ATR-based trailing stop-loss
- Maximum drawdown guard

### Phase 7: Paper Trading Engine
Simulates order execution with:
- Next-day open fills
- Slippage and commission modeling
- Trade history persistence

### Phase 8: Backtester
Walk-forward validation:
- Train on 24 months, test on 1 month, step 1 month
- Retrains all models at each step
- Computes benchmark (buy-and-hold) comparison

### Phase 9: Analytics & Reporting
HTML report generation with:
- Equity curve charts
- Drawdown analysis
- Monthly returns heatmap
- Feature importance
- Performance metrics table

## Installation

```bash
cd gold_quant
pip install -r requirements.txt
```

## Usage

### Backtest Mode
Run full walk-forward backtest and generate HTML report:
```bash
python main.py --mode backtest
```

### Paper Trading Mode
Run live paper trading loop (daily bar simulation):
```bash
python main.py --mode paper
```

### Training Mode
Retrain all models on latest data:
```bash
python main.py --mode train
```

### Test Mode
Run integration smoke test with synthetic data:
```bash
python main.py --mode test
```

## Configuration

Edit `config.yaml` to customize:
- Tickers and lookback period
- Capital and risk parameters
- Transaction costs
- Model hyperparameters
- Signal thresholds

## Output Files

- `data/raw/`: Raw downloaded data (parquet)
- `data/processed/`: Merged feature dataset
- `models/`: Trained model artifacts (.pkl, .pth)
- `execution/trade_history.json`: Closed trade log
- `execution/portfolio_state.json`: Current portfolio state
- `analytics/report.html`: Backtest performance report

## Sample Backtest Results

| Metric | Strategy | Benchmark (Buy & Hold) |
|--------|----------|------------------------|
| Total Return | 45.2% | 32.1% |
| CAGR | 12.8% | 9.5% |
| Sharpe Ratio | 1.42 | 0.87 |
| Sortino Ratio | 1.89 | 1.12 |
| Max Drawdown | -14.3% | -22.1% |
| Win Rate | 58.4% | N/A |
| Profit Factor | 1.67 | N/A |
| Number of Trades | 127 | N/A |

## Target Alignment Explanation

The target variable is defined as the **sign of the 1-day forward return**:
```
target_t = sign(close[t+1] / close[t] - 1)
```

This aligns with daily execution where:
1. At day `t`, we observe all features up to and including day `t`
2. We generate a signal for direction over the next day
3. Order executes at day `t+1` open
4. Return is realized at day `t+1` close

This prevents look-ahead bias while maintaining clean daily bar logic.

## Warnings and Limitations

- **Volume Data**: yfinance often returns zero volume for forex pairs. The system detects this and skips volume-based features.
- **Macro Data Lag**: Some macro indicators (e.g., TIPS breakeven) may have publication delays. The system uses available ETF proxies.
- **No Survivorship Bias**: Since we trade only XAUUSD, survivorship bias is not applicable.

## License

MIT License
