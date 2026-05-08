## ROLE

You are an expert quantitative developer and financial engineer. Your task is to autonomously design and fully implement a production‑grade AI quant trading system **specialized exclusively for XAUUSD (gold spot)**. Work iteratively through the night, planning, building, testing, and refining every component without waiting for human input between phases.

## MISSION

Build a complete, runnable AI‑driven trading system in Python that trades only XAUUSD. Treat this as a real engineering project: make deliberate architectural decisions, write clean modular code, add inline comments, and validate each phase before proceeding. By the end, every component must be wired together and functional.

## CONSTRAINTS

- Language: Python 3.10+
- No paid data APIs required. Primary data source: **yfinance** (`XAUUSD=X` for spot gold, plus freely available macro symbols like `DX-Y.NYB`, `^TNX`, `TIP`, `SPY`, `TLT`, `GLD`, `^VIX`).
- No live broker connection required — paper trading simulation is sufficient (both long and short gold positions).
- All ML models must be trainable locally (scikit‑learn, XGBoost, LightGBM, PyTorch; fall back to sklearn MLP if GPU unavailable). Retraining must be computationally feasible in a walk‑forward loop; use full retraining only for tree/linear models, and a fast MLP (not LSTM) that can be retrained quickly.
- Code must be fully modular: one file per component, one orchestrator entry point.
- After each phase, print a brief status summary and list what was completed.

## PHASES — execute in order, do not skip

─── PHASE 1 · Project Scaffold ───────────────────────────────
Create the full project directory structure and initialise essential files:

gold_quant/
├── data/          # raw + processed XAUUSD and macro data
├── features/      # feature engineering pipeline
├── models/        # trained model artifacts
├── strategies/    # signal generation logic
├── risk/          # position sizing, stop‑loss, exposure limits
├── execution/     # order router + paper trading engine (long & short gold)
├── backtest/      # walk‑forward backtesting framework
├── analytics/     # performance metrics + reporting
├── tests/         # unit & integration tests
├── config.yaml    # central configuration file (all paths use pathlib)
├── main.py        # orchestrator entry point
├── requirements.txt
└── README.md      # setup & usage

Write config.yaml with sensible defaults:

- primary_ticker: "XAUUSD=X"
- macro_tickers: ["DX-Y.NYB","^TNX","TIP","SPY","GLD","TLT","^VIX"]
- lookback_days: 1260  # 5 years (ensures enough data for rolling windows)
- train_test_split: 0.8
- initial_capital: 100000
- max_position_pct: 0.25
- max_drawdown_limit: 0.20
- commission_per_unit: 0.0005  # 0.05% of notional per trade (spread + commission)
- slippage_per_unit: 0.0003   # additional 0.03% slippage
- lot_size: 100               # 1 standard lot = 100 oz (for notional calculation)
- entry_size_mult: 0.1        # initial order size = 0.1 lots per signal (scales with Kelly)

─── PHASE 2 · Data Pipeline ──────────────────────────────────
Build data/fetcher.py:
- Download daily OHLCV data for `XAUUSD=X` from yfinance.
- Download data for all `macro_tickers`.
- Align all time series to the same date index; forward‑fill gaps.
- **Freshness check**: After fetching, verify that the last date in the XAUUSD data is within 2 calendar days of the current system date. If not, log a warning.
- Save to data/raw/ as parquet files.
- Log data quality stats (missing %, date range, last available date).

Build data/preprocessor.py:
- For XAUUSD: compute adjusted returns (pct_change), log returns, rolling volatility (20d), high-low spread.
- Volume column: `yfinance` often returns zero volume for forex. Check if all volume values are zero; set a global flag `volume_available = False` (persist this flag in metadata).
- For macro series: align and forward‑fill; compute transformations (e.g., 10Y real yield = ^TNX – TIP breakeven proxy).
- Winsorize gold returns at 1%/99%.
- Output a single merged DataFrame (all features) and a metadata dict with `volume_available`.

─── PHASE 3 · Feature Engineering ────────────────────────────
Build features/engineer.py with at least **30 gold‑specific features**, all forward‑fill safe. **Crucially, every indicator that requires a lookback window (e.g. MA, RSI) must be computed with `min_periods` equal to that window, leaving NaNs in the initial rows. The first 200 rows (max lookback) will be dropped before any training or signal generation.**

Features must never peek into the future. A dedicated look_ahead_check() function will verify this by shifting the target (1‑day forward return) and confirming no feature has a correlation > 0.01 with that future target.

Categories and examples:

**PRICE MOMENTUM**
- 5/10/20/60/120‑day log returns
- RSI(14), Stochastic %K(14,3), MACD(12,26,9) histogram
- Rate of change (ROC), 20‑day moving average slope

**MEAN REVERSION & OVERBOUGHT/OVERSOLD**
- Z‑score of close vs 20/50/200‑day MA
- Bollinger %B (20,2), distance from 52‑week high/low
- CCI(20), Williams %R(14)

**VOLATILITY & RISK**
- ATR(14), realized vol (5/20/60‑day)
- Garman‑Klass, Parkinson, and Rogers‑Satchell volatility estimators

**VOLUME (conditional)**
- If `volume_available` is True: OBV, volume z‑score (20d).
- If False: skip all volume features. Add a boolean feature `volume_flag` (always 0) to keep feature dimensions consistent.

**GOLD‑SPECIFIC MACRO FACTORS (all using publicly available data)**
- DXY daily return & 20‑day momentum
- Real interest rate: 10Y yield minus TIPS breakeven
- TLT/SPY ratio (risk sentiment)
- Gold/SPY relative strength (20‑day)
- Gold/TLT relative strength
- VIX level and VIX 20‑day change
- (Optionally) GLD ETF flows proxy: daily change in GLD shares outstanding

**ALL FEATURES must be output as a single DataFrame with a DateTimeIndex.**

Write at least 3 unit tests in `tests/test_features.py` covering:
- Feature shape consistency
- No NaN in feature columns after dropping warm‑up rows
- The `look_ahead_check()` function catching leakage

─── PHASE 4 · ML Signal Models ───────────────────────────────
Build models/trainer.py:

**Target definition (to avoid future‑leakage mismatch):**  
At each day `t`, the target is the **sign of the 1‑day forward return**:  
`target_t = sign(close[t+1] / close[t] - 1)`.  
This aligns perfectly with a daily bar loop where you trade at the next open and can update signals every day. (If you later want a 5‑day horizon, you must hold the position for 5 days; but we use 1‑day for clean daily execution.)

Train THREE models per ticker (only XAUUSD, but the pipeline is built ticker‑agnostic):

**Model A – LightGBM (or XGBoost)**
- TimeSeriesSplit cross‑validation (5 folds, with a 5‑day gap between train & test to prevent leakage).
- Hyperparameter tuning: max_depth, n_estimators, learning_rate.
- Log feature importance.

**Model B – Logistic Regression with Elastic Net**
- Scale features with RobustScaler.
- Tune C and l1_ratio via nested TimeSeriesSplit CV.

**Model C – Feedforward Neural Network (MLP) in PyTorch**
- Input: flattened vector of the last 20 days of the top‑10 features (from Model A importance).
- Architecture: 2 hidden layers (128 → 64), ReLU, Dropout(0.3), output (1) with Sigmoid.
- Train with binary cross‑entropy, Adam, early stopping.
- **This is fast enough to be retrained every walk‑forward step** (no LSTM, no heavy recurrence).

**Model persistence**: After training, save all models to the `models/` directory:
- Models A & B: `joblib` (`.pkl`)
- Model C: `torch.save` (`.pth`)
Include the train‑end date in the filename. Also save the scaler and feature list.

Build models/ensemble.py:
- **Ensemble probability** = average of the three models’ predicted probabilities.
- **Signal logic**:  
  - LONG signal if ensemble probability > 0.60  
  - SHORT signal if ensemble probability < 0.40  
  - HOLD otherwise  
- This uses a symmetric confidence margin.  
- Output a dict: `{ticker: {"signal": "LONG"/"SHORT"/"HOLD", "confidence": ensemble_prob}}`.  
  (Confidence is the raw ensemble probability; signals are only generated when confident enough.)

─── PHASE 5 · Strategy Layer ─────────────────────────────────
Build strategies/signal_generator.py:
- Convert ensemble output into trade signals for XAUUSD.
- Apply a **regime filter**: if SPY 50‑day MA < SPY 200‑day MA (bear stock market) **AND** XAUUSD close < XAUUSD 200‑day MA, suppress all LONG signals (convert to HOLD).
- Apply a **volatility filter**: if the current XAUUSD 20‑day realized vol is above its 1‑year 95th percentile, skip trading (HOLD) for that day.

Build strategies/position_manager.py:
- Because only one instrument, the “portfolio” is a single position.
- **Entry logic**: only enter if there is no current position.
  - If signal is LONG → open long position.
  - If signal is SHORT → open short position.
  - Size determined by risk manager.
- **Exit hierarchy (checked each day, applied at next open)**:
  1. **Stop‑loss hit**: if price crosses the trailing stop level intra‑day (simulated at bar open/high/low), exit immediately at the stop level. (Trailing stop calculation shown in Phase 6.)
  2. **Opposite signal**: if a new signal (LONG when short, or SHORT when long) appears, exit the current position first, then enter the new one (execute exit at next open, then entry at same open).
  3. **Confidence decay**: if no stop‑loss and no opposite signal, but the ensemble confidence for the current direction drops below 0.55, exit the position at the next open.
- **No pyramiding**: no additional entries while already in a position in the same direction.

─── PHASE 6 · Risk Management ────────────────────────────────
Build risk/risk_manager.py:
- **Position sizing (Kelly Criterion)**:
  - Use the last 20 closed trades (loaded from `execution/trade_history.json` if available; otherwise empty). Compute win rate `p` and win/loss ratio `R` (average win / average loss). If fewer than 10 trades, fall back to `entry_size_mult * account_equity` (e.g., 0.1 lots).
  - Kelly fraction: `f = p - (1-p)/R`. Apply **half‑Kelly**: `f_half = f / 2`.
  - Notional size = `f_half * equity`, converted to whole lots (round down). Cap at `max_position_pct * equity`.
  - Minimum size = 0.01 lots; if size < 0.01, reject trade.

- **Trailing stop‑loss** (ATR‑based):
  - For a **long** position: at entry, initial stop = entry_price - 2 * ATR(14). Each day, update stop to max(current_stop, high_since_entry - 2 * ATR). Exit if close <= stop (or low <= stop during the bar – simulate at stop price).
  - For a **short** position: initial stop = entry_price + 2 * ATR. Update stop to min(current_stop, low_since_entry + 2 * ATR). Exit if close >= stop.
  - Record exit price as the stop level (slippage + commission apply).

- **Drawdown guard**: If current drawdown from equity peak exceeds `max_drawdown_limit`, halt all new entries until drawdown recovers below 75% of the limit (e.g., if limit is 20%, resume trading when drawdown recovers back below 15%).

- **Risk check before entry**: Receive a signal and current portfolio state, return an approved size (possibly zero if checks fail). This is a pure function: `check_signal(signal, portfolio_state) -> ApprovedOrder`.

Build risk/monitor.py:
- Provide a real‑time (simulated) risk dashboard dict: `{drawdown, VaR_95 (historical simulation, 1‑day), current position (side, size, entry price), trailing stop, exposure % of equity}`.

─── PHASE 7 · Paper Trading Engine ───────────────────────────
Build execution/paper_broker.py:
- Simulate XAUUSD order execution:
  - Orders placed at day `t` fill at the **next day’s open** (open of `t+1`).
  - Slippage: adjust fill price by `slippage_per_unit` (multiply by 1 + slippage for buys, 1 - slippage for sells).
  - Commission: charge `commission_per_unit * notional` (notional = lots * lot_size * fill_price).
- Support long/short via sign of quantity (positive = long, negative = short).
- Daily mark‑to‑market P&L using the close price. Account equity = cash + unrealised P&L.
- Maintain trade history: list of closed trades with entry/exit prices, P&L, date, holding period. Append to `execution/trade_history.json` after each closed trade.
- After each trading day, serialize full portfolio state to `execution/portfolio_state.json` (positions, cash, equity, trade history).

Build execution/order_router.py:
- Accept signals from `position_manager` (already filtered and sized via risk manager).
- Pass the approved order to `paper_broker` for execution.
- Log all orders with timestamps and a unique order ID.

─── PHASE 8 · Backtester ─────────────────────────────────────
Build backtest/engine.py — walk‑forward backtester:
- **Walk‑forward**: train on 24 months, test on 1 month, step 1 month.  
- For each test month:
  - Retrain Models A and B on all data up to the test month start. Model C is retrained as well (it's an MLP, fast; if needed, you may reuse a previously trained model as a warm start and fine‑tune with a few epochs on new data to save time, but full retrain is acceptable).
  - Generate daily signals for each day in the test month **without look‑ahead**.
  - Simulate execution using the paper broker (which handles entry/exit logic, stop‑loss, etc.).
  - Collect daily equity curve, returns, and trade log.
- **Survivorship bias**: State in logs that we use only XAUUSD, so survivorship is irrelevant.
- **Benchmark**: Compute buy‑and‑hold returns for XAUUSD over the same test months. Include benchmark metrics in final report.
- Output: equity curve (DataFrame), daily returns, trade log.

Build backtest/metrics.py:
- Compute for both strategy and benchmark: total return, CAGR, Sharpe ratio (annualized, risk‑free = 4.5%), Sortino ratio, max drawdown and duration, Calmar ratio, win rate, profit factor, average win/loss, number of trades.
- Also compute **annualized turnover** for the strategy.
- Provide a function that returns a dict of all metrics.

─── PHASE 9 · Analytics & Reporting ──────────────────────────
Build analytics/reporter.py:
- Generate an HTML report (saved to `analytics/report.html`) containing:
  - Equity curve chart with benchmark overlay (matplotlib → base64)
  - Drawdown chart
  - Monthly returns heatmap
  - Feature importance bar chart (from Model A)
  - Metrics table (side‑by‑side strategy vs benchmark)
  - Top 10 trades by P&L
- Print a clean ASCII summary to console.

─── PHASE 10 · Orchestrator, Integration, & Final Delivery ────
Build `main.py` with CLI:
ython main.py --mode backtest # full walk‑forward backtest + HTML report
python main.py --mode paper # live paper trading loop (daily bar)
python main.py --mode train # retrain all models on latest data and save artifacts

**Live paper trading loop (--mode paper)**:  
- Run once per day at a fixed time (e.g., 23:30 UTC). Use `time.sleep()` to poll until the next expected run. On each iteration, fetch the latest daily bar; if a new bar is available (date > last processed date), run the full signal‑execution pipeline, then save state and wait for the next day.

**Integration smoke test**:  
- Add an `--mode test` flag or integrate into the orchestrator that runs a quick end‑to‑end test with **synthetic data** (a generated sine wave plus noise). It should verify that: the pipeline runs without crashing, a backtest completes in under 2 minutes, and basic metrics (positive Sharpe, trade count > 0) are present. This ensures all components are wired correctly.

Wire all components with structured logging (Python logging module) at DEBUG/INFO/WARNING levels. Wrap the top‑level execution in a try/except that saves partial state and logs the error.

Write a comprehensive `README.md`:
- Setup instructions (`pip install -r requirements.txt`)
- How to run each mode
- Architecture overview (one paragraph per phase)
- A sample backtest result table (use plausible placeholder values)
- Explanation of the target alignment and why 1‑day returns were chosen

## QUALITY STANDARDS
- Every function has a docstring.
- Use `pathlib` and `config.yaml` for all paths.
- Absolutely **no look‑ahead bias** at any point.
- Use type hints throughout.
- At least 3 unit tests in `tests/test_features.py` + the integration smoke test.
- After each phase, print:  
  ✅ PHASE [N] COMPLETE  
  Files created: [list]  
  Key decisions: [1‑3 bullets]  
  Next: [Phase N+1]

## BEGIN
Start immediately with Phase 1. Do not ask for clarification. Make reasonable assumptions, state them briefly, and proceed. Build the entire system from start to finish.