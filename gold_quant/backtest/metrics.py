"""
Performance Metrics Module

Computes comprehensive performance metrics:
- Total return, CAGR
- Sharpe ratio, Sortino ratio
- Maximum drawdown and duration
- Calmar ratio
- Win rate, profit factor
- Average win/loss
- Annualized turnover
"""

import logging
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_cagr(equity_curve: pd.Series, years: float = None) -> float:
    """
    Compute Compound Annual Growth Rate.
    
    Args:
        equity_curve: Series of equity values
        years: Number of years (auto-calculated if None)
        
    Returns:
        CAGR as decimal
    """
    if len(equity_curve) < 2:
        return 0.0
    
    start_equity = equity_curve.iloc[0]
    end_equity = equity_curve.iloc[-1]
    
    if years is None:
        # Estimate years from index
        if hasattr(equity_curve.index, 'to_series'):
            days = (equity_curve.index.to_series()[-1] - equity_curve.index.to_series()[0]).days
        else:
            days = len(equity_curve)  # Assume daily
        years = days / 365.25
    
    if years <= 0 or start_equity <= 0:
        return 0.0
    
    cagr = (end_equity / start_equity) ** (1 / years) - 1
    
    return cagr


def compute_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.045, 
                        annualize: bool = True) -> float:
    """
    Compute Sharpe ratio.
    
    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate
        annualize: Whether to annualize
        
    Returns:
        Sharpe ratio
    """
    if len(returns) < 10 or returns.std() == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    
    sharpe = excess_returns.mean() / excess_returns.std()
    
    if annualize:
        sharpe *= np.sqrt(252)
    
    return sharpe


def compute_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.045,
                         annualize: bool = True) -> float:
    """
    Compute Sortino ratio (downside deviation).
    
    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate
        annualize: Whether to annualize
        
    Returns:
        Sortino ratio
    """
    if len(returns) < 10:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252
    
    # Downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return float('inf')  # No downside
    
    downside_std = np.sqrt((downside_returns ** 2).mean())
    
    sortino = excess_returns.mean() / downside_std
    
    if annualize:
        sortino *= np.sqrt(252)
    
    return sortino


def compute_max_drawdown(equity_curve: pd.Series) -> Tuple[float, int, int]:
    """
    Compute maximum drawdown and its duration.
    
    Args:
        equity_curve: Series of equity values
        
    Returns:
        Tuple of (max_drawdown, peak_idx, trough_idx)
    """
    # Running maximum
    running_max = equity_curve.expanding().max()
    
    # Drawdown series
    drawdown = (equity_curve - running_max) / running_max
    
    # Find max drawdown
    max_dd = drawdown.min()
    trough_idx = drawdown.idxmin()
    
    # Find peak before trough
    peak_idx = equity_curve[:trough_idx].idxmax()
    
    # Duration (in days)
    if hasattr(peak_idx, 'days'):
        duration = (trough_idx - peak_idx).days
    else:
        # Find recovery point
        post_trough = equity_curve[trough_idx:]
        recovery_level = equity_curve[peak_idx]
        recovered = post_trough[post_trough >= recovery_level]
        
        if len(recovered) > 0:
            recovery_idx = recovered.index[0]
            if hasattr(trough_idx, 'days'):
                duration = (recovery_idx - peak_idx).days
            else:
                duration = len(equity_curve[peak_idx:recovery_idx])
        else:
            duration = len(equity_curve[peak_idx:])
    
    return abs(max_dd), peak_idx, trough_idx


def compute_calmar_ratio(cagr: float, max_drawdown: float) -> float:
    """
    Compute Calmar ratio (CAGR / Max Drawdown).
    
    Args:
        cagr: Compound annual growth rate
        max_drawdown: Maximum drawdown (as positive decimal)
        
    Returns:
        Calmar ratio
    """
    if max_drawdown == 0:
        return 0.0
    
    return cagr / max_drawdown


def compute_win_rate(trade_history: list) -> float:
    """
    Compute win rate from trade history.
    
    Args:
        trade_history: List of trades with 'pnl' field
        
    Returns:
        Win rate as decimal
    """
    if len(trade_history) == 0:
        return 0.0
    
    wins = sum(1 for t in trade_history if t.get('pnl', 0) > 0)
    
    return wins / len(trade_history)


def compute_profit_factor(trade_history: list) -> float:
    """
    Compute profit factor (gross profit / gross loss).
    
    Args:
        trade_history: List of trades with 'pnl' field
        
    Returns:
        Profit factor
    """
    if len(trade_history) == 0:
        return 1.0
    
    gross_profit = sum(t.get('pnl', 0) for t in trade_history if t.get('pnl', 0) > 0)
    gross_loss = abs(sum(t.get('pnl', 0) for t in trade_history if t.get('pnl', 0) < 0))
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 1.0
    
    return gross_profit / gross_loss


def compute_avg_win_loss(trade_history: list) -> Tuple[float, float]:
    """
    Compute average win and average loss.
    
    Args:
        trade_history: List of trades with 'pnl' field
        
    Returns:
        Tuple of (avg_win, avg_loss)
    """
    wins = [t.get('pnl', 0) for t in trade_history if t.get('pnl', 0) > 0]
    losses = [t.get('pnl', 0) for t in trade_history if t.get('pnl', 0) < 0]
    
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = abs(np.mean(losses)) if losses else 0.0
    
    return avg_win, avg_loss


def compute_var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Compute historical Value at Risk.
    
    Args:
        returns: Series of daily returns
        confidence: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        VaR as positive decimal (loss)
    """
    if len(returns) < 10:
        return 0.0
    
    var = returns.quantile(1 - confidence)
    
    return abs(var)


def compute_metrics(equity_curve: pd.DataFrame, benchmark: pd.DataFrame = None,
                   trade_history: list = None, risk_free_rate: float = 0.045) -> Dict:
    """
    Compute comprehensive performance metrics.
    
    Args:
        equity_curve: DataFrame with 'equity' column and DateTimeIndex
        benchmark: Benchmark equity curve (optional)
        trade_history: List of closed trades
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Dictionary of metrics
    """
    # Extract equity series
    if isinstance(equity_curve, pd.DataFrame):
        equity = equity_curve['equity']
    else:
        equity = equity_curve
    
    # Compute daily returns
    returns = equity.pct_change().dropna()
    
    # Basic metrics
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    
    # CAGR
    if hasattr(equity.index, 'to_series'):
        years = (equity.index.to_series()[-1] - equity.index.to_series()[0]).days / 365.25
    else:
        years = len(equity) / 252
    
    cagr = compute_cagr(equity, years)
    
    # Risk-adjusted metrics
    sharpe = compute_sharpe_ratio(returns, risk_free_rate)
    sortino = compute_sortino_ratio(returns, risk_free_rate)
    
    # Drawdown
    max_dd, peak_idx, trough_idx = compute_max_drawdown(equity)
    
    # Calmar ratio
    calmar = compute_calmar_ratio(cagr, max_dd)
    
    # Trade statistics
    num_trades = len(trade_history) if trade_history else 0
    win_rate = compute_win_rate(trade_history) if trade_history else 0.0
    profit_factor = compute_profit_factor(trade_history) if trade_history else 1.0
    avg_win, avg_loss = compute_avg_win_loss(trade_history) if trade_history else (0.0, 0.0)
    
    # VaR
    var_95 = compute_var_historical(returns, 0.95)
    
    # Benchmark comparison
    benchmark_metrics = {}
    if benchmark is not None and len(benchmark) > 0:
        bench_equity = benchmark['equity'] if 'equity' in benchmark.columns else benchmark.iloc[:, 0]
        bench_returns = bench_equity.pct_change().dropna()
        
        benchmark_metrics = {
            'benchmark_total_return': (bench_equity.iloc[-1] / bench_equity.iloc[0]) - 1,
            'benchmark_cagr': compute_cagr(bench_equity, years),
            'benchmark_sharpe': compute_sharpe_ratio(bench_returns, risk_free_rate),
            'benchmark_max_drawdown': compute_max_drawdown(bench_equity)[0]
        }
    
    # Compile all metrics
    metrics = {
        'total_return': total_return,
        'cagr': cagr,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_dd,
        'calmar_ratio': calmar,
        'var_95': var_95,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'start_date': str(equity.index[0]),
        'end_date': str(equity.index[-1]),
        'initial_equity': equity.iloc[0],
        'final_equity': equity.iloc[-1],
        **benchmark_metrics
    }
    
    logger.info(f"Metrics computed: Sharpe={sharpe:.2f}, "
               f"Total Return={total_return:.2%}, Max DD={max_dd:.2%}")
    
    return metrics
