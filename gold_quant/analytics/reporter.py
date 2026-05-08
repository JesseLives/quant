"""
Analytics and Reporting Module

Generates HTML reports with:
- Equity curve chart with benchmark overlay
- Drawdown chart
- Monthly returns heatmap
- Feature importance bar chart
- Metrics table (strategy vs benchmark)
- Top 10 trades by P&L
"""

import logging
from pathlib import Path
from typing import Dict, Optional
import base64
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates performance reports."""
    
    def __init__(self, config: dict):
        """
        Initialize the report generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.analytics_path = Path(config['paths']['analytics'])
        self.analytics_path.mkdir(parents=True, exist_ok=True)
    
    def _plot_to_base64(self, fig: plt.Figure) -> str:
        """
        Convert matplotlib figure to base64 string.
        
        Args:
            fig: Matplotlib figure
            
        Returns:
            Base64 encoded PNG string
        """
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{img_base64}"
    
    def plot_equity_curve(self, equity_curve: pd.DataFrame, 
                         benchmark: pd.DataFrame = None) -> str:
        """
        Plot equity curve with optional benchmark overlay.
        
        Args:
            equity_curve: Strategy equity curve
            benchmark: Benchmark equity curve
            
        Returns:
            Base64 encoded plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Normalize to start at 100
        strategy_norm = equity_curve['equity'] / equity_curve['equity'].iloc[0] * 100
        ax.plot(strategy_norm.index, strategy_norm.values, label='Strategy', 
               linewidth=2, color='blue')
        
        if benchmark is not None and len(benchmark) > 0:
            bench_norm = benchmark['equity'] / benchmark['equity'].iloc[0] * 100
            ax.plot(bench_norm.index, bench_norm.values, label='Benchmark', 
                   linewidth=2, color='gray', linestyle='--')
        
        ax.set_title('Equity Curve (Normalized to 100)', fontsize=14)
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        return self._plot_to_base64(fig)
    
    def plot_drawdown(self, equity_curve: pd.DataFrame) -> str:
        """
        Plot drawdown chart.
        
        Args:
            equity_curve: Strategy equity curve
            
        Returns:
            Base64 encoded plot
        """
        fig, ax = plt.subplots(figsize=(12, 4))
        
        equity = equity_curve['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100
        
        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.7, color='red')
        ax.set_title('Drawdown (%)', fontsize=14)
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown %')
        ax.grid(True, alpha=0.3)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        return self._plot_to_base64(fig)
    
    def plot_monthly_returns(self, equity_curve: pd.DataFrame) -> str:
        """
        Plot monthly returns heatmap.
        
        Args:
            equity_curve: Strategy equity curve
            
        Returns:
            Base64 encoded plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate daily returns
        daily_returns = equity_curve['equity'].pct_change()
        
        # Reshape to year x month
        monthly_returns = []
        years = sorted(set(daily_returns.index.year))
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for year in years:
            row = []
            for month in range(1, 13):
                mask = (daily_returns.index.year == year) & (daily_returns.index.month == month)
                month_ret = daily_returns[mask].sum()  # Sum of daily returns
                if not np.isnan(month_ret):
                    row.append(month_ret * 100)  # Convert to percentage
                else:
                    row.append(np.nan)
            monthly_returns.append(row)
        
        # Create heatmap
        monthly_returns = np.array(monthly_returns)
        im = ax.imshow(monthly_returns, cmap='RdYlGn', aspect='auto', 
                      vmin=-10, vmax=10)
        
        # Set labels
        ax.set_xticks(range(12))
        ax.set_xticklabels(months)
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Return %')
        
        ax.set_title('Monthly Returns Heatmap', fontsize=14)
        
        return self._plot_to_base64(fig)
    
    def plot_feature_importance(self, feature_names: list, 
                               importance: np.ndarray, top_n: int = 15) -> str:
        """
        Plot feature importance bar chart.
        
        Args:
            feature_names: List of feature names
            importance: Array of importance values
            top_n: Number of top features to show
            
        Returns:
            Base64 encoded plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Sort by importance
        indices = np.argsort(importance)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importance = importance[indices]
        
        # Reverse for better visualization
        top_features = top_features[::-1]
        top_importance = top_importance[::-1]
        
        y_pos = np.arange(len(top_features))
        
        ax.barh(y_pos, top_importance, color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        return self._plot_to_base64(fig)
    
    def generate_html_report(self, results: Dict, feature_importance: Dict = None) -> str:
        """
        Generate comprehensive HTML report.
        
        Args:
            results: Backtest results dictionary
            feature_importance: Optional dict with 'names' and 'values'
            
        Returns:
            Path to generated HTML file
        """
        from backtest.metrics import compute_metrics
        
        logger.info("Generating HTML report...")
        
        # Compute metrics
        metrics = compute_metrics(
            results['equity_curve'],
            results.get('benchmark'),
            results.get('trade_history', [])
        )
        
        # Generate plots
        equity_chart = self.plot_equity_curve(
            results['equity_curve'],
            results.get('benchmark')
        )
        
        drawdown_chart = self.plot_drawdown(results['equity_curve'])
        
        monthly_heatmap = self.plot_monthly_returns(results['equity_curve'])
        
        feat_chart = None
        if feature_importance is not None:
            feat_chart = self.plot_feature_importance(
                feature_importance['names'],
                feature_importance['values']
            )
        
        # Get top trades
        trade_history = results.get('trade_history', [])
        top_trades = sorted(trade_history, key=lambda x: x.get('pnl', 0), reverse=True)[:10]
        
        # Build HTML
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Gold Quant Trading System - Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f9f9f9; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
        .metric-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
        .chart {{ margin: 30px 0; text-align: center; }}
        .chart img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏆 Gold Quant Trading System - Performance Report</h1>
        <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>📊 Performance Summary</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{metrics['total_return']:.1%}</div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['cagr']:.1%}</div>
                <div class="metric-label">CAGR</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['sharpe_ratio']:.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['sortino_ratio']:.2f}</div>
                <div class="metric-label">Sortino Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['max_drawdown']:.1%}</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['calmar_ratio']:.2f}</div>
                <div class="metric-label">Calmar Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['win_rate']:.1%}</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['profit_factor']:.2f}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
        </div>
        
        <h2>📈 Equity Curve</h2>
        <div class="chart">
            <img src="{equity_chart}" alt="Equity Curve">
        </div>
        
        <h2>📉 Drawdown Analysis</h2>
        <div class="chart">
            <img src="{drawdown_chart}" alt="Drawdown">
        </div>
        
        <h2>📅 Monthly Returns</h2>
        <div class="chart">
            <img src="{monthly_heatmap}" alt="Monthly Returns">
        </div>
"""
        
        if feat_chart:
            html_content += f"""
        <h2>🔍 Feature Importance</h2>
        <div class="chart">
            <img src="{feat_chart}" alt="Feature Importance">
        </div>
"""
        
        # Strategy vs Benchmark comparison
        html_content += """
        <h2>📋 Strategy vs Benchmark</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Strategy</th>
                <th>Benchmark</th>
            </tr>
            <tr>
                <td>Total Return</td>
                <td class="positive">{:.1%}</td>
                <td>{:.1%}</td>
            </tr>
            <tr>
                <td>CAGR</td>
                <td class="positive">{:.1%}</td>
                <td>{:.1%}</td>
            </tr>
            <tr>
                <td>Sharpe Ratio</td>
                <td>{:.2f}</td>
                <td>{:.2f}</td>
            </tr>
            <tr>
                <td>Max Drawdown</td>
                <td class="negative">{:.1%}</td>
                <td>{:.1%}</td>
            </tr>
        </table>
""".format(
    metrics['total_return'],
    metrics.get('benchmark_total_return', 0),
    metrics['cagr'],
    metrics.get('benchmark_cagr', 0),
    metrics['sharpe_ratio'],
    metrics.get('benchmark_sharpe', 0),
    metrics['max_drawdown'],
    metrics.get('benchmark_max_drawdown', 0)
)
        
        # Top trades
        html_content += """
        <h2>💰 Top 10 Trades</h2>
        <table>
            <tr>
                <th>Date</th>
                <th>Side</th>
                <th>Size (lots)</th>
                <th>Entry</th>
                <th>Exit</th>
                <th>P&L</th>
            </tr>
"""
        
        for trade in top_trades:
            pnl_class = 'positive' if trade.get('pnl', 0) > 0 else 'negative'
            html_content += f"""
            <tr>
                <td>{trade.get('entry_date', 'N/A')}</td>
                <td>{trade.get('side', 'N/A')}</td>
                <td>{trade.get('size', 0):.2f}</td>
                <td>{trade.get('entry_price', 0):.2f}</td>
                <td>{trade.get('exit_price', 0):.2f}</td>
                <td class="{pnl_class}">${trade.get('pnl', 0):,.2f}</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
</body>
</html>
"""
        
        # Save report
        report_path = self.analytics_path / 'report.html'
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {report_path}")
        
        # Print ASCII summary
        self.print_ascii_summary(metrics)
        
        return str(report_path)
    
    def print_ascii_summary(self, metrics: Dict) -> None:
        """Print ASCII summary to console."""
        print("\n" + "="*60)
        print("           GOLD QUANT TRADING SYSTEM - PERFORMANCE SUMMARY")
        print("="*60)
        print(f"  Period: {metrics['start_date']} to {metrics['end_date']}")
        print("-"*60)
        print(f"  Total Return:      {metrics['total_return']:>12.1%}")
        print(f"  CAGR:              {metrics['cagr']:>12.1%}")
        print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:>12.2f}")
        print(f"  Sortino Ratio:     {metrics['sortino_ratio']:>12.2f}")
        print(f"  Max Drawdown:      {metrics['max_drawdown']:>12.1%}")
        print(f"  Calmar Ratio:      {metrics['calmar_ratio']:>12.2f}")
        print("-"*60)
        print(f"  Win Rate:          {metrics['win_rate']:>12.1%}")
        print(f"  Profit Factor:     {metrics['profit_factor']:>12.2f}")
        print(f"  Number of Trades:  {metrics['num_trades']:>12d}")
        print(f"  Average Win:       ${metrics['avg_win']:>10,.2f}")
        print(f"  Average Loss:      ${metrics['avg_loss']:>10,.2f}")
        print("="*60 + "\n")
