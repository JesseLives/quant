"""
Feature Engineering Module

Generates 30+ gold-specific features across categories:
- Price momentum
- Mean reversion & overbought/oversold
- Volatility & risk
- Volume (conditional)
- Gold-specific macro factors

All features are forward-fill safe with proper min_periods to avoid look-ahead bias.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Any, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Generates features for XAUUSD trading."""
    
    def __init__(self, config: dict):
        """
        Initialize the feature engineer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.primary_ticker = config['primary_ticker']
        self.macro_tickers = config['macro_tickers']
    
    def compute_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute price momentum features.
        
        Features:
        - 5/10/20/60/120-day log returns
        - RSI(14)
        - Stochastic %K(14,3)
        - MACD(12,26,9) histogram
        - Rate of change (ROC)
        - 20-day moving average slope
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with momentum features added
        """
        df = df.copy()
        close = df['XAUUSD_Close']
        
        # Log returns at different horizons
        for period in [5, 10, 20, 60, 120]:
            df[f'return_{period}d'] = np.log(close / close.shift(period))
        
        # RSI(14)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Stochastic %K(14,3)
        low_14 = df['XAUUSD_Low'].rolling(window=14, min_periods=14).min()
        high_14 = df['XAUUSD_High'].rolling(window=14, min_periods=14).max()
        df['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14).replace(0, np.nan)
        
        # MACD(12,26,9)
        ema_12 = close.ewm(span=12, min_periods=12).mean()
        ema_26 = close.ewm(span=26, min_periods=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, min_periods=9).mean()
        df['macd_histogram'] = macd_line - signal_line
        df['macd_line'] = macd_line
        df['macd_signal'] = signal_line
        
        # Rate of Change (12-day)
        df['roc_12'] = (close - close.shift(12)) / close.shift(12)
        
        # 20-day MA slope (linear regression slope)
        ma_20 = close.rolling(window=20, min_periods=20).mean()
        # Simple slope approximation: (MA_today - MA_5days_ago) / 5
        df['ma_slope_20'] = (ma_20 - ma_20.shift(5)) / 5
        
        return df
    
    def compute_mean_reversion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute mean reversion and overbought/oversold features.
        
        Features:
        - Z-score vs 20/50/200-day MA
        - Bollinger %B (20,2)
        - Distance from 52-week high/low
        - CCI(20)
        - Williams %R(14)
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with mean reversion features added
        """
        df = df.copy()
        close = df['XAUUSD_Close']
        
        # Moving averages
        ma_20 = close.rolling(window=20, min_periods=20).mean()
        ma_50 = close.rolling(window=50, min_periods=50).mean()
        ma_200 = close.rolling(window=200, min_periods=200).mean()
        
        df['ma_20'] = ma_20
        df['ma_50'] = ma_50
        df['ma_200'] = ma_200
        
        # Z-scores
        std_20 = close.rolling(window=20, min_periods=20).std()
        std_50 = close.rolling(window=50, min_periods=50).std()
        std_200 = close.rolling(window=200, min_periods=200).std()
        
        df['zscore_20'] = (close - ma_20) / std_20.replace(0, np.nan)
        df['zscore_50'] = (close - ma_50) / std_50.replace(0, np.nan)
        df['zscore_200'] = (close - ma_200) / std_200.replace(0, np.nan)
        
        # Bollinger Bands %B
        bb_upper = ma_20 + 2 * std_20
        bb_lower = ma_20 - 2 * std_20
        df['bb_pct_b'] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
        
        # Distance from 52-week high/low (252 trading days)
        high_52w = df['XAUUSD_High'].rolling(window=252, min_periods=252).max()
        low_52w = df['XAUUSD_Low'].rolling(window=252, min_periods=252).min()
        
        df['dist_from_52w_high'] = (close - high_52w) / high_52w
        df['dist_from_52w_low'] = (close - low_52w) / low_52w
        
        # CCI(20) - Commodity Channel Index
        tp = (df['XAUUSD_High'] + df['XAUUSD_Low'] + close) / 3
        tp_ma_20 = tp.rolling(window=20, min_periods=20).mean()
        tp_std_20 = tp.rolling(window=20, min_periods=20).std()
        df['cci_20'] = (tp - tp_ma_20) / (0.015 * tp_std_20.replace(0, np.nan))
        
        # Williams %R(14)
        high_14 = df['XAUUSD_High'].rolling(window=14, min_periods=14).max()
        low_14 = df['XAUUSD_Low'].rolling(window=14, min_periods=14).min()
        df['williams_r'] = (high_14 - close) / (high_14 - low_14).replace(0, np.nan) * -100
        
        return df
    
    def compute_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute volatility and risk features.
        
        Features:
        - ATR(14)
        - Realized vol (5/20/60-day)
        - Garman-Klass volatility
        - Parkinson volatility
        - Rogers-Satchell volatility
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volatility features added
        """
        df = df.copy()
        high = df['XAUUSD_High']
        low = df['XAUUSD_Low']
        close = df['XAUUSD_Close']
        open_p = df['XAUUSD_Open']
        
        # ATR(14) - Average True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(window=14, min_periods=14).mean()
        
        # Realized volatility (log returns std)
        log_ret = np.log(close / close.shift(1))
        for period in [5, 20, 60]:
            df[f'realized_vol_{period}d'] = log_ret.rolling(
                window=period, min_periods=period
            ).std() * np.sqrt(252)  # Annualized
        
        # Garman-Klass volatility (more efficient than simple close-to-close)
        # GK = sqrt(1/n * sum(ln(H/L)^2 - (2*ln(2)-1)*ln(C/O)^2))
        ln_hl = np.log(high / low)
        ln_co = np.log(close / open_p)
        gk_sq = ln_hl ** 2 - (2 * np.log(2) - 1) * ln_co ** 2
        df['garman_klass_vol'] = np.sqrt(
            gk_sq.rolling(window=20, min_periods=20).mean()
        ) * np.sqrt(252)
        
        # Parkinson volatility
        # P = sqrt(1/(4*n*ln(2)) * sum(ln(H/L)^2))
        parkinson_sq = ln_hl ** 2 / (4 * np.log(2))
        df['parkinson_vol'] = np.sqrt(
            parkinson_sq.rolling(window=20, min_periods=20).mean()
        ) * np.sqrt(252)
        
        # Rogers-Satchell volatility
        # RS = sqrt(1/n * sum(ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)))
        ln_hc = np.log(high / close)
        ln_ho = np.log(high / open_p)
        ln_lc = np.log(low / close)
        ln_lo = np.log(low / open_p)
        rs_sq = ln_hc * ln_ho + ln_lc * ln_lo
        # Handle negative values (shouldn't happen but just in case)
        rs_sq = rs_sq.clip(lower=0)
        df['rogers_satchell_vol'] = np.sqrt(
            rs_sq.rolling(window=20, min_periods=20).mean()
        ) * np.sqrt(252)
        
        return df
    
    def compute_volume_features(self, df: pd.DataFrame, volume_available: bool) -> pd.DataFrame:
        """
        Compute volume-based features (if volume is available).
        
        Features:
        - OBV (On-Balance Volume)
        - Volume z-score (20d)
        - volume_flag (always present for consistency)
        
        Args:
            df: DataFrame with OHLCV data
            volume_available: Boolean flag indicating if volume data exists
            
        Returns:
            DataFrame with volume features added
        """
        df = df.copy()
        
        # Always add volume_flag for consistency
        df['volume_flag'] = 1 if volume_available else 0
        
        if not volume_available:
            logger.info("Skipping volume features (volume unavailable)")
            return df
        
        volume = df['XAUUSD_Volume']
        close = df['XAUUSD_Close']
        
        # OBV (On-Balance Volume)
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        df['obv'] = obv
        
        # Volume z-score (20-day)
        vol_ma_20 = volume.rolling(window=20, min_periods=20).mean()
        vol_std_20 = volume.rolling(window=20, min_periods=20).std()
        df['volume_zscore'] = (volume - vol_ma_20) / vol_std_20.replace(0, np.nan)
        
        logger.info("Volume features computed")
        
        return df
    
    def compute_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute gold-specific macro factor features.
        
        Features:
        - DXY daily return & 20-day momentum
        - Real interest rate proxy (10Y yield - TIPS breakeven proxy)
        - TLT/SPY ratio (risk sentiment)
        - Gold/SPY relative strength (20-day)
        - Gold/TLT relative strength
        - VIX level and VIX 20-day change
        - GLD daily change (ETF flows proxy)
        
        Args:
            df: DataFrame with merged data
            
        Returns:
            DataFrame with macro features added
        """
        df = df.copy()
        
        # Helper function to safely get column
        def get_col(ticker: str, col: str = 'Close') -> pd.Series:
            full_name = f'{ticker}_{col}'
            if full_name in df.columns:
                return df[full_name]
            return None
        
        # DXY (Dollar Index) features
        dxy_close = get_col('DX-Y.NYB')
        if dxy_close is not None:
            df['dxy_return'] = dxy_close.pct_change()
            df['dxy_momentum_20'] = dxy_close.pct_change(periods=20)
        
        # Real interest rate proxy: 10Y yield (^TNX) - inflation proxy (TIP momentum)
        tnx_close = get_col('^TNX')
        tip_close = get_col('TIP')
        if tnx_close is not None and tip_close is not None:
            # TIP breakeven inflation proxy (simplified: use TIP returns as inverse proxy)
            # More accurately, we'd use TIPS breakeven rate directly
            df['real_rate_proxy'] = tnx_close - tip_close.pct_change().rolling(20, min_periods=20).mean() * 100
        
        # TLT/SPY ratio (risk sentiment: bonds vs stocks)
        tlt_close = get_col('TLT')
        spy_close = get_col('SPY')
        if tlt_close is not None and spy_close is not None:
            df['tlt_spy_ratio'] = tlt_close / spy_close
            df['tlt_spy_momentum'] = (tlt_close / spy_close).pct_change(periods=20)
        
        # Gold/SPY relative strength (20-day)
        gold_close = df['XAUUSD_Close']
        if spy_close is not None:
            gold_spy_ratio = gold_close / spy_close
            df['gold_spy_rs'] = gold_spy_ratio
            df['gold_spy_rs_momentum'] = gold_spy_ratio.pct_change(periods=20)
        
        # Gold/TLT relative strength
        if tlt_close is not None:
            gold_tlt_ratio = gold_close / tlt_close
            df['gold_tlt_rs'] = gold_tlt_ratio
            df['gold_tlt_rs_momentum'] = gold_tlt_ratio.pct_change(periods=20)
        
        # VIX features
        vix_close = get_col('^VIX')
        if vix_close is not None:
            df['vix_level'] = vix_close
            df['vix_change_20'] = vix_close.pct_change(periods=20)
            df['vix_ma_20'] = vix_close.rolling(window=20, min_periods=20).mean()
        
        # GLD ETF flows proxy (daily change in GLD price as proxy for flows)
        gld_close = get_col('GLD')
        if gld_close is not None:
            df['gld_return'] = gld_close.pct_change()
            df['gld_momentum_20'] = gld_close.pct_change(periods=20)
        
        return df
    
    def generate_features(self, data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Main feature generation pipeline.
        
        Args:
            data: Dictionary with preprocessed data (from Preprocessor.process())
            
        Returns:
            Tuple of (features DataFrame, metadata dict)
        """
        logger.info("Starting feature engineering...")
        
        # Extract merged DataFrame and metadata
        if isinstance(data, tuple):
            merged_df, metadata = data
        else:
            merged_df = data
            metadata = {}
        
        # Get volume availability flag
        volume_available = metadata.get('volume_available', False)
        
        # Compute all feature categories
        features_df = merged_df.copy()
        
        logger.info("Computing momentum features...")
        features_df = self.compute_momentum_features(features_df)
        
        logger.info("Computing mean reversion features...")
        features_df = self.compute_mean_reversion_features(features_df)
        
        logger.info("Computing volatility features...")
        features_df = self.compute_volatility_features(features_df)
        
        logger.info("Computing volume features...")
        features_df = self.compute_volume_features(features_df, volume_available)
        
        logger.info("Computing macro features...")
        features_df = self.compute_macro_features(features_df)
        
        # Create target variable (sign of 1-day forward return)
        # This is aligned properly: target at time t uses close[t+1] and close[t]
        features_df['target'] = np.sign(
            features_df['XAUUSD_Close'].shift(-1) / features_df['XAUUSD_Close'] - 1
        )
        
        # Drop original OHLCV columns that are not features
        cols_to_drop = ['XAUUSD_Open', 'XAUUSD_High', 'XAUUSD_Low', 'XAUUSD_Close', 'XAUUSD_Volume']
        cols_to_drop += [c for c in features_df.columns if c.endswith('_Close') and c != 'XAUUSD_Close']
        
        # Keep only feature columns (not raw prices)
        feature_cols = [
            c for c in features_df.columns 
            if c not in cols_to_drop and c != 'target'
        ]
        
        features_df = features_df[feature_cols + ['target']]
        
        # Metadata about features
        feature_metadata = {
            'total_features': len(feature_cols),
            'feature_names': feature_cols,
            'volume_available': volume_available,
            'warmup_rows_needed': 252,  # Max lookback period (52-week high/low)
        }
        
        metadata['features'] = feature_metadata
        
        logger.info(f"Generated {len(feature_cols)} features")
        
        return features_df, metadata
    
    def look_ahead_check(self, features_df: pd.DataFrame, max_correlation: float = 0.01) -> bool:
        """
        Check for look-ahead bias by testing correlation with future target.
        
        Shifts the target by 1 day (to simulate having access to tomorrow's return)
        and checks if any feature has correlation > max_correlation with that shifted target.
        
        Args:
            features_df: DataFrame with features and target
            max_correlation: Maximum allowed absolute correlation
            
        Returns:
            True if no leakage detected, False otherwise
        """
        logger.info("Running look-ahead bias check...")
        
        # Create shifted target (tomorrow's target = day after tomorrow's return sign)
        # If features correlate with this, there's leakage
        shifted_target = features_df['target'].shift(-1)
        
        feature_cols = [c for c in features_df.columns if c != 'target']
        
        leakage_detected = False
        problematic_features = []
        
        for col in feature_cols:
            # Compute correlation with shifted target
            corr = features_df[col].corr(shifted_target)
            
            if abs(corr) > max_correlation and not np.isnan(corr):
                problematic_features.append((col, corr))
                leakage_detected = True
        
        if leakage_detected:
            logger.warning(f"Look-ahead bias detected in {len(problematic_features)} features:")
            for feat, corr in problematic_features[:10]:  # Show first 10
                logger.warning(f"  {feat}: correlation={corr:.4f}")
        else:
            logger.info("No look-ahead bias detected")
        
        return not leakage_detected
    
    def drop_warmup_rows(self, features_df: pd.DataFrame, warmup_period: int = 200) -> pd.DataFrame:
        """
        Drop initial rows where features have NaN due to lookback windows.
        
        Args:
            features_df: DataFrame with features
            warmup_period: Number of initial rows to drop
            
        Returns:
            DataFrame with warmup rows removed
        """
        logger.info(f"Dropping first {warmup_period} rows (warmup period)...")
        
        # First drop any rows with NaN
        df_clean = features_df.dropna()
        
        # Then drop the first warmup_period rows
        if len(df_clean) > warmup_period:
            df_clean = df_clean.iloc[warmup_period:]
        
        logger.info(f"Remaining rows after warmup drop: {len(df_clean)}")
        
        return df_clean
