"""
Ensemble Predictor Module

Combines predictions from three models (LightGBM, Logistic Regression, MLP)
into a single ensemble probability and generates trading signals.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import torch

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Ensemble predictor combining multiple ML models."""
    
    def __init__(self, config: dict, model_date: str = None):
        """
        Initialize the ensemble predictor.
        
        Args:
            config: Configuration dictionary
            model_date: Date string for model files (default: latest available)
        """
        self.config = config
        self.models_path = Path(config['paths']['models'])
        self.signal_thresholds = config.get('signal_thresholds', {
            'long_threshold': 0.60,
            'short_threshold': 0.40,
            'confidence_exit': 0.55
        })
        
        # Load models
        self.lgb_model = None
        self.logreg_model = None
        self.mlp_model = None
        self.feature_names = None
        
        self._load_models(model_date)
    
    def _find_latest_model_date(self) -> Optional[str]:
        """Find the most recent model files."""
        if not self.models_path.exists():
            return None
        
        # Look for model metadata files
        metadata_files = list(self.models_path.glob('model_metadata_*.json'))
        
        if not metadata_files:
            return None
        
        # Extract dates and find latest
        dates = []
        for f in metadata_files:
            try:
                date_str = f.stem.split('_')[-1]
                dates.append(date_str)
            except:
                continue
        
        if dates:
            return max(dates)
        
        return None
    
    def _load_models(self, model_date: str = None) -> None:
        """Load trained models from disk."""
        if model_date is None:
            model_date = self._find_latest_model_date()
        
        if model_date is None:
            logger.warning("No trained models found. Run training first.")
            return
        
        logger.info(f"Loading models from date: {model_date}")
        
        # Load LightGBM
        lgb_path = self.models_path / f'lgb_model_{model_date}.pkl'
        if lgb_path.exists():
            self.lgb_model = joblib.load(lgb_path)
            logger.info(f"Loaded LightGBM model from {lgb_path}")
        else:
            logger.warning(f"LightGBM model not found: {lgb_path}")
        
        # Load Logistic Regression
        logreg_path = self.models_path / f'logreg_model_{model_date}.pkl'
        if logreg_path.exists():
            self.logreg_model = joblib.load(logreg_path)
            logger.info(f"Loaded Logistic Regression model from {logreg_path}")
        else:
            logger.warning(f"Logistic Regression model not found: {logreg_path}")
        
        # Load MLP
        mlp_path = self.models_path / f'mlp_model_{model_date}.pth'
        if mlp_path.exists():
            self.mlp_model = torch.load(mlp_path, map_location='cpu')
            self.mlp_model.eval()
            logger.info(f"Loaded MLP model from {mlp_path}")
        else:
            logger.warning(f"MLP model not found: {mlp_path}")
        
        # Load feature names
        metadata_path = self.models_path / f'model_metadata_{model_date}.json'
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.feature_names = metadata.get('feature_names', [])
            logger.info(f"Loaded {len(self.feature_names)} feature names")
    
    def predict_lgb(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions from LightGBM.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of probabilities for class 1 (up)
        """
        if self.lgb_model is None:
            return np.full(len(X), 0.5)
        
        proba = self.lgb_model.predict_proba(X)[:, 1]
        return proba
    
    def predict_logreg(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions from Logistic Regression.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of probabilities for class 1 (up)
        """
        if self.logreg_model is None:
            return np.full(len(X), 0.5)
        
        # Scale features
        scaler = getattr(self.logreg_model, 'scaler', None)
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        proba = self.logreg_model.predict_proba(X_scaled)[:, 1]
        return proba
    
    def predict_mlp(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions from MLP.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of probabilities for class 1 (up)
        """
        if self.mlp_model is None:
            return np.full(len(X), 0.5)
        
        # Select top features
        feature_indices = getattr(self.mlp_model, 'feature_indices', None)
        if feature_indices is not None:
            X_selected = X[:, feature_indices]
        else:
            X_selected = X
        
        # Scale features
        scaler = getattr(self.mlp_model, 'scaler', None)
        if scaler is not None:
            X_scaled = scaler.transform(X_selected)
        else:
            X_scaled = X_selected
        
        # Convert to tensor
        X_t = torch.FloatTensor(X_scaled)
        
        # Predict
        with torch.no_grad():
            self.mlp_model.eval()
            proba = self.mlp_model(X_t).squeeze().numpy()
        
        return proba
    
    def predict_ensemble(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Get ensemble predictions by averaging model probabilities.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Array of ensemble probabilities
        """
        # Prepare feature matrix
        if self.feature_names is None:
            logger.warning("Feature names not loaded. Using all columns except 'target'.")
            feature_cols = [c for c in features_df.columns if c != 'target']
        else:
            # Use only features that exist in both the model and the dataframe
            feature_cols = [c for c in self.feature_names if c in features_df.columns]
        
        if len(feature_cols) == 0:
            logger.error("No matching features found!")
            return np.full(len(features_df), 0.5)
        
        X = features_df[feature_cols].values
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        # Get predictions from each model
        proba_lgb = self.predict_lgb(X)
        proba_logreg = self.predict_logreg(X)
        proba_mlp = self.predict_mlp(X)
        
        # Average ensemble
        ensemble_proba = (proba_lgb + proba_logreg + proba_mlp) / 3
        
        logger.debug(f"Ensemble predictions: mean={ensemble_proba.mean():.4f}, "
                    f"std={ensemble_proba.std():.4f}")
        
        return ensemble_proba
    
    def generate_signals(self, features_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Generate trading signals for all dates in the dataframe.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Dictionary mapping dates to signal dicts
        """
        # Get ensemble probabilities
        proba = self.predict_ensemble(features_df)
        
        # Generate signals
        signals = {}
        for idx, (date, row) in enumerate(features_df.iterrows()):
            p = proba[idx]
            
            if p > self.signal_thresholds['long_threshold']:
                signal = 'LONG'
            elif p < self.signal_thresholds['short_threshold']:
                signal = 'SHORT'
            else:
                signal = 'HOLD'
            
            signals[date] = {
                'signal': signal,
                'confidence': float(p),
                'raw_probability': float(p)
            }
        
        return signals
    
    def get_latest_signal(self, features_df: pd.DataFrame) -> Dict:
        """
        Get the most recent trading signal.
        
        Args:
            features_df: DataFrame with features (must have at least one row)
            
        Returns:
            Signal dictionary for the latest date
        """
        signals = self.generate_signals(features_df)
        latest_date = features_df.index[-1]
        return signals[latest_date]
