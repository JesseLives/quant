"""
ML Model Trainer Module

Trains three models for XAUUSD signal generation:
- Model A: LightGBM with time-series cross-validation
- Model B: Logistic Regression with Elastic Net
- Model C: Feedforward Neural Network (PyTorch MLP)

All models predict the sign of 1-day forward return.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, List, Any
import json

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss
import lightgbm as lgb

logger = logging.getLogger(__name__)


class MLPClassifier(nn.Module):
    """Feedforward Neural Network for binary classification."""
    
    def __init__(self, input_dim: int, hidden_layers: List[int] = [128, 64], 
                 dropout: float = 0.3):
        """
        Initialize MLP.
        
        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class ModelTrainer:
    """Trains and saves ML models for signal generation."""
    
    def __init__(self, config: dict):
        """
        Initialize the model trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models_path = Path(config['paths']['models'])
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Model parameters from config
        self.lgb_params = config.get('model_settings', {}).get('lgb_params', {
            'max_depth': 6,
            'n_estimators': 200,
            'learning_rate': 0.05,
            'num_leaves': 31
        })
        
        self.logreg_params = config.get('model_settings', {}).get('logreg_params', {
            'C': 1.0,
            'l1_ratio': 0.5
        })
        
        self.mlp_params = config.get('model_settings', {}).get('mlp_params', {
            'hidden_layers': [128, 64],
            'dropout': 0.3,
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001
        })
    
    def prepare_data(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for training.
        
        Args:
            features_df: DataFrame with features and target
            
        Returns:
            Tuple of (X array, y array, feature names)
        """
        # Drop rows with NaN target
        df = features_df.dropna(subset=['target'])
        
        # Separate features and target
        feature_cols = [c for c in df.columns if c != 'target']
        X = df[feature_cols].values
        y = df['target'].values
        
        # Convert target to binary (0/1) for classification
        # Target is -1, 0, or 1; we map to 0 (down/flat) and 1 (up)
        y_binary = (y > 0).astype(int)
        
        logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Class distribution: {np.bincount(y_binary)}")
        
        return X, y_binary, feature_cols
    
    def train_lightgbm(self, X: np.ndarray, y: np.ndarray, 
                       feature_names: List[str]) -> lgb.LGBMClassifier:
        """
        Train LightGBM model with time-series cross-validation.
        
        Args:
            X: Feature matrix
            y: Binary target
            feature_names: List of feature names
            
        Returns:
            Trained LightGBM classifier
        """
        logger.info("Training LightGBM model...")
        
        # Time series split with gap
        tscv = TimeSeriesSplit(n_splits=5, gap=5)
        
        # Initialize model
        model = lgb.LGBMClassifier(
            max_depth=self.lgb_params['max_depth'],
            n_estimators=self.lgb_params['n_estimators'],
            learning_rate=self.lgb_params['learning_rate'],
            num_leaves=self.lgb_params['num_leaves'],
            random_state=42,
            verbose=-1
        )
        
        # Cross-validation scores
        cv_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            cv_scores.append(acc)
            logger.debug(f"LightGBM Fold {fold+1} accuracy: {acc:.4f}")
        
        logger.info(f"LightGBM CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        
        # Retrain on full data
        model.fit(X, y)
        
        # Log feature importance
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top 10 LightGBM features:\n{importance.head(10)}")
        
        return model
    
    def train_logistic_regression(self, X: np.ndarray, y: np.ndarray) -> SGDClassifier:
        """
        Train Logistic Regression with Elastic Net regularization.
        
        Args:
            X: Feature matrix
            y: Binary target
            
        Returns:
            Trained logistic regression classifier
        """
        logger.info("Training Logistic Regression model...")
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5, gap=5)
        
        # Initialize model with Elastic Net penalty
        model = SGDClassifier(
            penalty='elasticnet',
            alpha=self.logreg_params['C'],
            l1_ratio=self.logreg_params['l1_ratio'],
            random_state=42,
            max_iter=1000,
            tol=1e-4
        )
        
        # Cross-validation
        cv_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            cv_scores.append(acc)
            logger.debug(f"LogReg Fold {fold+1} accuracy: {acc:.4f}")
        
        logger.info(f"LogReg CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        
        # Retrain on full data
        model.fit(X_scaled, y)
        
        # Store scaler with model
        model.scaler = scaler
        
        return model
    
    def train_mlp(self, X: np.ndarray, y: np.ndarray) -> MLPClassifier:
        """
        Train PyTorch MLP model.
        
        Args:
            X: Feature matrix
            y: Binary target
            
        Returns:
            Trained MLP model
        """
        logger.info("Training MLP model...")
        
        # Select top features based on variance (for faster training)
        # In production, you'd use feature importance from LightGBM
        variance = np.var(X, axis=0)
        top_indices = np.argsort(variance)[-10:]  # Top 10 features
        X_selected = X[:, top_indices]
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        # Split into train/validation (last 20% for validation)
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.mlp_params['batch_size'], 
                                  shuffle=True)
        
        # Initialize model
        input_dim = X_train.shape[1]
        model = MLPClassifier(
            input_dim=input_dim,
            hidden_layers=self.mlp_params['hidden_layers'],
            dropout=self.mlp_params['dropout']
        )
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.mlp_params['learning_rate'])
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.mlp_params['epochs']):
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t).squeeze()
                val_loss = criterion(val_outputs, y_val_t).item()
            
            avg_train_loss = train_loss / len(train_loader)
            logger.debug(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Store scaler and feature indices with model
        model.scaler = scaler
        model.feature_indices = top_indices
        
        logger.info(f"MLP training complete. Best val_loss: {best_val_loss:.4f}")
        
        return model
    
    def train_all_models(self, features_df: pd.DataFrame, save_date: str = None) -> Dict[str, Any]:
        """
        Train all three models and save artifacts.
        
        Args:
            features_df: DataFrame with features and target
            save_date: Date string for filename (default: today)
            
        Returns:
            Dictionary with trained models and metadata
        """
        from datetime import datetime
        
        if save_date is None:
            save_date = datetime.now().strftime('%Y%m%d')
        
        # Prepare data
        X, y, feature_names = self.prepare_data(features_df)
        
        # Train models
        lgb_model = self.train_lightgbm(X, y, feature_names)
        logreg_model = self.train_logistic_regression(X, y)
        mlp_model = self.train_mlp(X, y)
        
        # Save models
        lgb_path = self.models_path / f'lgb_model_{save_date}.pkl'
        logreg_path = self.models_path / f'logreg_model_{save_date}.pkl'
        mlp_path = self.models_path / f'mlp_model_{save_date}.pth'
        
        joblib.dump(lgb_model, lgb_path)
        logger.info(f"Saved LightGBM model to {lgb_path}")
        
        joblib.dump(logreg_model, logreg_path)
        logger.info(f"Saved Logistic Regression model to {logreg_path}")
        
        torch.save(mlp_model, mlp_path)
        logger.info(f"Saved MLP model to {mlp_path}")
        
        # Save feature list and scaler
        metadata = {
            'feature_names': feature_names,
            'train_date': save_date,
            'n_samples': len(X),
            'n_features': len(feature_names)
        }
        
        metadata_path = self.models_path / f'model_metadata_{save_date}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved model metadata to {metadata_path}")
        
        return {
            'lgb': lgb_model,
            'logreg': logreg_model,
            'mlp': mlp_model,
            'metadata': metadata
        }
