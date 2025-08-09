#!/usr/bin/env python3
"""
Simplified Algorithmic Trading Exercise
=====================================

Implementation based on "Python for Algorithmic Trading" book:
- AdaBoost features (pages 298-301)
- Single training without retraining
- SOL-BRL with R$100 budget
- 6h session = 360 bars

Author: Everaldo Gomes
Date: 2025-08-08
"""

import numpy as np
import pandas as pd
import pickle
import structlog
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import asyncio

# Scikit-learn imports
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Imports locais
import sys
sys.path.append('/home/everaldo/Code/mercado-bitcoin-python')
from mercado_bitcoin_python import MBTpqoa

# Configurar logging estruturado
logging = structlog.get_logger()


@dataclass
class TradingConfig:
    """Trading exercise configuration"""
    # Asset
    asset_symbol: str = "SOL-BRL"
    
    # Data
    historical_bars: int = 2016  # 1 week 5min = 5*12*24*7
    session_bars: int = 360      # 6 hours 5min = 5*12*6
    timeframe: str = "5m"
    
    # Budget
    total_budget: float = 100.0  # R$100
    max_positions: int = 5       # max 5 positions
    position_size: float = 20.0  # R$20 each
    
    # ML Model
    tree_weight: float = 0.6     # 60% DecisionTree
    mlp_weight: float = 0.4      # 40% MLP
    confidence_threshold: float = 0.55
    
    # Features (based on book)
    window_size: int = 20        # Rolling window
    lags: int = 6               # Temporal lags
    
    # Paths
    data_path: str = "data/exercise"
    model_path: str = "data/exercise/model.pkl"


class SimpleMLEnsembleStrategy:
    """
    Simplified ML Ensemble Strategy based on AdaBoost book.
    
    Inline features from book (pages 298-301):
    - SMA(10,20,50)
    - Log returns
    - Momentum & Volatility  
    - Price Min/Max
    - Volume Ratio
    - 6 period lags
    
    No retraining during execution (as in the book).
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logging.bind(strategy="SimpleMLEnsemble", asset=config.asset_symbol)
        
        # Models
        self.tree_model: Optional[BaggingRegressor] = None
        self.mlp_model: Optional[BaggingRegressor] = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Features
        self.feature_columns: List[str] = []
        self.model_metrics: Dict[str, Any] = {}
        
        # Data
        self.data_path = Path(config.data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("SimpleMLEnsembleStrategy initialized", 
                        historical_bars=config.historical_bars,
                        session_bars=config.session_bars)
    
    def create_features_inline(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create sophisticated features based on SMA strategies from live_trading_system.
        
        AdaBoost book features + advanced strategy logic:
        - SMA slopes (trend direction)
        - Cross detection & strength
        - Price position relative to SMAs
        - Distance strength
        - Volume analysis
        + 6-period lags
        """
        df = data.copy()
        window = self.config.window_size
        
        # 1. Log returns (core of AdaBoost in book)
        df['return'] = np.log(df['close'] / df['close'].shift(1))
        
        # 2. SMAs (chosen periods: 10, 20, 50)
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # 3. AdaBoost book features
        df['momentum'] = df['return'].rolling(window).mean()
        df['volatility'] = df['return'].rolling(window).std()
        df['price_min'] = df['close'].rolling(window).min()
        df['price_max'] = df['close'].rolling(window).max()
        df['price_norm'] = (df['close'] - df['price_min']) / (df['price_max'] - df['price_min'])
        
        # 4. Volume ratio
        if 'volume' in df.columns:
            df['volume_avg'] = df['volume'].rolling(window).mean()
            df['volume_ratio'] = df['volume'] / df['volume_avg']
        else:
            df['volume_ratio'] = 1.0
        
        # ===== ADVANCED SMA STRATEGY FEATURES =====
        
        # 5. SMA Slopes (trend direction) - from live_trading_system
        df['sma_10_slope'] = self._calculate_slope(df['sma_10'])
        df['sma_20_slope'] = self._calculate_slope(df['sma_20'])
        df['sma_50_slope'] = self._calculate_slope(df['sma_50'])
        
        # 6. Cross Detection & Strength - from SMACrossoverStrategy
        df['cross_10_20'] = self._detect_cross(df['sma_10'], df['sma_20'])
        df['cross_20_50'] = self._detect_cross(df['sma_20'], df['sma_50'])
        df['cross_strength_10_20'] = abs((df['sma_10'] - df['sma_20']) / df['sma_20'])
        df['cross_strength_20_50'] = abs((df['sma_20'] - df['sma_50']) / df['sma_50'])
        
        # 7. Price Position - from SMACrossoverStrategy
        df['price_vs_smas'] = self._price_position_numeric(df['close'], df['sma_10'], df['sma_20'], df['sma_50'])
        
        # 8. Distance Strength - from SMASimpleStrategy
        df['price_dist_sma10'] = (df['close'] - df['sma_10']) / df['sma_10']
        df['price_dist_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['price_dist_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
        
        # 9. Trend Alignment (all SMAs in same direction)
        df['sma_trend_alignment'] = self._calculate_trend_alignment(
            df['sma_10_slope'], df['sma_20_slope'], df['sma_50_slope']
        )
        
        # 10. SMA Convergence/Divergence
        df['sma_convergence_10_20'] = self._calculate_convergence(df['sma_10'], df['sma_20'])
        df['sma_convergence_20_50'] = self._calculate_convergence(df['sma_20'], df['sma_50'])
        
        # 11. Lags of main features (6 periods as in book)
        base_features = [
            'return', 'momentum', 'volatility', 'volume_ratio', 'price_norm',
            'cross_strength_10_20', 'cross_strength_20_50', 'price_vs_smas',
            'sma_trend_alignment'
        ]
        
        for feature in base_features:
            if feature in df.columns:
                for lag in range(1, self.config.lags + 1):
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        return df
    
    def _calculate_slope(self, series: pd.Series) -> pd.Series:
        """
        Calculate slope/trend of a series (copied from live_trading_system).
        """
        slopes = pd.Series(index=series.index, dtype=float)
        
        for i in range(3, len(series)):
            recent_values = series.iloc[i-2:i+1]  # Last 3 points
            if len(recent_values.dropna()) >= 2:
                slope = (recent_values.iloc[-1] - recent_values.iloc[0]) / 2
                slopes.iloc[i] = slope
            else:
                slopes.iloc[i] = 0
        
        return slopes.fillna(0)
    
    def _detect_cross(self, fast_sma: pd.Series, slow_sma: pd.Series) -> pd.Series:
        """
        Detecta cruzamentos entre SMAs (copiado do SMACrossoverStrategy).
        
        Returns:
            +1: Golden Cross (fast cruza para cima)
            -1: Death Cross (fast cruza para baixo)
             0: Sem cruzamento
        """
        crosses = pd.Series(index=fast_sma.index, dtype=float).fillna(0)
        
        for i in range(1, len(fast_sma)):
            if pd.isna(fast_sma.iloc[i]) or pd.isna(slow_sma.iloc[i]):
                continue
            if pd.isna(fast_sma.iloc[i-1]) or pd.isna(slow_sma.iloc[i-1]):
                continue
                
            prev_fast, prev_slow = fast_sma.iloc[i-1], slow_sma.iloc[i-1]
            curr_fast, curr_slow = fast_sma.iloc[i], slow_sma.iloc[i]
            
            # Golden Cross
            if prev_fast <= prev_slow and curr_fast > curr_slow:
                crosses.iloc[i] = 1
            # Death Cross
            elif prev_fast >= prev_slow and curr_fast < curr_slow:
                crosses.iloc[i] = -1
        
        return crosses
    
    def _price_position_numeric(self, price: pd.Series, sma_10: pd.Series, 
                               sma_20: pd.Series, sma_50: pd.Series) -> pd.Series:
        """
        Posição numérica do preço relativa às SMAs.
        
        Returns:
            +2: Acima de todas as SMAs
            +1: Acima de algumas SMAs
             0: Entre SMAs
            -1: Abaixo de algumas SMAs
            -2: Abaixo de todas as SMAs
        """
        positions = pd.Series(index=price.index, dtype=float)
        
        for i in range(len(price)):
            if any(pd.isna(val) for val in [price.iloc[i], sma_10.iloc[i], sma_20.iloc[i], sma_50.iloc[i]]):
                positions.iloc[i] = 0
                continue
            
            p, s10, s20, s50 = price.iloc[i], sma_10.iloc[i], sma_20.iloc[i], sma_50.iloc[i]
            smas = [s10, s20, s50]
            
            if p > max(smas):
                positions.iloc[i] = 2  # Acima de todas
            elif p < min(smas):
                positions.iloc[i] = -2  # Abaixo de todas
            elif p > np.median(smas):
                positions.iloc[i] = 1   # Acima da maioria
            elif p < np.median(smas):
                positions.iloc[i] = -1  # Abaixo da maioria
            else:
                positions.iloc[i] = 0   # Entre
        
        return positions.fillna(0)
    
    def _calculate_trend_alignment(self, slope_10: pd.Series, slope_20: pd.Series, 
                                  slope_50: pd.Series) -> pd.Series:
        """
        Calcula alinhamento de tendências das SMAs.
        
        Returns:
            +1: Todas subindo
            -1: Todas descendo
             0: Divergentes
        """
        alignment = pd.Series(index=slope_10.index, dtype=float)
        
        for i in range(len(slope_10)):
            s10, s20, s50 = slope_10.iloc[i], slope_20.iloc[i], slope_50.iloc[i]
            
            if all(s > 0 for s in [s10, s20, s50]):
                alignment.iloc[i] = 1   # Todas subindo
            elif all(s < 0 for s in [s10, s20, s50]):
                alignment.iloc[i] = -1  # Todas descendo
            else:
                alignment.iloc[i] = 0   # Divergentes
        
        return alignment.fillna(0)
    
    def _calculate_convergence(self, fast_sma: pd.Series, slow_sma: pd.Series) -> pd.Series:
        """
        Calcula se SMAs estão convergindo ou divergindo.
        
        Returns:
            Positivo: Convergindo
            Negativo: Divergindo
        """
        diff = abs(fast_sma - slow_sma)
        convergence = -diff.diff()  # Negativo da derivada da diferença
        
        return convergence.fillna(0)
    
    def create_target(self, data: pd.DataFrame) -> pd.Series:
        """
        Criar target baseado no livro: direção do próximo movimento.
        """
        # Próximo retorno
        future_return = data['close'].shift(-1).pct_change(fill_method=None)
        
        # Direção: +1 para up, -1 para down (como no livro AdaBoost)
        target = np.where(future_return > 0, 1, -1)
        
        return pd.Series(target, index=data.index, name='direction')
    
    def prepare_ml_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preparar dados para ML como no livro."""
        # Criar features
        features_df = self.create_features_inline(data)
        
        # Criar target
        target = self.create_target(features_df)
        
        # Selecionar features para ML (excluir auxiliares)
        exclude_cols = {
            'direction', 'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'sma_10', 'sma_20', 'sma_50', 'price_min', 'price_max', 'volume_avg'
        }
        
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # Preparar X e y
        X = features_df[feature_cols]
        y = target
        
        # Remover NaNs
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        self.feature_columns = feature_cols
        self.logger.info("ML data prepared", 
                        features=len(feature_cols), 
                        samples=len(X))
        
        return X, y
    
    def create_bagging_models(self) -> Tuple[BaggingRegressor, BaggingRegressor]:
        """Criar modelos BaggingRegressor conforme especificação."""
        
        # DecisionTree base
        tree_base = DecisionTreeRegressor(
            max_depth=5,
            min_samples_leaf=10,
            random_state=42
        )
        
        tree_model = BaggingRegressor(
            estimator=tree_base,
            n_estimators=100,
            max_samples=0.8,
            max_features=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        # MLP base
        mlp_base = MLPRegressor(
            hidden_layer_sizes=(50, 25),
            activation='relu',
            solver='adam',
            alpha=0.01,
            max_iter=1000,
            early_stopping=True,
            random_state=42
        )
        
        mlp_model = BaggingRegressor(
            estimator=mlp_base,
            n_estimators=20,  # Menos para MLP (mais lento)
            max_samples=0.9,
            random_state=42,
            n_jobs=2
        )
        
        return tree_model, mlp_model
    
    def train_once(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Treinamento único como no livro (sem retraining).
        
        Split sequencial: 70% treino, 30% teste (como no livro página 301).
        """
        try:
            self.logger.info("Starting single training session")
            
            # Preparar dados ML
            X, y = self.prepare_ml_data(historical_data)
            
            if len(X) < 100:
                error_msg = f"Dados insuficientes: {len(X)} < 100"
                self.logger.error(error_msg)
                return {'status': 'failed', 'error': error_msg}
            
            # Split sequencial como no livro (70% treino)
            split = int(len(X) * 0.7)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]
            
            self.logger.info("Data split", 
                           train_samples=len(X_train), 
                           test_samples=len(X_test))
            
            # Check and handle NaN values
            self.logger.info(f"Checking for NaN values in training data...")
            nan_mask = pd.isna(X_train).any(axis=1) | pd.isna(y_train)
            if nan_mask.sum() > 0:
                self.logger.warning(f"Found {nan_mask.sum()} rows with NaN values, removing them")
                X_train = X_train[~nan_mask]
                y_train = y_train[~nan_mask]
            
            nan_mask_test = pd.isna(X_test).any(axis=1)
            if nan_mask_test.sum() > 0:
                self.logger.warning(f"Found {nan_mask_test.sum()} rows with NaN values in test set, removing them")
                X_test = X_test[~nan_mask_test]
                y_test = y_test[~nan_mask_test]
            
            # Normalização (como no livro)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Criar e treinar modelos
            self.tree_model, self.mlp_model = self.create_bagging_models()
            
            self.logger.info("Training DecisionTree BaggingRegressor")
            self.tree_model.fit(X_train_scaled, y_train)
            
            self.logger.info("Training MLP BaggingRegressor")
            self.mlp_model.fit(X_train_scaled, y_train)
            
            # Avaliar modelos
            tree_pred = self.tree_model.predict(X_test_scaled)
            mlp_pred = self.mlp_model.predict(X_test_scaled)
            
            tree_r2 = r2_score(y_test, tree_pred)
            mlp_r2 = r2_score(y_test, mlp_pred)
            
            # Salvar métricas
            self.model_metrics = {
                'tree_r2': tree_r2,
                'mlp_r2': mlp_r2,
                'n_samples': len(X),
                'n_features': X.shape[1],
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'training_date': datetime.now(),
                'feature_columns': self.feature_columns
            }
            
            self.is_trained = True
            
            # Salvar modelo (como no livro - pickle)
            self.save_model()
            
            self.logger.info("Training completed successfully",
                           tree_r2=tree_r2,
                           mlp_r2=mlp_r2,
                           features=X.shape[1])
            
            return {
                'status': 'success',
                'metrics': self.model_metrics
            }
            
        except Exception as e:
            error_msg = f"Erro no treinamento: {str(e)}"
            self.logger.error(error_msg)
            return {'status': 'failed', 'error': error_msg}
    
    def predict_ensemble(self, current_data: pd.DataFrame) -> Tuple[float, float, float, float]:
        """
        Predição ensemble: 60% Tree + 40% MLP (como configurado).
        """
        try:
            if not self.is_trained:
                return 0.0, 0.0, 0.0, 0.0
            
            # Preparar features
            features_df = self.create_features_inline(current_data)
            X_pred = features_df[self.feature_columns].tail(1).ffill()
            
            # Check for NaN values in prediction data
            if X_pred.isna().any().any():
                self.logger.warning("NaN values found in prediction features, using forward fill")
                X_pred = X_pred.ffill().bfill()  # ffill + bfill as fallback
                
                # If still NaN after ffill/bfill, use mean from training data
                if X_pred.isna().any().any():
                    self.logger.warning("Still have NaN after ffill/bfill, using feature means")
                    feature_means = pd.Series(self.scaler.mean_, index=X_pred.columns)
                    X_pred = X_pred.fillna(feature_means)
            
            X_pred_scaled = self.scaler.transform(X_pred)
            
            # Predições individuais
            tree_pred = self.tree_model.predict(X_pred_scaled)[0]
            mlp_pred = self.mlp_model.predict(X_pred_scaled)[0]
            
            # Ensemble final (60% tree, 40% mlp)
            final_pred = (tree_pred * self.config.tree_weight + 
                         mlp_pred * self.config.mlp_weight)
            
            # Confiança baseada na concordância
            agreement = 1 - abs(tree_pred - mlp_pred) / 2
            confidence = min(abs(agreement), 1.0)
            
            return final_pred, tree_pred, mlp_pred, confidence
            
        except Exception as e:
            self.logger.error("Prediction error", error=str(e))
            return 0.0, 0.0, 0.0, 0.0
    
    def generate_signal(self, current_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Gerar sinal de trading baseado na predição.
        """
        if not self.is_trained:
            return None
        
        final_pred, tree_pred, mlp_pred, confidence = self.predict_ensemble(current_data)
        
        # Verificar threshold de confiança
        if confidence < self.config.confidence_threshold:
            return None
        
        # Determinar direção
        signal_type = "buy" if final_pred > 0 else "sell"
        current_price = current_data['close'].iloc[-1]
        
        signal = {
            'timestamp': pd.Timestamp.now(),
            'asset_symbol': self.config.asset_symbol,
            'signal_type': signal_type,
            'price': current_price,
            'confidence': confidence,
            'predictions': {
                'tree': tree_pred,
                'mlp': mlp_pred,
                'final': final_pred
            },
            'reason': f"ML Ensemble: Tree={tree_pred:.3f}, MLP={mlp_pred:.3f}, Final={final_pred:.3f}"
        }
        
        self.logger.info("Signal generated",
                        signal_type=signal_type,
                        confidence=confidence,
                        prediction=final_pred)
        
        return signal
    
    def save_model(self):
        """Salvar modelo como no livro (pickle)."""
        try:
            model_data = {
                'tree_model': self.tree_model,
                'mlp_model': self.mlp_model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'model_metrics': self.model_metrics,
                'config': self.config
            }
            
            with open(self.config.model_path, 'wb') as f:
                pickle.dump(model_data, f)
                
            self.logger.info("Model saved", path=self.config.model_path)
            
        except Exception as e:
            self.logger.error("Error saving model", error=str(e))
    
    def load_model(self) -> bool:
        """Carregar modelo salvo."""
        try:
            if not Path(self.config.model_path).exists():
                return False
            
            with open(self.config.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.tree_model = model_data['tree_model']
            self.mlp_model = model_data['mlp_model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.model_metrics = model_data['model_metrics']
            
            self.is_trained = True
            
            self.logger.info("Model loaded successfully", 
                           features=len(self.feature_columns))
            return True
            
        except Exception as e:
            self.logger.error("Error loading model", error=str(e))
            return False


if __name__ == "__main__":
    # Teste básico da estratégia
    print("=== Simple ML Ensemble Strategy Test ===")
    
    # Configuração
    config = TradingConfig()
    
    # Dados de exemplo
    dates = pd.date_range('2024-01-01', periods=1000, freq='5T')
    np.random.seed(42)
    
    base_price = 150.0  # SOL price
    price_changes = np.random.randn(1000) * 5
    prices = base_price + np.cumsum(price_changes)
    
    test_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices - np.random.rand(1000) * 2,
        'high': prices + np.random.rand(1000) * 3,
        'low': prices - np.random.rand(1000) * 3,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    test_data.set_index('timestamp', inplace=True)
    
    # Testar estratégia
    strategy = SimpleMLEnsembleStrategy(config)
    
    print(f"Strategy created: {strategy.__class__.__name__}")
    print(f"Asset: {config.asset_symbol}")
    print(f"Historical bars needed: {config.historical_bars}")
    print(f"Session bars: {config.session_bars}")
    
    # Testar criação de features
    features_df = strategy.create_features_inline(test_data)
    print(f"Features created: {len([col for col in features_df.columns if col not in test_data.columns])} new columns")
    
    # Testar preparação de dados ML
    X, y = strategy.prepare_ml_data(test_data)
    print(f"ML data prepared: {len(X)} samples, {X.shape[1]} features")
    
    # Testar treinamento
    result = strategy.train_once(test_data)
    print(f"Training result: {result['status']}")
    
    if result['status'] == 'success':
        metrics = result['metrics']
        print(f"Tree R²: {metrics['tree_r2']:.4f}")
        print(f"MLP R²: {metrics['mlp_r2']:.4f}")
        
        # Testar predição
        signal = strategy.generate_signal(test_data.tail(100))
        if signal:
            print(f"Signal generated: {signal['signal_type']} at {signal['price']:.2f} (confidence: {signal['confidence']:.3f})")
        else:
            print("No signal generated (low confidence)")
    
    print("=== Test completed ===")