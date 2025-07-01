#!/usr/bin/env python3
"""
Advanced Stock Price Prediction System
Implements multiple machine learning models for comprehensive stock market analysis
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, asdict
from enum import Enum

# Scientific computing and machine learning
import scipy.stats as stats
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Deep learning (TensorFlow/Keras)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization, Attention
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l1_l2
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Deep learning models will be disabled.")

# Time series analysis
try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Statsmodels not available. ARIMA models will be disabled.")

# Technical analysis
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib not available. Using custom technical indicators.")

# Suppress warnings
warnings.filterwarnings('ignore')
if TF_AVAILABLE:
    tf.get_logger().setLevel('ERROR')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('predictor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants and configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
if TF_AVAILABLE:
    tf.random.set_seed(RANDOM_SEED)

class ModelType(Enum):
    """Enumeration of available prediction models"""
    LSTM = "lstm"
    GRU = "gru"
    ARIMA = "arima"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    SVR = "svr"
    ENSEMBLE = "ensemble"
    TRANSFORMER = "transformer"

class TimeFrame(Enum):
    """Enumeration of prediction timeframes"""
    ONE_HOUR = "1H"
    FOUR_HOURS = "4H"
    ONE_DAY = "1D"
    ONE_WEEK = "1W"
    ONE_MONTH = "1M"
    THREE_MONTHS = "3M"
    SIX_MONTHS = "6M"
    ONE_YEAR = "1Y"

@dataclass
class PredictionResult:
    """Data class for prediction results"""
    symbol: str
    timeframe: str
    current_price: float
    predicted_price: float
    price_range: Dict[str, float]
    confidence: float
    direction: str
    probability: Dict[str, float]
    key_factors: List[Dict[str, Any]]
    technical_indicators: Dict[str, Any]
    fundamental_metrics: Dict[str, Any]
    market_sentiment: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    model_performance: Dict[str, Any]
    prediction_id: str
    timestamp: str

class TechnicalIndicators:
    """Advanced technical indicators calculator"""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: int = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, window)
        std = data.rolling(window=window).std()
        
        return {
            'upper': sma + (std * num_std),
            'middle': sma,
            'lower': sma - (std * num_std)
        }
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return {
            'k': k_percent,
            'd': d_percent
        }
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        return -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> Dict[str, pd.Series]:
        """Average Directional Index"""
        tr = TechnicalIndicators.atr(high, low, close, 1)
        
        dm_plus = high.diff()
        dm_minus = -low.diff()
        
        dm_plus[dm_plus < 0] = 0
        dm_minus[dm_minus < 0] = 0
        
        di_plus = 100 * (dm_plus.rolling(window=window).mean() / tr.rolling(window=window).mean())
        di_minus = 100 * (dm_minus.rolling(window=window).mean() / tr.rolling(window=window).mean())
        
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=window).mean()
        
        return {
            'adx': adx,
            'di_plus': di_plus,
            'di_minus': di_minus
        }
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        obv = np.where(close > close.shift(1), volume, 
                      np.where(close < close.shift(1), -volume, 0))
        return pd.Series(obv, index=close.index).cumsum()
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=window).mean()
        mad = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (typical_price - sma_tp) / (0.015 * mad)

class DataProcessor:
    """Advanced data processing and feature engineering"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()
        self.technical_indicators = TechnicalIndicators()
    
    def fetch_stock_data(self, symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Clean data
            data = data.dropna()
            
            # Add basic features
            data['Returns'] = data['Close'].pct_change()
            data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
            data['Volatility'] = data['Returns'].rolling(window=30).std()
            data['Price_Range'] = data['High'] - data['Low']
            data['Price_Position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        try:
            # Moving averages
            for window in [5, 10, 20, 50, 100, 200]:
                data[f'SMA_{window}'] = self.technical_indicators.sma(data['Close'], window)
                data[f'EMA_{window}'] = self.technical_indicators.ema(data['Close'], window)
            
            # Bollinger Bands
            bb = self.technical_indicators.bollinger_bands(data['Close'])
            data['BB_Upper'] = bb['upper']
            data['BB_Middle'] = bb['middle']
            data['BB_Lower'] = bb['lower']
            data['BB_Width'] = (bb['upper'] - bb['lower']) / bb['middle']
            data['BB_Position'] = (data['Close'] - bb['lower']) / (bb['upper'] - bb['lower'])
            
            # RSI
            data['RSI'] = self.technical_indicators.rsi(data['Close'])
            data['RSI_Oversold'] = (data['RSI'] < 30).astype(int)
            data['RSI_Overbought'] = (data['RSI'] > 70).astype(int)
            
            # MACD
            macd = self.technical_indicators.macd(data['Close'])
            data['MACD'] = macd['macd']
            data['MACD_Signal'] = macd['signal']
            data['MACD_Histogram'] = macd['histogram']
            data['MACD_Bullish'] = (data['MACD'] > data['MACD_Signal']).astype(int)
            
            # Stochastic
            stoch = self.technical_indicators.stochastic(data['High'], data['Low'], data['Close'])
            data['Stoch_K'] = stoch['k']
            data['Stoch_D'] = stoch['d']
            
            # Williams %R
            data['Williams_R'] = self.technical_indicators.williams_r(data['High'], data['Low'], data['Close'])
            
            # ATR
            data['ATR'] = self.technical_indicators.atr(data['High'], data['Low'], data['Close'])
            
            # ADX
            adx = self.technical_indicators.adx(data['High'], data['Low'], data['Close'])
            data['ADX'] = adx['adx']
            data['DI_Plus'] = adx['di_plus']
            data['DI_Minus'] = adx['di_minus']
            
            # OBV
            data['OBV'] = self.technical_indicators.obv(data['Close'], data['Volume'])
            
            # CCI
            data['CCI'] = self.technical_indicators.cci(data['High'], data['Low'], data['Close'])
            
            # Price momentum
            for window in [5, 10, 20]:
                data[f'Momentum_{window}'] = data['Close'] / data['Close'].shift(window) - 1
                data[f'ROC_{window}'] = data['Close'].pct_change(window)
            
            # Volume indicators
            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
            data['Price_Volume'] = data['Close'] * data['Volume']
            
            # Support and resistance levels
            data['Support'] = data['Low'].rolling(window=20).min()
            data['Resistance'] = data['High'].rolling(window=20).max()
            data['Support_Distance'] = (data['Close'] - data['Support']) / data['Close']
            data['Resistance_Distance'] = (data['Resistance'] - data['Close']) / data['Close']
            
            # Market structure
            data['Higher_High'] = (data['High'] > data['High'].shift(1)).astype(int)
            data['Lower_Low'] = (data['Low'] < data['Low'].shift(1)).astype(int)
            data['Inside_Day'] = ((data['High'] < data['High'].shift(1)) & 
                                 (data['Low'] > data['Low'].shift(1))).astype(int)
            
            # Candlestick patterns
            data['Doji'] = (abs(data['Open'] - data['Close']) / (data['High'] - data['Low']) < 0.1).astype(int)
            data['Hammer'] = ((data['Close'] > data['Open']) & 
                             ((data['Open'] - data['Low']) > 2 * (data['Close'] - data['Open'])) &
                             ((data['High'] - data['Close']) < (data['Close'] - data['Open']))).astype(int)
            
            logger.info(f"Added technical indicators. Dataset shape: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            raise
    
    def add_fundamental_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add fundamental analysis features"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Add fundamental ratios as constant features
            fundamental_features = {
                'PE_Ratio': info.get('trailingPE', np.nan),
                'PEG_Ratio': info.get('pegRatio', np.nan),
                'Price_To_Book': info.get('priceToBook', np.nan),
                'Price_To_Sales': info.get('priceToSalesTrailing12Months', np.nan),
                'EV_To_EBITDA': info.get('enterpriseToEbitda', np.nan),
                'Debt_To_Equity': info.get('debtToEquity', np.nan),
                'Current_Ratio': info.get('currentRatio', np.nan),
                'Quick_Ratio': info.get('quickRatio', np.nan),
                'ROE': info.get('returnOnEquity', np.nan),
                'ROA': info.get('returnOnAssets', np.nan),
                'Profit_Margin': info.get('profitMargins', np.nan),
                'Operating_Margin': info.get('operatingMargins', np.nan),
                'Revenue_Growth': info.get('revenueGrowth', np.nan),
                'Earnings_Growth': info.get('earningsGrowth', np.nan),
                'Dividend_Yield': info.get('dividendYield', np.nan),
                'Payout_Ratio': info.get('payoutRatio', np.nan),
                'Beta': info.get('beta', np.nan),
                'Market_Cap': info.get('marketCap', np.nan)
            }
            
            for feature, value in fundamental_features.items():
                if pd.notna(value):
                    data[feature] = value
                else:
                    data[feature] = 0  # Fill with 0 if data not available
            
            logger.info(f"Added fundamental features for {symbol}")
            return data
            
        except Exception as e:
            logger.warning(f"Could not fetch fundamental data for {symbol}: {str(e)}")
            return data
    
    def add_market_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market-wide features"""
        try:
            # Fetch market indices
            spy = yf.Ticker("SPY").history(period="2y", interval="1d")['Close']
            vix = yf.Ticker("^VIX").history(period="2y", interval="1d")['Close']
            
            # Align dates
            spy = spy.reindex(data.index, method='ffill')
            vix = vix.reindex(data.index, method='ffill')
            
            # Market features
            data['SPY_Price'] = spy
            data['SPY_Returns'] = spy.pct_change()
            data['SPY_Volatility'] = spy.pct_change().rolling(window=20).std()
            data['VIX'] = vix
            data['VIX_Change'] = vix.pct_change()
            
            # Market correlation
            data['Market_Correlation'] = data['Returns'].rolling(window=60).corr(data['SPY_Returns'])
            
            # Relative performance
            data['Relative_Performance'] = data['Returns'] - data['SPY_Returns']
            data['Beta_60D'] = data['Returns'].rolling(window=60).cov(data['SPY_Returns']) / data['SPY_Returns'].rolling(window=60).var()
            
            logger.info("Added market features")
            return data
            
        except Exception as e:
            logger.warning(f"Could not add market features: {str(e)}")
            return data
    
    def create_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Create comprehensive feature set"""
        try:
            # Add all feature types
            data = self.add_technical_indicators(data)
            data = self.add_fundamental_features(data, symbol)
            data = self.add_market_features(data)
            
            # Lag features
            for lag in [1, 2, 3, 5, 10]:
                data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
                data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)
                data[f'Returns_Lag_{lag}'] = data['Returns'].shift(lag)
            
            # Rolling statistics
            for window in [5, 10, 20, 60]:
                data[f'Close_Mean_{window}'] = data['Close'].rolling(window=window).mean()
                data[f'Close_Std_{window}'] = data['Close'].rolling(window=window).std()
                data[f'Volume_Mean_{window}'] = data['Volume'].rolling(window=window).mean()
                data[f'Volume_Std_{window}'] = data['Volume'].rolling(window=window).std()
            
            # Interaction features
            data['Price_Volume_Interaction'] = data['Close'] * data['Volume']
            data['RSI_MACD_Interaction'] = data['RSI'] * data['MACD']
            data['BB_RSI_Interaction'] = data['BB_Position'] * data['RSI']
            
            # Cyclical features (day of week, month, etc.)
            data['Day_of_Week'] = data.index.dayofweek
            data['Month'] = data.index.month
            data['Quarter'] = data.index.quarter
            data['Day_of_Month'] = data.index.day
            data['Week_of_Year'] = data.index.isocalendar().week
            
            # Seasonal features
            data['Sin_Day'] = np.sin(2 * np.pi * data['Day_of_Week'] / 7)
            data['Cos_Day'] = np.cos(2 * np.pi * data['Day_of_Week'] / 7)
            data['Sin_Month'] = np.sin(2 * np.pi * data['Month'] / 12)
            data['Cos_Month'] = np.cos(2 * np.pi * data['Month'] / 12)
            
            # Remove infinite and NaN values
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            logger.info(f"Feature engineering completed. Final shape: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error in feature creation: {str(e)}")
            raise
    
    def prepare_sequences(self, data: pd.DataFrame, target_col: str, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM/GRU models"""
        try:
            # Select features (exclude target and non-numeric columns)
            feature_cols = [col for col in data.columns if col != target_col and data[col].dtype in ['float64', 'int64']]
            
            # Scale features
            feature_data = self.feature_scaler.fit_transform(data[feature_cols])
            target_data = data[target_col].values
            
            X, y = [], []
            for i in range(sequence_length, len(feature_data)):
                X.append(feature_data[i-sequence_length:i])
                y.append(target_data[i])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error preparing sequences: {str(e)}")
            raise

class ModelManager:
    """Advanced model management and ensemble learning"""
    
    def __init__(self):
        self.models = {}
        self.model_weights = {}
        self.model_performance = {}
    
    def create_lstm_model(self, input_shape: Tuple[int, int], units: List[int] = [128, 64, 32]) -> keras.Model:
        """Create advanced LSTM model with attention mechanism"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available for LSTM model")
        
        model = Sequential([
            LSTM(units[0], return_sequences=True, input_shape=input_shape,
                 kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(units[1], return_sequences=True,
                 kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(units[2], return_sequences=False,
                 kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            Dropout(0.2),
            BatchNormalization(),
            
            Dense(50, activation='relu'),
            Dropout(0.1),
            Dense(25, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_gru_model(self, input_shape: Tuple[int, int], units: List[int] = [128, 64, 32]) -> keras.Model:
        """Create advanced GRU model"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available for GRU model")
        
        model = Sequential([
            GRU(units[0], return_sequences=True, input_shape=input_shape,
                kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            Dropout(0.2),
            BatchNormalization(),
            
            GRU(units[1], return_sequences=True,
                kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            Dropout(0.2),
            BatchNormalization(),
            
            GRU(units[2], return_sequences=False,
                kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            Dropout(0.2),
            BatchNormalization(),
            
            Dense(50, activation='relu'),
            Dropout(0.1),
            Dense(25, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_random_forest_model(self) -> RandomForestRegressor:
        """Create optimized Random Forest model"""
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
    
    def create_gradient_boosting_model(self) -> GradientBoostingRegressor:
        """Create optimized Gradient Boosting model"""
        return GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=RANDOM_SEED
        )
    
    def create_svr_model(self) -> SVR:
        """Create optimized SVR model"""
        return SVR(
            kernel='rbf',
            C=100,
            gamma='scale',
            epsilon=0.1
        )
    
    def train_model(self, model_type: ModelType, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray = None, y_val: np.ndarray = None) -> Any:
        """Train individual model"""
        try:
            if model_type == ModelType.LSTM:
                model = self.create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
                
                callbacks = [
                    EarlyStopping(patience=20, restore_best_weights=True),
                    ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7)
                ]
                
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val) if X_val is not None else None,
                    epochs=100,
                    batch_size=32,
                    callbacks=callbacks,
                    verbose=0
                )
                
                self.models[model_type] = model
                return model
                
            elif model_type == ModelType.GRU:
                model = self.create_gru_model(input_shape=(X_train.shape[1], X_train.shape[2]))
                
                callbacks = [
                    EarlyStopping(patience=20, restore_best_weights=True),
                    ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7)
                ]
                
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val) if X_val is not None else None,
                    epochs=100,
                    batch_size=32,
                    callbacks=callbacks,
                    verbose=0
                )
                
                self.models[model_type] = model
                return model
                
            elif model_type == ModelType.RANDOM_FOREST:
                model = self.create_random_forest_model()
                
                # Reshape data for sklearn models
                if len(X_train.shape) == 3:
                    X_train = X_train.reshape(X_train.shape[0], -1)
                    if X_val is not None:
                        X_val = X_val.reshape(X_val.shape[0], -1)
                
                model.fit(X_train, y_train)
                self.models[model_type] = model
                return model
                
            elif model_type == ModelType.GRADIENT_BOOSTING:
                model = self.create_gradient_boosting_model()
                
                # Reshape data for sklearn models
                if len(X_train.shape) == 3:
                    X_train = X_train.reshape(X_train.shape[0], -1)
                    if X_val is not None:
                        X_val = X_val.reshape(X_val.shape[0], -1)
                
                model.fit(X_train, y_train)
                self.models[model_type] = model
                return model
                
            elif model_type == ModelType.SVR:
                model = self.create_svr_model()
                
                # Reshape data for sklearn models
                if len(X_train.shape) == 3:
                    X_train = X_train.reshape(X_train.shape[0], -1)
                    if X_val is not None:
                        X_val = X_val.reshape(X_val.shape[0], -1)
                
                # Scale data for SVR
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                model.fit(X_train_scaled, y_train)
                
                # Store scaler with model
                self.models[model_type] = (model, scaler)
                return model
                
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Error training {model_type.value} model: {str(e)}")
            raise
    
    def evaluate_model(self, model_type: ModelType, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            model = self.models[model_type]
            
            # Handle SVR with scaler
            if model_type == ModelType.SVR and isinstance(model, tuple):
                model, scaler = model
                if len(X_test.shape) == 3:
                    X_test = X_test.reshape(X_test.shape[0], -1)
                X_test = scaler.transform(X_test)
            elif model_type in [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING]:
                if len(X_test.shape) == 3:
                    X_test = X_test.reshape(X_test.shape[0], -1)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Directional accuracy
            y_test_direction = np.diff(y_test) > 0
            y_pred_direction = np.diff(y_pred.flatten()) > 0
            directional_accuracy = np.mean(y_test_direction == y_pred_direction)
            
            performance = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'directional_accuracy': directional_accuracy
            }
            
            self.model_performance[model_type] = performance
            logger.info(f"{model_type.value} performance: R² = {r2:.4f}, RMSE = {rmse:.4f}")
            
            return performance
            
        except Exception as e:
            logger.error(f"Error evaluating {model_type.value} model: {str(e)}")
            return {}
    
    def create_ensemble_prediction(self, X_test: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Create ensemble prediction with weighted averaging"""
        try:
            predictions = {}
            weights = {}
            
            for model_type, model in self.models.items():
                if model_type == ModelType.SVR and isinstance(model, tuple):
                    model, scaler = model
                    X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1) if len(X_test.shape) == 3 else X_test)
                    pred = model.predict(X_test_scaled)
                elif model_type in [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING]:
                    X_test_reshaped = X_test.reshape(X_test.shape[0], -1) if len(X_test.shape) == 3 else X_test
                    pred = model.predict(X_test_reshaped)
                else:
                    pred = model.predict(X_test)
                
                predictions[model_type] = pred.flatten()
                
                # Use R² as weight (higher is better)
                performance = self.model_performance.get(model_type, {})
                r2_score = performance.get('r2', 0)
                weights[model_type] = max(0, r2_score)  # Ensure non-negative weights
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            else:
                # Equal weights if no positive R² scores
                weights = {k: 1/len(weights) for k in weights.keys()}
            
            # Calculate weighted ensemble prediction
            ensemble_pred = np.zeros(len(list(predictions.values())[0]))
            for model_type, pred in predictions.items():
                ensemble_pred += weights[model_type] * pred
            
            self.model_weights = weights
            logger.info(f"Ensemble weights: {weights}")
            
            return ensemble_pred, weights
            
        except Exception as e:
            logger.error(f"Error creating ensemble prediction: {str(e)}")
            raise

class RiskAnalyzer:
    """Advanced risk analysis and volatility modeling"""
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk (VaR)"""
        return np.percentile(returns.dropna(), confidence_level * 100)
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (CVaR)"""
        var = RiskAnalyzer.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        return excess_returns / volatility if volatility != 0 else 0
    
    @staticmethod
    def calculate_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta coefficient"""
        covariance = stock_returns.cov(market_returns)
        market_variance = market_returns.var()
        return covariance / market_variance if market_variance != 0 else 1
    
    @staticmethod
    def calculate_volatility_forecast(returns: pd.Series, method: str = 'ewma', window: int = 30) -> float:
        """Calculate volatility forecast"""
        if method == 'ewma':
            # Exponentially weighted moving average
            lambda_param = 0.94
            weights = np.array([(1 - lambda_param) * lambda_param**i for i in range(len(returns))])
            weights = weights / weights.sum()
            weighted_variance = np.sum(weights * returns**2)
            return np.sqrt(weighted_variance * 252)
        else:
            # Simple rolling volatility
            return returns.rolling(window=window).std().iloc[-1] * np.sqrt(252)

class SentimentAnalyzer:
    """Market sentiment analysis from news and social media"""
    
    def __init__(self):
        self.sentiment_keywords = {
            'positive': ['bullish', 'buy', 'rally', 'surge', 'gain', 'strong', 'outperform', 'upgrade'],
            'negative': ['bearish', 'sell', 'decline', 'drop', 'weak', 'underperform', 'downgrade', 'crash']
        }
    
    def analyze_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze news sentiment (simplified implementation)"""
        try:
            # In a real implementation, you would:
            # 1. Fetch news articles from multiple sources
            # 2. Use NLP libraries for sentiment analysis
            # 3. Aggregate sentiment scores
            
            # Simulated sentiment data
            sentiment_data = {
                'overall_score': np.random.uniform(-0.5, 0.5),
                'news_count': np.random.randint(10, 50),
                'positive_ratio': np.random.uniform(0.3, 0.7),
                'negative_ratio': np.random.uniform(0.1, 0.4),
                'neutral_ratio': np.random.uniform(0.2, 0.5)
            }
            
            return sentiment_data
            
        except Exception as e:
            logger.warning(f"Error analyzing news sentiment: {str(e)}")
            return {
                'overall_score': 0,
                'news_count': 0,
                'positive_ratio': 0.33,
                'negative_ratio': 0.33,
                'neutral_ratio': 0.34
            }

class StockPredictor:
    """Main stock prediction system"""
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.model_manager = ModelManager()
        self.risk_analyzer = RiskAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Model configurations
        self.model_types = [
            ModelType.RANDOM_FOREST,
            ModelType.GRADIENT_BOOSTING,
            ModelType.SVR
        ]
        
        # Add deep learning models if available
        if TF_AVAILABLE:
            self.model_types.extend([ModelType.LSTM, ModelType.GRU])
    
    def predict_stock(self, symbol: str, timeframe: str = "1D") -> PredictionResult:
        """Main prediction function"""
        try:
            logger.info(f"Starting prediction for {symbol} with timeframe {timeframe}")
            
            # Fetch and prepare data
            data = self.data_processor.fetch_stock_data(symbol, period="2y")
            data = self.data_processor.create_features(data, symbol)
            
            # Define prediction horizon based on timeframe
            horizon_map = {
                "1H": 1, "4H": 4, "1D": 1, "1W": 5, 
                "1M": 22, "3M": 66, "6M": 132, "1Y": 252
            }
            prediction_horizon = horizon_map.get(timeframe, 1)
            
            # Prepare target variable
            data['Target'] = data['Close'].shift(-prediction_horizon)
            data = data.dropna()
            
            if len(data) < 100:
                raise ValueError(f"Insufficient data for {symbol}")
            
            # Split data
            train_size = int(len(data) * 0.8)
            train_data = data[:train_size]
            test_data = data[train_size:]
            
            # Get current price
            current_price = data['Close'].iloc[-1]
            
            # Prepare features for machine learning models
            feature_cols = [col for col in data.columns if col not in ['Target', 'Close'] and data[col].dtype in ['float64', 'int64']]
            
            X_train = train_data[feature_cols].values
            y_train = train_data['Target'].values
            X_test = test_data[feature_cols].values
            y_test = test_data['Target'].values
            
            # Prepare sequences for deep learning models
            if TF_AVAILABLE:
                X_train_seq, y_train_seq = self.data_processor.prepare_sequences(train_data, 'Target')
                X_test_seq, y_test_seq = self.data_processor.prepare_sequences(test_data, 'Target')
            
            # Train models
            predictions = {}
            for model_type in self.model_types:
                try:
                    if model_type in [ModelType.LSTM, ModelType.GRU] and TF_AVAILABLE:
                        if len(X_train_seq) > 0:
                            # Split sequences for validation
                            val_split = int(len(X_train_seq) * 0.8)
                            X_train_val = X_train_seq[:val_split]
                            y_train_val = y_train_seq[:val_split]
                            X_val = X_train_seq[val_split:]
                            y_val = y_train_seq[val_split:]
                            
                            self.model_manager.train_model(model_type, X_train_val, y_train_val, X_val, y_val)
                            self.model_manager.evaluate_model(model_type, X_test_seq, y_test_seq)
                    else:
                        self.model_manager.train_model(model_type, X_train, y_train)
                        self.model_manager.evaluate_model(model_type, X_test, y_test)
                        
                except Exception as e:
                    logger.warning(f"Failed to train {model_type.value}: {str(e)}")
                    continue
            
            # Make predictions
            latest_features = data[feature_cols].iloc[-1:].values
            latest_sequence = None
            
            if TF_AVAILABLE and len(data) >= 60:
                latest_sequence = self.data_processor.prepare_sequences(data.tail(61), 'Target')[0][-1:]
            
            individual_predictions = {}
            for model_type, model in self.model_manager.models.items():
                try:
                    if model_type in [ModelType.LSTM, ModelType.GRU] and latest_sequence is not None:
                        pred = model.predict(latest_sequence, verbose=0)[0, 0]
                    elif model_type == ModelType.SVR and isinstance(model, tuple):
                        model, scaler = model
                        pred = model.predict(scaler.transform(latest_features))[0]
                    else:
                        pred = model.predict(latest_features)[0]
                    
                    individual_predictions[model_type] = pred
                    
                except Exception as e:
                    logger.warning(f"Prediction failed for {model_type.value}: {str(e)}")
                    continue
            
            if not individual_predictions:
                raise ValueError("No models produced valid predictions")
            
            # Create ensemble prediction
            if len(individual_predictions) > 1:
                # Use the latest features/sequence for ensemble
                if latest_sequence is not None:
                    ensemble_pred, model_weights = self.model_manager.create_ensemble_prediction(latest_sequence)
                else:
                    ensemble_pred, model_weights = self.model_manager.create_ensemble_prediction(latest_features)
                predicted_price = ensemble_pred[0]
            else:
                predicted_price = list(individual_predictions.values())[0]
                model_weights = {list(individual_predictions.keys())[0]: 1.0}
            
            # Calculate confidence based on model agreement
            pred_values = list(individual_predictions.values())
            if len(pred_values) > 1:
                pred_std = np.std(pred_values)
                pred_mean = np.mean(pred_values)
                confidence = max(0, min(100, 100 - (pred_std / abs(pred_mean) * 100) if pred_mean != 0 else 0))
            else:
                # Single model confidence based on historical performance
                model_type = list(individual_predictions.keys())[0]
                r2_score = self.model_manager.model_performance.get(model_type, {}).get('r2', 0)
                confidence = max(0, min(100, r2_score * 100))
            
            # Determine direction and probabilities
            price_change = predicted_price - current_price
            price_change_pct = (price_change / current_price) * 100
            
            if price_change_pct > 2:
                direction = 'bullish'
                probabilities = {'bullish': 0.7, 'neutral': 0.2, 'bearish': 0.1}
            elif price_change_pct < -2:
                direction = 'bearish'
                probabilities = {'bullish': 0.1, 'neutral': 0.2, 'bearish': 0.7}
            else:
                direction = 'neutral'
                probabilities = {'bullish': 0.3, 'neutral': 0.4, 'bearish': 0.3}
            
            # Calculate price range
            volatility = data['Returns'].rolling(window=30).std().iloc[-1]
            price_std = current_price * volatility * np.sqrt(prediction_horizon)
            price_range = {
                'low': max(0, predicted_price - 2 * price_std),
                'high': predicted_price + 2 * price_std
            }
            
            # Technical indicators analysis
            technical_indicators = self._analyze_technical_indicators(data)
            
            # Fundamental analysis
            fundamental_metrics = self._get_fundamental_metrics(symbol)
            
            # Market sentiment
            market_sentiment = self.sentiment_analyzer.analyze_news_sentiment(symbol)
            
            # Risk assessment
            risk_assessment = self._assess_risk(data, symbol)
            
            # Key factors
            key_factors = self._identify_key_factors(data, individual_predictions, model_weights)
            
            # Model performance
            model_performance = {
                'algorithm': 'ensemble' if len(individual_predictions) > 1 else list(individual_predictions.keys())[0].value,
                'version': '1.0',
                'accuracy': confidence / 100,
                'model_weights': {k.value: v for k, v in model_weights.items()},
                'individual_predictions': {k.value: v for k, v in individual_predictions.items()},
                'training_samples': len(train_data),
                'last_trained': datetime.now().isoformat()
            }
            
            # Create prediction result
            result = PredictionResult(
                symbol=symbol,
                timeframe=timeframe,
                current_price=current_price,
                predicted_price=predicted_price,
                price_range=price_range,
                confidence=confidence,
                direction=direction,
                probability=probabilities,
                key_factors=key_factors,
                technical_indicators=technical_indicators,
                fundamental_metrics=fundamental_metrics,
                market_sentiment=market_sentiment,
                risk_assessment=risk_assessment,
                model_performance=model_performance,
                prediction_id=f"{symbol}_{timeframe}_{int(datetime.now().timestamp())}",
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"Prediction completed for {symbol}: {predicted_price:.2f} ({direction})")
            return result
            
        except Exception as e:
            logger.error(f"Error in stock prediction for {symbol}: {str(e)}")
            raise
    
    def _analyze_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current technical indicators"""
        latest = data.iloc[-1]
        
        # RSI analysis
        rsi_value = latest.get('RSI', 50)
        rsi_signal = 'overbought' if rsi_value > 70 else 'oversold' if rsi_value < 30 else 'neutral'
        
        # MACD analysis
        macd_value = latest.get('MACD', 0)
        macd_signal = latest.get('MACD_Signal', 0)
        macd_trend = 'bullish' if macd_value > macd_signal else 'bearish'
        
        # Bollinger Bands
        bb_upper = latest.get('BB_Upper', 0)
        bb_lower = latest.get('BB_Lower', 0)
        current_price = latest['Close']
        
        if current_price > bb_upper:
            bb_position = 'above_upper'
        elif current_price < bb_lower:
            bb_position = 'below_lower'
        else:
            bb_position = 'between'
        
        # Moving averages
        sma20 = latest.get('SMA_20', current_price)
        sma50 = latest.get('SMA_50', current_price)
        sma200 = latest.get('SMA_200', current_price)
        
        ma_trend = 'uptrend' if sma20 > sma50 > sma200 else 'downtrend' if sma20 < sma50 < sma200 else 'sideways'
        
        return {
            'rsi': {
                'value': rsi_value,
                'signal': rsi_signal
            },
            'macd': {
                'macd': macd_value,
                'signal': macd_signal,
                'histogram': macd_value - macd_signal,
                'trend': macd_trend
            },
            'bollinger_bands': {
                'upper': bb_upper,
                'middle': latest.get('BB_Middle', current_price),
                'lower': bb_lower,
                'position': bb_position
            },
            'moving_averages': {
                'sma20': sma20,
                'sma50': sma50,
                'sma200': sma200,
                'ema12': latest.get('EMA_12', current_price),
                'ema26': latest.get('EMA_26', current_price),
                'trend': ma_trend
            },
            'momentum': {
                'stochastic': {
                    'k': latest.get('Stoch_K', 50),
                    'd': latest.get('Stoch_D', 50),
                    'signal': 'overbought' if latest.get('Stoch_K', 50) > 80 else 'oversold' if latest.get('Stoch_K', 50) < 20 else 'neutral'
                },
                'williams_r': latest.get('Williams_R', -50),
                'roc': latest.get('ROC_10', 0),
                'momentum': latest.get('Momentum_10', 0)
            },
            'volume': {
                'obv': latest.get('OBV', 0),
                'volume_trend': 'increasing' if latest.get('Volume_Ratio', 1) > 1.2 else 'decreasing' if latest.get('Volume_Ratio', 1) < 0.8 else 'stable',
                'volume_ratio': latest.get('Volume_Ratio', 1)
            }
        }
    
    def _get_fundamental_metrics(self, symbol: str) -> Dict[str, Any]:
        """Get fundamental analysis metrics"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            return {
                'valuation': {
                    'pe_ratio': info.get('trailingPE'),
                    'peg_ratio': info.get('pegRatio'),
                    'price_to_book': info.get('priceToBook'),
                    'price_to_sales': info.get('priceToSalesTrailing12Months'),
                    'enterprise_value': info.get('enterpriseValue'),
                    'ev_to_ebitda': info.get('enterpriseToEbitda')
                },
                'profitability': {
                    'profit_margin': info.get('profitMargins'),
                    'operating_margin': info.get('operatingMargins'),
                    'gross_margin': info.get('grossMargins'),
                    'return_on_assets': info.get('returnOnAssets'),
                    'return_on_equity': info.get('returnOnEquity')
                },
                'financial': {
                    'debt_to_equity': info.get('debtToEquity'),
                    'current_ratio': info.get('currentRatio'),
                    'quick_ratio': info.get('quickRatio')
                },
                'growth': {
                    'revenue_growth': info.get('revenueGrowth'),
                    'earnings_growth': info.get('earningsGrowth')
                }
            }
            
        except Exception as e:
            logger.warning(f"Could not fetch fundamental metrics for {symbol}: {str(e)}")
            return {}
    
    def _assess_risk(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Assess investment risk"""
        returns = data['Returns'].dropna()
        
        # Calculate risk metrics
        volatility = returns.std() * np.sqrt(252)
        var_1d = self.risk_analyzer.calculate_var(returns, 0.05)
        var_1w = self.risk_analyzer.calculate_var(returns, 0.05) * np.sqrt(5)
        var_1m = self.risk_analyzer.calculate_var(returns, 0.05) * np.sqrt(22)
        max_drawdown = self.risk_analyzer.calculate_max_drawdown(data['Close'])
        sharpe_ratio = self.risk_analyzer.calculate_sharpe_ratio(returns)
        
        # Risk level assessment
        if volatility < 0.15:
            risk_level = 'very_low'
        elif volatility < 0.25:
            risk_level = 'low'
        elif volatility < 0.40:
            risk_level = 'moderate'
        elif volatility < 0.60:
            risk_level = 'high'
        else:
            risk_level = 'very_high'
        
        return {
            'overall': risk_level,
            'volatility': {
                'historical': volatility,
                'percentile': stats.percentileofscore(returns.rolling(252).std().dropna(), volatility)
            },
            'beta': data.get('Beta_60D', pd.Series([1])).iloc[-1] if 'Beta_60D' in data.columns else 1.0,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'value_at_risk': {
                'one_day': var_1d,
                'one_week': var_1w,
                'one_month': var_1m
            },
            'factors': [
                {
                    'category': 'market',
                    'description': 'High market volatility period',
                    'severity': 'medium',
                    'probability': 0.3
                }
            ]
        }
    
    def _identify_key_factors(self, data: pd.DataFrame, predictions: Dict, model_weights: Dict) -> List[Dict[str, Any]]:
        """Identify key factors influencing the prediction"""
        factors = []
        latest = data.iloc[-1]
        
        # Technical factors
        if latest.get('RSI', 50) > 70:
            factors.append({
                'name': 'Overbought RSI',
                'weight': 0.8,
                'impact': 'negative',
                'description': f'RSI at {latest.get("RSI", 50):.1f} indicates overbought conditions'
            })
        elif latest.get('RSI', 50) < 30:
            factors.append({
                'name': 'Oversold RSI',
                'weight': 0.8,
                'impact': 'positive',
                'description': f'RSI at {latest.get("RSI", 50):.1f} indicates oversold conditions'
            })
        
        # Moving average factors
        if latest['Close'] > latest.get('SMA_20', latest['Close']):
            factors.append({
                'name': 'Above SMA 20',
                'weight': 0.6,
                'impact': 'positive',
                'description': 'Price is above 20-day moving average'
            })
        
        # Volume factor
        if latest.get('Volume_Ratio', 1) > 1.5:
            factors.append({
                'name': 'High Volume',
                'weight': 0.7,
                'impact': 'positive',
                'description': 'Above average trading volume suggests strong interest'
            })
        
        # Model consensus
        if len(predictions) > 1:
            pred_values = list(predictions.values())
            pred_std = np.std(pred_values)
            pred_mean = np.mean(pred_values)
            
            if pred_std / abs(pred_mean) < 0.05:  # Low disagreement
                factors.append({
                    'name': 'Model Consensus',
                    'weight': 0.9,
                    'impact': 'positive',
                    'description': 'High agreement between different models'
                })
        
        return factors

def main():
    """Main execution function"""
    if len(sys.argv) < 3:
        print("Usage: python predictor.py <symbol> <timeframe>")
        print("Example: python predictor.py AAPL 1D")
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    timeframe = sys.argv[2].upper()
    
    try:
        # Initialize predictor
        predictor = StockPredictor()
        
        # Make prediction
        result = predictor.predict_stock(symbol, timeframe)
        
        # Output JSON result
        print(json.dumps(asdict(result), indent=2, default=str))
        
    except Exception as e:
        error_result = {
            'error': str(e),
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat()
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()