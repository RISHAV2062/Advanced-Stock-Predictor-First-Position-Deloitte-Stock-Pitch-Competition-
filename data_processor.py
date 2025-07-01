#!/usr/bin/env python3
"""
Advanced Data Processing and Analysis System
Handles real-time data ingestion, preprocessing, and feature engineering
"""

import os
import sys
import json
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import sqlite3
import redis
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from pathlib import Path

# Scientific computing libraries
import scipy.stats as stats
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer, KNNImputer

# Financial data sources
import yfinance as yf
import alpha_vantage
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.techindicators import TechIndicators

# News and sentiment analysis
import feedparser
import requests
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Database connections
import pymongo
from sqlalchemy import create_engine, text
import psycopg2

# Time series processing
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
CACHE_EXPIRY = 300  # 5 minutes
MAX_WORKERS = 10
BATCH_SIZE = 100
DEFAULT_LOOKBACK_DAYS = 365

@dataclass
class MarketData:
    """Market data container"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None
    dividend_amount: Optional[float] = None
    split_coefficient: Optional[float] = None

@dataclass
class NewsArticle:
    """News article container"""
    title: str
    content: str
    source: str
    published_at: datetime
    url: str
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    relevance_score: Optional[float] = None
    tickers: Optional[List[str]] = None

@dataclass
class EconomicIndicator:
    """Economic indicator container"""
    name: str
    value: float
    unit: str
    date: datetime
    source: str
    previous_value: Optional[float] = None
    change: Optional[float] = None

class DataSource:
    """Abstract base class for data sources"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.session = None
        self.rate_limit_delay = 0.1
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

class YahooFinanceSource(DataSource):
    """Yahoo Finance data source"""
    
    def __init__(self):
        super().__init__()
        self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"
    
    async def fetch_stock_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> List[MarketData]:
        """Fetch stock data from Yahoo Finance API"""
        try:
            url = f"{self.base_url}{symbol}"
            params = {
                'period1': int((datetime.now() - timedelta(days=365)).timestamp()),
                'period2': int(datetime.now().timestamp()),
                'interval': interval,
                'includePrePost': 'true',
                'events': 'div,splits'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_yahoo_response(data, symbol)
                else:
                    logger.error(f"Yahoo Finance API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data for {symbol}: {str(e)}")
            return []
    
    def _parse_yahoo_response(self, data: Dict, symbol: str) -> List[MarketData]:
        """Parse Yahoo Finance API response"""
        try:
            chart_data = data['chart']['result'][0]
            timestamps = chart_data['timestamp']
            ohlcv = chart_data['indicators']['quote'][0]
            
            market_data = []
            for i, timestamp in enumerate(timestamps):
                if all(ohlcv[key][i] is not None for key in ['open', 'high', 'low', 'close', 'volume']):
                    market_data.append(MarketData(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(timestamp, tz=timezone.utc),
                        open=ohlcv['open'][i],
                        high=ohlcv['high'][i],
                        low=ohlcv['low'][i],
                        close=ohlcv['close'][i],
                        volume=ohlcv['volume'][i],
                        adjusted_close=chart_data['indicators'].get('adjclose', [{}])[0].get('adjclose', [None])[i]
                    ))
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error parsing Yahoo Finance response: {str(e)}")
            return []

class AlphaVantageSource(DataSource):
    """Alpha Vantage data source"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.ts = TimeSeries(key=api_key, output_format='pandas')
        self.fd = FundamentalData(key=api_key, output_format='pandas')
        self.ti = TechIndicators(key=api_key, output_format='pandas')
    
    async def fetch_intraday_data(self, symbol: str, interval: str = '5min') -> List[MarketData]:
        """Fetch intraday data from Alpha Vantage"""
        try:
            data, meta_data = self.ts.get_intraday(symbol=symbol, interval=interval, outputsize='full')
            return self._parse_alpha_vantage_data(data, symbol)
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage intraday data for {symbol}: {str(e)}")
            return []
    
    async def fetch_daily_data(self, symbol: str) -> List[MarketData]:
        """Fetch daily data from Alpha Vantage"""
        try:
            data, meta_data = self.ts.get_daily_adjusted(symbol=symbol, outputsize='full')
            return self._parse_alpha_vantage_data(data, symbol, adjusted=True)
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage daily data for {symbol}: {str(e)}")
            return []
    
    async def fetch_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch fundamental data from Alpha Vantage"""
        try:
            overview = self.fd.get_company_overview(symbol)[0]
            income_statement = self.fd.get_income_statement_annual(symbol)[0]
            balance_sheet = self.fd.get_balance_sheet_annual(symbol)[0]
            cash_flow = self.fd.get_cash_flow_annual(symbol)[0]
            
            return {
                'overview': overview.to_dict() if not overview.empty else {},
                'income_statement': income_statement.to_dict() if not income_statement.empty else {},
                'balance_sheet': balance_sheet.to_dict() if not balance_sheet.empty else {},
                'cash_flow': cash_flow.to_dict() if not cash_flow.empty else {}
            }
            
        except Exception as e:
            logger.error(f"Error fetching fundamental data for {symbol}: {str(e)}")
            return {}
    
    def _parse_alpha_vantage_data(self, data: pd.DataFrame, symbol: str, adjusted: bool = False) -> List[MarketData]:
        """Parse Alpha Vantage response data"""
        try:
            market_data = []
            for timestamp, row in data.iterrows():
                market_data.append(MarketData(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=row.get('1. open', row.get('1. Open', 0)),
                    high=row.get('2. high', row.get('2. High', 0)),
                    low=row.get('3. low', row.get('3. Low', 0)),
                    close=row.get('4. close', row.get('4. Close', 0)),
                    volume=int(row.get('5. volume', row.get('6. Volume', 0))),
                    adjusted_close=row.get('5. adjusted close') if adjusted else None,
                    dividend_amount=row.get('7. dividend amount') if adjusted else None,
                    split_coefficient=row.get('8. split coefficient') if adjusted else None
                ))
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error parsing Alpha Vantage data: {str(e)}")
            return []

class NewsDataSource(DataSource):
    """News data aggregation source"""
    
    def __init__(self, api_keys: Dict[str, str]):
        super().__init__()
        self.api_keys = api_keys
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
    
    async def fetch_news(self, symbols: List[str], sources: List[str] = None) -> List[NewsArticle]:
        """Fetch news articles from multiple sources"""
        articles = []
        
        # RSS feeds
        rss_sources = [
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'https://feeds.reuters.com/reuters/businessNews',
            'https://feeds.bloomberg.com/markets/news.rss',
            'https://rss.cnn.com/rss/money_latest.rss'
        ]
        
        for rss_url in rss_sources:
            try:
                articles.extend(await self._fetch_rss_news(rss_url, symbols))
            except Exception as e:
                logger.warning(f"Error fetching RSS from {rss_url}: {str(e)}")
        
        # NewsAPI
        if 'newsapi' in self.api_keys:
            try:
                articles.extend(await self._fetch_newsapi_articles(symbols))
            except Exception as e:
                logger.warning(f"Error fetching NewsAPI articles: {str(e)}")
        
        # Process sentiment for all articles
        for article in articles:
            article.sentiment_score, article.sentiment_label = self._analyze_sentiment(article.content)
            article.tickers = self._extract_tickers(article.title + " " + article.content, symbols)
            article.relevance_score = self._calculate_relevance(article, symbols)
        
        # Filter and sort by relevance
        relevant_articles = [a for a in articles if a.relevance_score > 0.3]
        relevant_articles.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return relevant_articles[:100]  # Return top 100 most relevant articles
    
    async def _fetch_rss_news(self, rss_url: str, symbols: List[str]) -> List[NewsArticle]:
        """Fetch news from RSS feeds"""
        try:
            feed = feedparser.parse(rss_url)
            articles = []
            
            for entry in feed.entries[:20]:  # Limit to recent articles
                articles.append(NewsArticle(
                    title=entry.title,
                    content=entry.get('summary', entry.get('description', '')),
                    source=feed.feed.get('title', rss_url.split('/')[2]),
                    published_at=datetime(*entry.published_parsed[:6]),
                    url=entry.link
                ))
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching RSS feed {rss_url}: {str(e)}")
            return []
    
    async def _fetch_newsapi_articles(self, symbols: List[str]) -> List[NewsArticle]:
        """Fetch articles from NewsAPI"""
        try:
            url = "https://newsapi.org/v2/everything"
            articles = []
            
            for symbol in symbols[:5]:  # Limit API calls
                params = {
                    'q': f'"{symbol}" OR "{symbol} stock"',
                    'domains': 'reuters.com,bloomberg.com,cnbc.com,marketwatch.com',
                    'sortBy': 'publishedAt',
                    'pageSize': 10,
                    'apiKey': self.api_keys['newsapi']
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        for article_data in data.get('articles', []):
                            articles.append(NewsArticle(
                                title=article_data['title'],
                                content=article_data['description'] or '',
                                source=article_data['source']['name'],
                                published_at=datetime.fromisoformat(article_data['publishedAt'].replace('Z', '+00:00')),
                                url=article_data['url']
                            ))
                
                await asyncio.sleep(0.1)  # Rate limiting
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching NewsAPI articles: {str(e)}")
            return []
    
    def _analyze_sentiment(self, text: str) -> Tuple[float, str]:
        """Analyze sentiment of text"""
        try:
            # NLTK VADER sentiment
            vader_scores = self.sia.polarity_scores(text)
            vader_score = vader_scores['compound']
            
            # TextBlob sentiment
            blob = TextBlob(text)
            textblob_score = blob.sentiment.polarity
            
            # Combine scores
            combined_score = (vader_score + textblob_score) / 2
            
            # Determine label
            if combined_score > 0.1:
                label = 'positive'
            elif combined_score < -0.1:
                label = 'negative'
            else:
                label = 'neutral'
            
            return combined_score, label
            
        except Exception as e:
            logger.warning(f"Error analyzing sentiment: {str(e)}")
            return 0.0, 'neutral'
    
    def _extract_tickers(self, text: str, symbols: List[str]) -> List[str]:
        """Extract stock tickers mentioned in text"""
        found_tickers = []
        text_upper = text.upper()
        
        for symbol in symbols:
            if symbol.upper() in text_upper:
                found_tickers.append(symbol)
        
        return found_tickers
    
    def _calculate_relevance(self, article: NewsArticle, symbols: List[str]) -> float:
        """Calculate relevance score for article"""
        score = 0.0
        
        # Ticker mentions
        if article.tickers:
            score += len(article.tickers) * 0.4
        
        # Financial keywords
        financial_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'merger', 'acquisition',
            'dividend', 'stock', 'share', 'market', 'trading', 'investment'
        ]
        
        text_lower = (article.title + " " + article.content).lower()
        keyword_count = sum(1 for keyword in financial_keywords if keyword in text_lower)
        score += keyword_count * 0.1
        
        # Recency bonus
        hours_ago = (datetime.now(timezone.utc) - article.published_at).total_seconds() / 3600
        if hours_ago < 24:
            score += 0.2
        elif hours_ago < 168:  # 1 week
            score += 0.1
        
        return min(score, 1.0)

class EconomicDataSource(DataSource):
    """Economic indicators data source"""
    
    def __init__(self, fred_api_key: str):
        super().__init__(fred_api_key)
        self.fred_base_url = "https://api.stlouisfed.org/fred/series/observations"
    
    async def fetch_economic_indicators(self) -> List[EconomicIndicator]:
        """Fetch key economic indicators"""
        indicators = []
        
        # Key economic indicators from FRED
        fred_series = {
            'GDP': {'series_id': 'GDP', 'name': 'Gross Domestic Product', 'unit': 'Billions of Dollars'},
            'CPI': {'series_id': 'CPIAUCSL', 'name': 'Consumer Price Index', 'unit': 'Index'},
            'UNEMPLOYMENT': {'series_id': 'UNRATE', 'name': 'Unemployment Rate', 'unit': 'Percent'},
            'FEDERAL_FUNDS_RATE': {'series_id': 'FEDFUNDS', 'name': 'Federal Funds Rate', 'unit': 'Percent'},
            'TREASURY_10Y': {'series_id': 'GS10', 'name': '10-Year Treasury Rate', 'unit': 'Percent'},
            'DOLLAR_INDEX': {'series_id': 'DTWEXBGS', 'name': 'Dollar Index', 'unit': 'Index'},
            'VIX': {'series_id': 'VIXCLS', 'name': 'VIX Volatility Index', 'unit': 'Index'}
        }
        
        for key, info in fred_series.items():
            try:
                indicator = await self._fetch_fred_data(info['series_id'], info['name'], info['unit'])
                if indicator:
                    indicators.append(indicator)
            except Exception as e:
                logger.warning(f"Error fetching {key}: {str(e)}")
        
        return indicators
    
    async def _fetch_fred_data(self, series_id: str, name: str, unit: str) -> Optional[EconomicIndicator]:
        """Fetch data from FRED API"""
        try:
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json',
                'limit': 2,
                'sort_order': 'desc'
            }
            
            async with self.session.get(self.fred_base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    observations = data.get('observations', [])
                    
                    if len(observations) >= 1:
                        latest = observations[0]
                        previous = observations[1] if len(observations) > 1 else None
                        
                        current_value = float(latest['value'])
                        previous_value = float(previous['value']) if previous and previous['value'] != '.' else None
                        change = current_value - previous_value if previous_value else None
                        
                        return EconomicIndicator(
                            name=name,
                            value=current_value,
                            unit=unit,
                            date=datetime.strptime(latest['date'], '%Y-%m-%d'),
                            source='FRED',
                            previous_value=previous_value,
                            change=change
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching FRED data for {series_id}: {str(e)}")
            return None

class TechnicalAnalysisProcessor:
    """Advanced technical analysis and indicator computation"""
    
    def __init__(self):
        self.indicators = {}
        self.patterns = {}
    
    def compute_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute comprehensive technical indicators"""
        try:
            df = data.copy()
            
            # Basic price features
            df['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100
            df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100
            df['Price_Range'] = df['High'] - df['Low']
            df['Body_Size'] = abs(df['Close'] - df['Open'])
            df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
            df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
            
            # Moving averages
            periods = [5, 10, 20, 50, 100, 200]
            for period in periods:
                df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
                df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
                df[f'WMA_{period}'] = self._weighted_moving_average(df['Close'], period)
            
            # Bollinger Bands
            for period in [20, 50]:
                df = self._add_bollinger_bands(df, period)
            
            # RSI
            for period in [14, 21]:
                df[f'RSI_{period}'] = self._compute_rsi(df['Close'], period)
            
            # MACD
            df = self._add_macd(df)
            
            # Stochastic Oscillator
            df = self._add_stochastic(df)
            
            # Williams %R
            df['Williams_R'] = self._compute_williams_r(df)
            
            # Average True Range
            df['ATR'] = self._compute_atr(df)
            
            # Commodity Channel Index
            df['CCI'] = self._compute_cci(df)
            
            # Money Flow Index
            df['MFI'] = self._compute_mfi(df)
            
            # On-Balance Volume
            df['OBV'] = self._compute_obv(df)
            
            # Accumulation/Distribution Line
            df['AD_Line'] = self._compute_ad_line(df)
            
            # Chaikin Money Flow
            df['CMF'] = self._compute_cmf(df)
            
            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            df['VWAP'] = self._compute_vwap(df)
            
            # Momentum indicators
            for period in [10, 20]:
                df[f'ROC_{period}'] = self._compute_roc(df['Close'], period)
                df[f'Momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
            
            # Volatility indicators
            df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
            df['Garman_Klass'] = self._compute_garman_klass_volatility(df)
            
            # Support and Resistance
            df = self._add_support_resistance(df)
            
            # Pattern recognition
            df = self._add_candlestick_patterns(df)
            
            # Advanced indicators
            df = self._add_ichimoku_cloud(df)
            df = self._add_parabolic_sar(df)
            
            # Fibonacci retracements
            df = self._add_fibonacci_levels(df)
            
            # Elliott Wave indicators
            df = self._add_elliott_wave_features(df)
            
            logger.info(f"Computed {len([col for col in df.columns if col not in data.columns])} technical indicators")
            return df
            
        except Exception as e:
            logger.error(f"Error computing technical indicators: {str(e)}")
            return data
    
    def _weighted_moving_average(self, series: pd.Series, period: int) -> pd.Series:
        """Compute weighted moving average"""
        weights = np.arange(1, period + 1)
        return series.rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    def _add_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """Add Bollinger Bands"""
        sma = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()
        
        df[f'BB_Upper_{period}'] = sma + (std * std_dev)
        df[f'BB_Middle_{period}'] = sma
        df[f'BB_Lower_{period}'] = sma - (std * std_dev)
        df[f'BB_Width_{period}'] = (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}']) / df[f'BB_Middle_{period}']
        df[f'BB_Position_{period}'] = (df['Close'] - df[f'BB_Lower_{period}']) / (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}'])
        
        return df
    
    def _compute_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Compute Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _add_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Add MACD indicators"""
        ema_fast = df['Close'].ewm(span=fast).mean()
        ema_slow = df['Close'].ewm(span=slow).mean()
        
        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD'].ewm(span=signal).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        return df
    
    def _add_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Add Stochastic Oscillator"""
        low_min = df['Low'].rolling(window=k_period).min()
        high_max = df['High'].rolling(window=k_period).max()
        
        df['Stoch_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=d_period).mean()
        
        return df
    
    def _compute_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Williams %R"""
        high_max = df['High'].rolling(window=period).max()
        low_min = df['Low'].rolling(window=period).min()
        return -100 * ((high_max - df['Close']) / (high_max - low_min))
    
    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Average True Range"""
        high_low = df['High'] - df['Low']
        high_close_prev = abs(df['High'] - df['Close'].shift())
        low_close_prev = abs(df['Low'] - df['Close'].shift())
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def _compute_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Compute Commodity Channel Index"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (typical_price - sma_tp) / (0.015 * mad)
    
    def _compute_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Money Flow Index"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(window=period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(window=period).sum()
        
        money_ratio = positive_flow / negative_flow
        return 100 - (100 / (1 + money_ratio))
    
    def _compute_obv(self, df: pd.DataFrame) -> pd.Series:
        """Compute On-Balance Volume"""
        obv = []
        obv.append(df['Volume'].iloc[0])
        
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        return pd.Series(obv, index=df.index)
    
    def _compute_ad_line(self, df: pd.DataFrame) -> pd.Series:
        """Compute Accumulation/Distribution Line"""
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        clv = clv.fillna(0)
        ad_line = (clv * df['Volume']).cumsum()
        return ad_line
    
    def _compute_cmf(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Compute Chaikin Money Flow"""
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        clv = clv.fillna(0)
        money_flow_volume = clv * df['Volume']
        return money_flow_volume.rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()
    
    def _compute_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Compute Volume Weighted Average Price"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        return (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    def _compute_roc(self, series: pd.Series, period: int) -> pd.Series:
        """Compute Rate of Change"""
        return series.pct_change(period) * 100
    
    def _compute_garman_klass_volatility(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Compute Garman-Klass volatility estimator"""
        log_hl = np.log(df['High'] / df['Low'])
        log_co = np.log(df['Close'] / df['Open'])
        
        gk = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
        return np.sqrt(gk.rolling(window=period).mean() * 252)
    
    def _add_support_resistance(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Add support and resistance levels"""
        df['Support'] = df['Low'].rolling(window=window).min()
        df['Resistance'] = df['High'].rolling(window=window).max()
        df['Support_Distance'] = (df['Close'] - df['Support']) / df['Close']
        df['Resistance_Distance'] = (df['Resistance'] - df['Close']) / df['Close']
        
        return df
    
    def _add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern recognition"""
        # Doji
        df['Doji'] = (abs(df['Close'] - df['Open']) <= (df['High'] - df['Low']) * 0.1).astype(int)
        
        # Hammer
        body = abs(df['Close'] - df['Open'])
        lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
        upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
        
        df['Hammer'] = ((lower_shadow > 2 * body) & (upper_shadow < 0.5 * body)).astype(int)
        
        # Shooting Star
        df['Shooting_Star'] = ((upper_shadow > 2 * body) & (lower_shadow < 0.5 * body)).astype(int)
        
        # Engulfing patterns
        df['Bullish_Engulfing'] = ((df['Close'] > df['Open']) & 
                                  (df['Close'].shift() < df['Open'].shift()) &
                                  (df['Open'] < df['Close'].shift()) &
                                  (df['Close'] > df['Open'].shift())).astype(int)
        
        df['Bearish_Engulfing'] = ((df['Close'] < df['Open']) & 
                                  (df['Close'].shift() > df['Open'].shift()) &
                                  (df['Open'] > df['Close'].shift()) &
                                  (df['Close'] < df['Open'].shift())).astype(int)
        
        return df
    
    def _add_ichimoku_cloud(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Ichimoku Cloud indicators"""
        # Tenkan-sen (Conversion Line)
        high_9 = df['High'].rolling(window=9).max()
        low_9 = df['Low'].rolling(window=9).min()
        df['Ichimoku_Tenkan'] = (high_9 + low_9) / 2
        
        # Kijun-sen (Base Line)
        high_26 = df['High'].rolling(window=26).max()
        low_26 = df['Low'].rolling(window=26).min()
        df['Ichimoku_Kijun'] = (high_26 + low_26) / 2
        
        # Senkou Span A (Leading Span A)
        df['Ichimoku_Senkou_A'] = ((df['Ichimoku_Tenkan'] + df['Ichimoku_Kijun']) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        high_52 = df['High'].rolling(window=52).max()
        low_52 = df['Low'].rolling(window=52).min()
        df['Ichimoku_Senkou_B'] = ((high_52 + low_52) / 2).shift(26)
        
        # Chikou Span (Lagging Span)
        df['Ichimoku_Chikou'] = df['Close'].shift(-26)
        
        return df
    
    def _add_parabolic_sar(self, df: pd.DataFrame, af_start: float = 0.02, af_increment: float = 0.02, af_max: float = 0.2) -> pd.DataFrame:
        """Add Parabolic SAR indicator"""
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        
        sar = np.zeros(len(df))
        trend = np.zeros(len(df))
        af = np.zeros(len(df))
        ep = np.zeros(len(df))
        
        # Initialize
        sar[0] = low[0]
        trend[0] = 1  # 1 for uptrend, -1 for downtrend
        af[0] = af_start
        ep[0] = high[0]
        
        for i in range(1, len(df)):
            if trend[i-1] == 1:  # Uptrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
                
                if low[i] <= sar[i]:
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    ep[i] = low[i]
                    af[i] = af_start
                else:
                    trend[i] = 1
                    
            else:  # Downtrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
                
                if high[i] >= sar[i]:
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    ep[i] = high[i]
                    af[i] = af_start
                else:
                    trend[i] = -1
        
        df['Parabolic_SAR'] = sar
        df['SAR_Trend'] = trend
        
        return df
    
    def _add_fibonacci_levels(self, df: pd.DataFrame, period: int = 50) -> pd.DataFrame:
        """Add Fibonacci retracement levels"""
        high_max = df['High'].rolling(window=period).max()
        low_min = df['Low'].rolling(window=period).min()
        
        diff = high_max - low_min
        
        df['Fib_0'] = high_max
        df['Fib_236'] = high_max - diff * 0.236
        df['Fib_382'] = high_max - diff * 0.382
        df['Fib_500'] = high_max - diff * 0.500
        df['Fib_618'] = high_max - diff * 0.618
        df['Fib_100'] = low_min
        
        return df
    
    def _add_elliott_wave_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Elliott Wave analysis features"""
        # Simplified Elliott Wave features
        # Peak and trough detection
        peaks, _ = find_peaks(df['High'].values, distance=5)
        troughs, _ = find_peaks(-df['Low'].values, distance=5)
        
        df['Peak'] = 0
        df['Trough'] = 0
        
        df.iloc[peaks, df.columns.get_loc('Peak')] = 1
        df.iloc[troughs, df.columns.get_loc('Trough')] = 1
        
        # Wave count (simplified)
        df['Wave_Count'] = 0
        wave_count = 0
        last_peak_trough = 0
        
        for i in range(len(df)):
            if df.iloc[i]['Peak'] == 1 or df.iloc[i]['Trough'] == 1:
                if df.iloc[i]['Peak'] != last_peak_trough or df.iloc[i]['Trough'] != last_peak_trough:
                    wave_count += 1
                    last_peak_trough = 1 if df.iloc[i]['Peak'] == 1 else -1
            df.iloc[i, df.columns.get_loc('Wave_Count')] = wave_count % 5 + 1
        
        return df

class DataProcessor:
    """Main data processing orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = None
        self.mongo_client = None
        self.sql_engine = None
        self.technical_processor = TechnicalAnalysisProcessor()
        
        # Initialize connections
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize database connections"""
        try:
            # Redis connection
            if self.config.get('redis_url'):
                self.redis_client = redis.Redis.from_url(
                    self.config['redis_url'],
                    decode_responses=True
                )
            
            # MongoDB connection
            if self.config.get('mongodb_uri'):
                self.mongo_client = pymongo.MongoClient(self.config['mongodb_uri'])
            
            # SQL connection
            if self.config.get('sql_connection_string'):
                self.sql_engine = create_engine(self.config['sql_connection_string'])
            
            logger.info("Database connections initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database connections: {str(e)}")
    
    async def process_market_data(self, symbols: List[str], data_sources: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Process market data for multiple symbols"""
        results = {}
        
        # Default data sources
        if not data_sources:
            data_sources = ['yahoo', 'alpha_vantage']
        
        # Initialize data sources
        sources = {}
        if 'yahoo' in data_sources:
            sources['yahoo'] = YahooFinanceSource()
        
        if 'alpha_vantage' in data_sources and self.config.get('alpha_vantage_api_key'):
            sources['alpha_vantage'] = AlphaVantageSource(self.config['alpha_vantage_api_key'])
        
        # Process each symbol
        async def process_symbol(symbol: str) -> Tuple[str, pd.DataFrame]:
            try:
                market_data = []
                
                # Fetch from multiple sources
                for source_name, source in sources.items():
                    async with source:
                        if source_name == 'yahoo':
                            data = await source.fetch_stock_data(symbol)
                        elif source_name == 'alpha_vantage':
                            data = await source.fetch_daily_data(symbol)
                        else:
                            continue
                        
                        market_data.extend(data)
                
                if not market_data:
                    logger.warning(f"No data fetched for {symbol}")
                    return symbol, pd.DataFrame()
                
                # Convert to DataFrame
                df = pd.DataFrame([asdict(item) for item in market_data])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').drop_duplicates(['timestamp'], keep='last')
                df.set_index('timestamp', inplace=True)
                
                # Add technical indicators
                df = self.technical_processor.compute_all_indicators(df)
                
                # Cache results
                if self.redis_client:
                    cache_key = f"market_data:{symbol}"
                    self.redis_client.setex(
                        cache_key,
                        CACHE_EXPIRY,
                        df.to_json()
                    )
                
                logger.info(f"Processed market data for {symbol}: {len(df)} records")
                return symbol, df
                
            except Exception as e:
                logger.error(f"Error processing market data for {symbol}: {str(e)}")
                return symbol, pd.DataFrame()
        
        # Process symbols concurrently
        tasks = [process_symbol(symbol) for symbol in symbols]
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in completed_tasks:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
            else:
                symbol, df = result
                if not df.empty:
                    results[symbol] = df
        
        return results
    
    async def process_news_data(self, symbols: List[str]) -> List[NewsArticle]:
        """Process news data for symbols"""
        news_source = NewsDataSource(self.config.get('api_keys', {}))
        
        async with news_source:
            articles = await news_source.fetch_news(symbols)
        
        # Store in database
        if self.mongo_client:
            try:
                db = self.mongo_client[self.config.get('mongodb_db', 'stockpredictor')]
                collection = db['news']
                
                news_docs = [asdict(article) for article in articles]
                if news_docs:
                    collection.insert_many(news_docs)
                    logger.info(f"Stored {len(news_docs)} news articles")
                    
            except Exception as e:
                logger.error(f"Error storing news data: {str(e)}")
        
        return articles
    
    async def process_economic_data(self) -> List[EconomicIndicator]:
        """Process economic indicators"""
        if not self.config.get('fred_api_key'):
            logger.warning("FRED API key not configured")
            return []
        
        economic_source = EconomicDataSource(self.config['fred_api_key'])
        
        async with economic_source:
            indicators = await economic_source.fetch_economic_indicators()
        
        # Store in database
        if self.mongo_client:
            try:
                db = self.mongo_client[self.config.get('mongodb_db', 'stockpredictor')]
                collection = db['economic_indicators']
                
                indicator_docs = [asdict(indicator) for indicator in indicators]
                if indicator_docs:
                    collection.insert_many(indicator_docs)
                    logger.info(f"Stored {len(indicator_docs)} economic indicators")
                    
            except Exception as e:
                logger.error(f"Error storing economic data: {str(e)}")
        
        return indicators
    
    def clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate market data"""
        try:
            original_len = len(df)
            
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Remove rows with all NaN values
            df = df.dropna(how='all')
            
            # Handle outliers using IQR method
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col in ['open', 'high', 'low', 'close', 'volume']:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Cap outliers instead of removing them
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
            # Fill missing values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Validate price relationships
            df = df[(df['high'] >= df['low']) & 
                   (df['high'] >= df['open']) & 
                   (df['high'] >= df['close']) &
                   (df['low'] <= df['open']) & 
                   (df['low'] <= df['close'])]
            
            # Validate volume (must be positive)
            df = df[df['volume'] > 0]
            
            logger.info(f"Data cleaning completed: {original_len} -> {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            return df
    
    def feature_engineering(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Advanced feature engineering"""
        try:
            # Lag features
            for lag in [1, 2, 3, 5, 10, 20]:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
                df[f'returns_lag_{lag}'] = df['close'].pct_change().shift(lag)
            
            # Rolling statistics
            for window in [5, 10, 20, 50]:
                df[f'close_mean_{window}'] = df['close'].rolling(window=window).mean()
                df[f'close_std_{window}'] = df['close'].rolling(window=window).std()
                df[f'close_min_{window}'] = df['close'].rolling(window=window).min()
                df[f'close_max_{window}'] = df['close'].rolling(window=window).max()
                df[f'volume_mean_{window}'] = df['volume'].rolling(window=window).mean()
                df[f'volume_std_{window}'] = df['volume'].rolling(window=window).std()
            
            # Interaction features
            df['price_volume'] = df['close'] * df['volume']
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Time-based features
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            df['is_month_end'] = df.index.is_month_end.astype(int)
            df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
            
            # Cyclical encoding
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Market regime features
            df['bull_market'] = (df['close'] > df['SMA_200']).astype(int)
            df['high_volatility'] = (df['Volatility'] > df['Volatility'].quantile(0.75)).astype(int)
            df['high_volume'] = (df['volume'] > df['Volume_SMA'] * 1.5).astype(int)
            
            logger.info(f"Feature engineering completed for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            return df
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in the data"""
        try:
            # Use Isolation Forest for anomaly detection
            features = ['close', 'volume', 'high', 'low', 'open']
            feature_data = df[features].fillna(method='ffill')
            
            # Normalize features
            scaler = StandardScaler()
            feature_data_scaled = scaler.fit_transform(feature_data)
            
            # Detect anomalies
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            anomaly_labels = iso_forest.fit_predict(feature_data_scaled)
            
            df['anomaly'] = anomaly_labels
            df['anomaly_score'] = iso_forest.decision_function(feature_data_scaled)
            
            anomaly_count = (anomaly_labels == -1).sum()
            logger.info(f"Detected {anomaly_count} anomalies out of {len(df)} records")
            
            return df
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return df

async def main():
    """Main execution function"""
    config = {
        'redis_url': os.getenv('REDIS_URL', 'redis://localhost:6379'),
        'mongodb_uri': os.getenv('MONGODB_URI', 'mongodb://localhost:27017'),
        'alpha_vantage_api_key': os.getenv('ALPHA_VANTAGE_API_KEY'),
        'fred_api_key': os.getenv('FRED_API_KEY'),
        'api_keys': {
            'newsapi': os.getenv('NEWS_API_KEY')
        }
    }
    
    # Initialize processor
    processor = DataProcessor(config)
    
    # Example usage
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    try:
        # Process market data
        market_data = await processor.process_market_data(symbols)
        
        # Process news data
        news_data = await processor.process_news_data(symbols)
        
        # Process economic data
        economic_data = await processor.process_economic_data()
        
        # Output results
        result = {
            'market_data_processed': len(market_data),
            'news_articles_processed': len(news_data),
            'economic_indicators_processed': len(economic_data),
            'timestamp': datetime.now().isoformat()
        }
        
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        error_result = {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        print(json.dumps(error_result, indent=2))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # CLI mode for specific operations
        operation = sys.argv[1]
        if operation == "process":
            asyncio.run(main())
        else:
            print("Usage: python data_processor.py process")
    else:
        # Run main processing
        asyncio.run(main())