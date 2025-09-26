#!/usr/bin/env python3
"""
Enhanced Technical Analysis Module with Investment Strategies and Watchlist Support
Performs strategy-based technical analysis and sends comprehensive email reports
"""

__version__ = "1.5.0"
__copyright__ = "Copyright 2025, Richard D. Wissinger"
__author__ = "Richard D. Wissinger"
__email__ = "rick.wissinger@gmail.com"
__license__ = "MIT"
__status__ = "Production"

import os
import sys
import threading
import math
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass, field
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tabulate import tabulate

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DB_PATH, MAX_WORKERS
from logger import setup_logger, PerformanceTimer, performance_monitor
from storage import get_database, get_latest_snapshot_date
from blacklist import get_blacklist
from watchlist_manager import (
    WatchlistManager, EmailNotifier, Watchlist,
    get_watchlist_manager, get_email_notifier
)

# Import strategies
from strategies import (
    ALL_STRATEGIES, GROWTH_STRATEGIES, VALUE_STRATEGIES,
    DIVIDEND_STRATEGIES, MOMENTUM_STRATEGIES, SECTOR_STRATEGIES,
    RISK_STRATEGIES, MARKET_STRATEGIES
)

logger = setup_logger(__name__)

# =============================================================================
# THREAD-SAFE RATE LIMITING
# =============================================================================

class YahooRateLimiter:
    """Thread-safe rate limiter for Yahoo Finance API calls"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for global rate limiter"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.last_call_time = 0.0
        self.empty_response_count = 0
        self.call_lock = threading.RLock()
        self.delay_between_calls = 0.10
        self.max_empty_responses = 3
        self.empty_response_delay_multiplier = 1.5
        self.consecutive_429s = 0
        self.max_retries = 3
        self.retry_delay = 2.0
        self.max_retry_delay = 60.0
        self.timeout = 30
        self._initialized = True
    
    def wait_if_needed(self):
        """Apply rate limiting between API calls"""
        with self.call_lock:
            now = time.time()
            elapsed = now - self.last_call_time
            min_delay = self.delay_between_calls
            
            if self.consecutive_429s > 0:
                min_delay = min(self.delay_between_calls * (2 ** self.consecutive_429s), 
                              self.max_retry_delay)
            
            if elapsed < min_delay:
                time.sleep(min_delay - elapsed)
            
            self.last_call_time = time.time()
    
    def register_empty_response(self) -> bool:
        """Register empty response and check if backoff needed"""
        with self.call_lock:
            self.empty_response_count += 1
            
            if self.empty_response_count >= self.max_empty_responses:
                delay = self.delay_between_calls * self.empty_response_delay_multiplier
                logger.info(f"Multiple empty responses detected, applying {delay:.2f}s backoff")
                time.sleep(delay)
                self.empty_response_count = 0
                return True
            
            return False
    
    def reset_empty_responses(self):
        """Reset empty response counter"""
        with self.call_lock:
            self.empty_response_count = 0
    
    def register_429(self):
        """Register a rate limit error"""
        with self.call_lock:
            self.consecutive_429s += 1
            logger.warning(f"Rate limit hit #{self.consecutive_429s}")
    
    def register_success(self):
        """Register successful call"""
        with self.call_lock:
            if self.consecutive_429s > 0:
                self.consecutive_429s = max(0, self.consecutive_429s - 1)
            self.reset_empty_responses()

# Global rate limiter instance
_rate_limiter = YahooRateLimiter()

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TechnicalIndicators:
    """Technical analysis results for a stock"""
    symbol: str
    name: str = ""
    sector: str = ""
    rsi: Optional[float] = None
    roe_pct: Optional[float] = None
    eps_yoy_pct: Optional[float] = None
    period_return_pct: Optional[float] = None
    volatility_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    three_up_pattern: bool = False
    cup_handle_valid: bool = False
    cup_handle_end_phase: bool = False
    pivot_price: Optional[float] = None
    current_price: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    market_cap: Optional[float] = None
    recommendation: str = "HOLD"
    strategy_scores: Dict[str, float] = field(default_factory=dict)
    best_strategy: Optional[str] = None
    best_score: Optional[float] = None
    analysis_date: str = ""

@dataclass
class StrategyMatch:
    """Represents a match between a stock and strategy"""
    symbol: str
    strategy_name: str
    score: float  # 0-100 score
    rsi_match: float
    roe_match: float
    eps_match: float
    recommendation: str
    details: Dict[str, Any]

# =============================================================================
# YAHOO FINANCE SYMBOL HANDLING
# =============================================================================

YAHOO_SYMBOL_CORRECTIONS = {
    'BRK.A': 'BRK-A', 'BRK.B': 'BRK-B', 'BRK/A': 'BRK-A', 'BRK/B': 'BRK-B',
    'BRKA': 'BRK-A', 'BRKB': 'BRK-B', 'CRD/A': 'CRD-A', 'CRD/B': 'CRD-B',
    'GOOG': 'GOOGL', 'BF.A': 'BF-A', 'BF.B': 'BF-B', 'LEN.B': 'LEN-B',
    'NWS': 'NWSA', 'SPDR': 'SPY', 'BTC': 'BTC-USD', 'ETH': 'ETH-USD',
}

def correct_symbol_for_yahoo(symbol: str) -> Optional[str]:
    """Correct a symbol for Yahoo Finance format"""
    import re
    symbol = symbol.upper().strip()
    
    if symbol in YAHOO_SYMBOL_CORRECTIONS:
        return YAHOO_SYMBOL_CORRECTIONS[symbol]
    
    # Pattern corrections
    patterns = [
        (r'^([A-Z]+)\.([A-Z])$', r'\1-\2'),
        (r'^([A-Z]+)/([A-Z])$', r'\1-\2'),
    ]
    
    for pattern, replacement in patterns:
        corrected = re.sub(pattern, replacement, symbol)
        if corrected != symbol:
            return corrected
    
    return symbol

# =============================================================================
# TECHNICAL INDICATORS (from original)
# =============================================================================

from functools import lru_cache

@lru_cache(maxsize=256)
def fetch_adj_close_series(ticker: str, period: str = "1y", end: Optional[str] = None) -> Optional[pd.Series]:
    """Fetch adjusted close price series with thread-safe rate limiting"""
    logger.debug(f"fetch_adj_close_series(ticker={ticker}, period={period}, end={end})")
    
    for attempt in range(_rate_limiter.max_retries):
        try:
            _rate_limiter.wait_if_needed()
            
            if end:
                try:
                    asof = pd.to_datetime(end).tz_localize(None)
                except:
                    return None
                start = (asof - pd.Timedelta(days=60)).date().isoformat()
                df = yf.download(
                    ticker,
                    start=start,
                    end=(asof + pd.Timedelta(days=1)).date().isoformat(),
                    progress=False,
                    auto_adjust=False,
                    timeout=_rate_limiter.timeout
                )
            else:
                df = yf.download(
                    ticker, 
                    period=period, 
                    progress=False, 
                    auto_adjust=False,
                    timeout=_rate_limiter.timeout
                )

            if df is None or df.empty:
                if _rate_limiter.register_empty_response():
                    continue
                logger.warning(f"No price data returned for {ticker}")
                if attempt < _rate_limiter.max_retries - 1:
                    time.sleep(_rate_limiter.retry_delay * (2 ** attempt))
                continue
            
            _rate_limiter.register_success()
            
            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            s = df[col]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s = pd.to_numeric(s, errors="coerce").dropna()
            s.index = pd.to_datetime(s.index)
            return s if not s.empty else None
            
        except Exception as e:
            error_str = str(e)
            if '429' in error_str or 'rate' in error_str.lower():
                _rate_limiter.register_429()
            
            logger.warning(f"Error fetching price series for {ticker}: {e}")
            if attempt < _rate_limiter.max_retries - 1:
                time.sleep(min(_rate_limiter.retry_delay * (2 ** attempt), 
                             _rate_limiter.max_retry_delay))
    
    return None

def rsi_from_series(price: pd.Series, period: int = 14) -> Optional[float]:
    """Compute the Relative Strength Index (RSI)"""
    if price is None or len(price) <= period:
        return None
    price = pd.to_numeric(price, errors="coerce").dropna()
    if len(price) <= period:
        return None
    delta = price.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return float(val) if np.isfinite(val) else None

def period_return_pct(price: pd.Series) -> Optional[float]:
    """Compute the simple percent return over the provided series window"""
    if price is None or len(price) < 2:
        return None
    try:
        start = float(price.iloc[0])
        end = float(price.iloc[-1])
        if start == 0:
            return None
        return round(((end - start) / abs(start)) * 100.0, 2)
    except Exception:
        return None

def annualized_volatility_pct(price: pd.Series) -> Optional[float]:
    """Estimate annualized volatility from daily log returns"""
    if price is None or len(price) < 2:
        return None
    try:
        rets = np.log(price / price.shift(1)).dropna()
        vol = float(rets.std() * np.sqrt(252) * 100.0)
        return round(vol, 2)
    except Exception:
        return None

def max_drawdown_pct(price: pd.Series) -> Optional[float]:
    """Compute the maximum drawdown over the window"""
    if price is None or len(price) < 2:
        return None
    try:
        cum_max = price.cummax()
        drawdown = (price / cum_max - 1.0) * 100.0
        return round(float(drawdown.min()), 2)
    except Exception:
        return None

def get_eps_yoy_percent(ticker: str) -> Optional[float]:
    """Compute EPS year-over-year growth percent with rate limiting"""
    _rate_limiter.wait_if_needed()
    tk = yf.Ticker(ticker)
    
    try:
        # Try quarterly financials first
        q_fin = tk.quarterly_financials
        if q_fin is not None and not q_fin.empty:
            # Look for EPS rows
            for idx in q_fin.index:
                if 'EPS' in str(idx).upper() or 'EARNINGS' in str(idx).upper():
                    eps_series = q_fin.loc[idx]
                    if len(eps_series) >= 5:
                        latest = eps_series.iloc[0]
                        year_ago = eps_series.iloc[4]
                        if pd.notna(year_ago) and year_ago != 0:
                            return round(((latest - year_ago) / abs(year_ago)) * 100.0, 2)
        
        # Fallback to info
        info = tk.info
        if info:
            earnings_growth = info.get('earningsGrowth')
            if earnings_growth:
                return round(earnings_growth * 100, 2)
    except:
        pass
    
    return None

def get_roe_percent_ttm(ticker: str) -> Optional[float]:
    """Estimate Return on Equity (TTM %) with rate limiting"""
    _rate_limiter.wait_if_needed()
    tk = yf.Ticker(ticker)
    
    try:
        info = tk.info
        if info:
            roe = info.get('returnOnEquity')
            if roe:
                return round(roe * 100, 2)
    except:
        pass
    
    return None

# =============================================================================
# PATTERN DETECTION
# =============================================================================

def three_up_2pct(px: pd.Series, asof: Optional[str] = None) -> bool:
    """Detect three-day ≥2% step-up pattern"""
    if px is None or len(px) < 4:
        return False
    try:
        p3 = float(px.iloc[-4])
        p2 = float(px.iloc[-3])
        p1 = float(px.iloc[-2])
        p0 = float(px.iloc[-1])
        return (p2 >= 1.02 * p3) and (p1 >= 1.02 * p2) and (p0 >= 1.02 * p1)
    except:
        return False

def detect_cup_handle(
    px: pd.Series,
    cup_len_min: int = 30,
    cup_len_max: int = 200,
    cup_depth_min: float = 12.0,
    cup_depth_max: float = 50.0,
    handle_len_min: int = 5,
    handle_len_max: int = 30,
    handle_depth_max: float = 15.0,
    rim_tolerance: float = 0.08,
    handle_end_band: float = 0.05,
) -> Dict[str, Any]:
    """Heuristic cup & handle finder"""
    out = {
        "CupHandle_Valid": False,
        "CupHandle_EndPhase": False,
        "PivotPrice": None,
    }
    
    if px is None or len(px) < (cup_len_min + handle_len_min + 5):
        return out
    
    # Simplified cup & handle detection
    try:
        closes = px.copy()
        ema3 = closes.ewm(span=3, adjust=False).mean()
        n = len(ema3)
        
        # Find potential cup patterns
        for i in range(cup_len_min, n - handle_len_min):
            window = ema3.iloc[i-cup_len_min:i]
            if len(window) < cup_len_min:
                continue
                
            # Check if we have a cup shape
            high_point = window.max()
            low_point = window.min()
            depth_pct = ((high_point - low_point) / high_point) * 100
            
            if cup_depth_min <= depth_pct <= cup_depth_max:
                # Check for handle
                handle = ema3.iloc[i:min(i+handle_len_max, n)]
                if len(handle) >= handle_len_min:
                    handle_depth = ((handle.max() - handle.min()) / handle.max()) * 100
                    if handle_depth <= handle_depth_max:
                        out["CupHandle_Valid"] = True
                        out["PivotPrice"] = round(high_point, 2)
                        
                        # Check if we're at the end of handle
                        last_price = ema3.iloc[-1]
                        if abs(last_price - high_point) / high_point <= handle_end_band:
                            out["CupHandle_EndPhase"] = True
                        break
    except:
        pass
    
    return out

# =============================================================================
# BASELINE SCREENING
# =============================================================================

def check_baseline_screening(indicators: TechnicalIndicators, config: Dict[str, Any]) -> Tuple[bool, Dict[str, str]]:
    """
    Check if stock passes baseline screening from .env
    Returns: (passes, details)
    """
    # Get baseline criteria from .env or config
    rsi_min = float(os.getenv('RSI_MIN', config.get('rsi_min', 40.0)))
    rsi_max = float(os.getenv('RSI_MAX', config.get('rsi_max', 70.0)))
    roe_min = float(os.getenv('ROE_MIN', config.get('roe_min', 10.0)))
    eps_yoy_min = float(os.getenv('EPS_YOY_MIN', config.get('eps_yoy_min', 0.0)))
    
    details = {}
    passes = True
    
    # Check RSI range
    if indicators.rsi is not None:
        if indicators.rsi < rsi_min:
            details['rsi'] = f"FAIL (RSI {indicators.rsi:.1f} < {rsi_min})"
            passes = False
        elif indicators.rsi > rsi_max:
            details['rsi'] = f"FAIL (RSI {indicators.rsi:.1f} > {rsi_max})"
            passes = False
        else:
            details['rsi'] = f"PASS ({indicators.rsi:.1f})"
    else:
        details['rsi'] = "N/A"
    
    # Check ROE minimum
    if indicators.roe_pct is not None:
        if indicators.roe_pct < roe_min:
            details['roe'] = f"FAIL (ROE {indicators.roe_pct:.1f}% < {roe_min}%)"
            passes = False
        else:
            details['roe'] = f"PASS ({indicators.roe_pct:.1f}%)"
    else:
        details['roe'] = "N/A"
    
    # Check EPS YoY minimum
    if indicators.eps_yoy_pct is not None:
        if indicators.eps_yoy_pct < eps_yoy_min:
            details['eps'] = f"FAIL (EPS {indicators.eps_yoy_pct:.1f}% < {eps_yoy_min}%)"
            passes = False
        else:
            details['eps'] = f"PASS ({indicators.eps_yoy_pct:.1f}%)"
    else:
        details['eps'] = "N/A"
    
    return passes, details



# =============================================================================
# STRATEGY SCORING
# =============================================================================

def calculate_strategy_score(indicators: TechnicalIndicators, 
                            strategy: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """
    Calculate how well a stock matches a strategy
    Returns: (overall_score, component_scores)
    """
    criteria = strategy['criteria']  # (RSI_target, ROE_target, EPS_target)
    tolerance = strategy['tolerance']  # (RSI_tol, ROE_tol, EPS_tol)
    
    # Initialize scores
    scores = {
        'rsi_score': 0.0,
        'roe_score': 0.0,
        'eps_score': 0.0
    }
    
    # RSI Score (33.33% weight)
    if indicators.rsi is not None:
        rsi_diff = abs(indicators.rsi - criteria[0])
        if rsi_diff <= tolerance[0]:
            # Linear scoring within tolerance
            scores['rsi_score'] = max(0, 100 * (1 - rsi_diff / tolerance[0]))
        else:
            # Exponential decay outside tolerance
            scores['rsi_score'] = max(0, 100 * math.exp(-0.1 * (rsi_diff - tolerance[0])))
    
    # ROE Score (33.33% weight)
    if indicators.roe_pct is not None:
        roe_diff = abs(indicators.roe_pct - criteria[1])
        if roe_diff <= tolerance[1]:
            scores['roe_score'] = max(0, 100 * (1 - roe_diff / tolerance[1]))
        else:
            scores['roe_score'] = max(0, 100 * math.exp(-0.1 * (roe_diff - tolerance[1])))
    
    # EPS Growth Score (33.33% weight)
    if indicators.eps_yoy_pct is not None:
        eps_diff = abs(indicators.eps_yoy_pct - criteria[2])
        if eps_diff <= tolerance[2]:
            scores['eps_score'] = max(0, 100 * (1 - eps_diff / tolerance[2]))
        else:
            scores['eps_score'] = max(0, 100 * math.exp(-0.1 * (eps_diff - tolerance[2])))
    
    # Calculate overall score (weighted average)
    weights = {
        'rsi_score': 0.33,
        'roe_score': 0.34,
        'eps_score': 0.33
    }
    
    overall_score = sum(scores[key] * weights[key] for key in scores)
    
    # Bonus points for patterns
    if indicators.three_up_pattern:
        overall_score = min(100, overall_score + 5)
    if indicators.cup_handle_valid:
        overall_score = min(100, overall_score + 5)
    if indicators.cup_handle_end_phase:
        overall_score = min(100, overall_score + 10)
    
    return overall_score, scores

def determine_strategy_recommendation(score: float, strategy_name: str) -> str:
    """
    Determine recommendation based on strategy score
    """
    if score >= 80:
        return "STRONG BUY"
    elif score >= 65:
        return "BUY"
    elif score >= 50:
        return "HOLD"
    elif score >= 35:
        return "WEAK HOLD"
    else:
        return "AVOID"

# =============================================================================
# ENHANCED DATA COLLECTION
# =============================================================================

@performance_monitor("collect_missing_data")
def collect_missing_technical_data(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Collect any missing technical analysis data for symbols
    Ensures all required data is in the database
    """
    db = get_database()
    missing_data = {}
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        snapshot_date = get_latest_snapshot_date()
        if not snapshot_date:
            logger.error("No snapshot data available")
            return {}
        
        for symbol in symbols:
            cursor.execute("""
                SELECT 
                    current_price, pe_ratio, dividend_yield, beta,
                    return_on_equity, return_on_assets, profit_margin,
                    revenue_growth, earnings_growth, debt_to_equity,
                    current_ratio, quick_ratio, price_to_book,
                    week_52_high, week_52_low, volume
                FROM stocks
                WHERE symbol = ? AND snapshot_date = ?
            """, (symbol, snapshot_date))
            
            row = cursor.fetchone()
            
            if row:
                missing_fields = []
                if row[0] is None:  # current_price
                    missing_fields.append('current_price')
                if row[4] is None:  # return_on_equity
                    missing_fields.append('return_on_equity')
                
                if missing_fields:
                    missing_data[symbol] = {
                        'missing_fields': missing_fields,
                        'needs_update': True
                    }
            else:
                missing_data[symbol] = {
                    'missing_fields': 'all',
                    'needs_update': True
                }
    
    if missing_data:
        logger.info(f"Collecting missing data for {len(missing_data)} symbols")
        _collect_from_yahoo(missing_data)
    
    return missing_data

def _collect_from_yahoo(missing_data: Dict[str, Dict[str, Any]]):
    """Collect missing data from Yahoo Finance and update database"""
    db = get_database()
    snapshot_date = get_latest_snapshot_date()
    
    for symbol, info in missing_data.items():
        try:
            _rate_limiter.wait_if_needed()
            
            ticker = yf.Ticker(symbol)
            ticker_info = ticker.info
            
            if not ticker_info:
                continue
            
            update_data = {}
            
            if 'current_price' in info.get('missing_fields', []):
                update_data['current_price'] = ticker_info.get('currentPrice') or ticker_info.get('regularMarketPrice')
            
            if 'return_on_equity' in info.get('missing_fields', []):
                roe = get_roe_percent_ttm(symbol)
                if roe:
                    update_data['return_on_equity'] = roe
            
            update_data['debt_to_equity'] = ticker_info.get('debtToEquity')
            update_data['current_ratio'] = ticker_info.get('currentRatio')
            update_data['earnings_growth'] = ticker_info.get('earningsGrowth')
            
            if update_data:
                _update_stock_data(symbol, snapshot_date, update_data)
                logger.debug(f"Updated {len(update_data)} fields for {symbol}")
                
        except Exception as e:
            logger.warning(f"Failed to collect data for {symbol}: {e}")

def _update_stock_data(symbol: str, snapshot_date: str, data: Dict[str, Any]):
    """Update stock data in database"""
    db = get_database()
    
    with db.get_connection() as conn:
        with db.transaction(conn) as cursor:
            update_fields = []
            values = []
            
            for field, value in data.items():
                if value is not None:
                    update_fields.append(f"{field} = ?")
                    values.append(value)
            
            if update_fields:
                values.extend([symbol, snapshot_date])
                query = f"""
                    UPDATE stocks
                    SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP
                    WHERE symbol = ? AND snapshot_date = ?
                """
                cursor.execute(query, values)

# =============================================================================
# STRATEGY-BASED WATCHLIST ANALYSIS
# =============================================================================

class StrategyWatchlistAnalyzer:
    """Analyzes watchlists using investment strategies"""
    
    def __init__(self, watchlist: Watchlist):
        self.watchlist = watchlist
        self.db = get_database()
        self.manager = get_watchlist_manager()
        self.notifier = get_email_notifier()
        self.results = []
        self.strategy_matches = []
        self.output_dir = Path(os.getenv('ANALYSIS_OUTPUT_DIR', 'analysis/reports'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    @performance_monitor("analyze_watchlist_with_strategies")
    def analyze(self) -> Dict[str, Any]:
        """Perform complete analysis on watchlist symbols using strategies"""
        logger.info(f"Starting strategy-based analysis for watchlist: {self.watchlist.name}")
        
        # Ensure we have all required data
        collect_missing_technical_data(self.watchlist.symbols)
        
        # Get analysis configuration
        config = self.watchlist.analysis_config or {}
        
        # Determine which strategies to use
        selected_strategies = self._get_selected_strategies(config)
        
        # Analyze each symbol
        self.results = self._analyze_symbols_with_strategies(selected_strategies, config)
        
        # Find best strategy matches
        self.strategy_matches = self._find_strategy_matches(selected_strategies)
        
        # Generate comprehensive summary
        summary = self._generate_strategy_summary()
        
        # Save results with strategy information
        output_files = self._save_strategy_results(summary)
        
        # Record in database
        self.manager.record_analysis(
            self.watchlist.id,
            summary,
            output_files.get('excel'),
            email_sent=False
        )
        
        return {
            'summary': summary,
            'results': self.results,
            'strategy_matches': self.strategy_matches,
            'output_files': output_files
        }
    
    def _get_selected_strategies(self, config: Dict[str, Any]) -> Dict[str, Dict]:
        """Get strategies to use for analysis"""
        # Check if specific strategies are configured
        strategy_names = config.get('strategies', [])
        
        if strategy_names:
            # Use specified strategies
            selected = {}
            for name in strategy_names:
                if name in ALL_STRATEGIES:
                    selected[name] = ALL_STRATEGIES[name]
            return selected if selected else ALL_STRATEGIES
        
        # Use strategy category if specified
        category = config.get('strategy_category', '').upper()
        
        if category == 'GROWTH':
            return GROWTH_STRATEGIES
        elif category == 'VALUE':
            return VALUE_STRATEGIES
        elif category == 'DIVIDEND':
            return DIVIDEND_STRATEGIES
        elif category == 'MOMENTUM':
            return MOMENTUM_STRATEGIES
        elif category == 'RISK':
            return RISK_STRATEGIES
        elif category == 'MARKET':
            return MARKET_STRATEGIES
        elif category == 'SECTOR':
            return SECTOR_STRATEGIES
        else:
            # Default: use all strategies
            return ALL_STRATEGIES
    
    def _analyze_symbols_with_strategies(self, strategies: Dict[str, Dict], 
                                        config: Dict[str, Any]) -> List[TechnicalIndicators]:
        """Analyze individual symbols and score against strategies"""
        results = []
        total = len(self.watchlist.symbols)
        
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, 4)) as executor:
            futures = {}
            
            for symbol in self.watchlist.symbols:
                future = executor.submit(
                    self._analyze_single_symbol_with_strategies,
                    symbol, strategies, config
                )
                futures[future] = symbol
            
            with tqdm(total=total, desc=f"Analyzing {self.watchlist.name}") as pbar:
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        result = future.result(timeout=60)
                        if result:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Failed to analyze {symbol}: {e}")
                    finally:
                        pbar.update(1)
        
        return results
    
    def _analyze_single_symbol_with_strategies(self, symbol: str, 
                                              strategies: Dict[str, Dict],
                                              config: Dict[str, Any]) -> Optional[TechnicalIndicators]:
        """Analyze a single symbol and score against all strategies"""
        try:
            # Get data from database
            db_data = self._get_stock_data_from_db(symbol)
            
            if not db_data:
                logger.warning(f"No database data for {symbol}")
                return None
            
            # Correct symbol for Yahoo Finance
            corrected = correct_symbol_for_yahoo(symbol)
            if not corrected:
                return None
            
            # Get price series for technical indicators
            px = fetch_adj_close_series(
                corrected,
                period=config.get('period', '1y'),
                end=config.get('asof')
            )
            
            # Create result object
            result = TechnicalIndicators(
                symbol=symbol,
                name=db_data.get('name', ''),
                sector=db_data.get('sector', ''),
                analysis_date=datetime.now().strftime("%Y-%m-%d")
            )
            
            # Calculate technical indicators
            if px is not None and len(px) > 14:
                result.rsi = rsi_from_series(px, period=config.get('rsi_window', 14))
                result.period_return_pct = period_return_pct(px)
                result.volatility_pct = annualized_volatility_pct(px)
                result.max_drawdown_pct = max_drawdown_pct(px)
                
                # Pattern detection
                if config.get('detect_three_up', True):
                    result.three_up_pattern = three_up_2pct(px, config.get('asof'))
                
                if config.get('detect_cup_handle', True):
                    cup_params = config.get('cup_params', {})
                    cup_result = detect_cup_handle(px, **cup_params)
                    result.cup_handle_valid = cup_result.get("CupHandle_Valid", False)
                    result.cup_handle_end_phase = cup_result.get("CupHandle_EndPhase", False)
                    result.pivot_price = cup_result.get("PivotPrice")
            
            # Use database data for fundamentals
            result.roe_pct = db_data.get('return_on_equity')
            
            # Get EPS YoY
            if db_data.get('earnings_growth') is not None:
                result.eps_yoy_pct = db_data['earnings_growth'] * 100
            else:
                result.eps_yoy_pct = get_eps_yoy_percent(corrected)
            
            # Additional data
            result.current_price = db_data.get('current_price')
            result.pe_ratio = db_data.get('pe_ratio')
            result.dividend_yield = db_data.get('dividend_yield')
            result.market_cap = db_data.get('market_cap')
            
            # Score against all strategies
            best_score = 0
            best_strategy = None
            
            for strategy_name, strategy_config in strategies.items():
                score, components = calculate_strategy_score(result, strategy_config)
                result.strategy_scores[strategy_name] = score
                
                if score > best_score:
                    best_score = score
                    best_strategy = strategy_name
            
            result.best_strategy = best_strategy
            result.best_score = best_score
            
            # Determine recommendation based on best strategy
            if best_strategy and best_score:
                result.recommendation = determine_strategy_recommendation(best_score, best_strategy)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def _get_stock_data_from_db(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get stock data from database"""
        snapshot_date = get_latest_snapshot_date()
        if not snapshot_date:
            return None
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    symbol, name, exchange, sector, industry,
                    market_cap, current_price, pe_ratio, dividend_yield, beta,
                    return_on_equity, return_on_assets, profit_margin,
                    revenue_growth, earnings_growth, debt_to_equity,
                    current_ratio, quick_ratio, price_to_book,
                    week_52_high, week_52_low, volume
                FROM stocks
                WHERE symbol = ? AND snapshot_date = ?
            """, (symbol, snapshot_date))
            
            row = cursor.fetchone()
            
            if row:
                return {
                    'symbol': row[0],
                    'name': row[1],
                    'exchange': row[2],
                    'sector': row[3],
                    'industry': row[4],
                    'market_cap': row[5],
                    'current_price': row[6],
                    'pe_ratio': row[7],
                    'dividend_yield': row[8],
                    'beta': row[9],
                    'return_on_equity': row[10],
                    'return_on_assets': row[11],
                    'profit_margin': row[12],
                    'revenue_growth': row[13],
                    'earnings_growth': row[14],
                    'debt_to_equity': row[15],
                    'current_ratio': row[16],
                    'quick_ratio': row[17],
                    'price_to_book': row[18],
                    'week_52_high': row[19],
                    'week_52_low': row[20],
                    'volume': row[21]
                }
            
            return None
    
    def _find_strategy_matches(self, strategies: Dict[str, Dict]) -> List[StrategyMatch]:
        """Find best strategy matches across all symbols"""
        matches = []
        
        for result in self.results:
            if not result.strategy_scores:
                continue
            
            for strategy_name, score in result.strategy_scores.items():
                if score >= 50:  # Only include decent matches
                    match = StrategyMatch(
                        symbol=result.symbol,
                        strategy_name=strategy_name,
                        score=score,
                        rsi_match=result.rsi or 0,
                        roe_match=result.roe_pct or 0,
                        eps_match=result.eps_yoy_pct or 0,
                        recommendation=determine_strategy_recommendation(score, strategy_name),
                        details={
                            'name': result.name,
                            'sector': result.sector,
                            'current_price': result.current_price,
                            'pe_ratio': result.pe_ratio,
                            'patterns': {
                                'three_up': result.three_up_pattern,
                                'cup_handle': result.cup_handle_valid
                            }
                        }
                    )
                    matches.append(match)
        
        # Sort by score
        matches.sort(key=lambda x: x.score, reverse=True)
        
        return matches
    
    def _generate_strategy_summary(self) -> Dict[str, Any]:
        """Generate comprehensive strategy-based analysis summary"""
        if not self.results:
            return {
                'total_analyzed': 0,
                'strategies_used': 0,
                'best_matches': []
            }
        
        # Get unique strategies used
        all_strategies_used = set()
        for result in self.results:
            all_strategies_used.update(result.strategy_scores.keys())
        
        # Group results by recommendation
        recommendations = {}
        for result in self.results:
            rec = result.recommendation
            if rec not in recommendations:
                recommendations[rec] = []
            recommendations[rec].append(result)
        
        # Find top matches for each strategy
        strategy_top_picks = {}
        for strategy_name in all_strategies_used:
            strategy_top_picks[strategy_name] = []
            
            for result in self.results:
                score = result.strategy_scores.get(strategy_name, 0)
                if score >= 65:  # Good matches only
                    strategy_top_picks[strategy_name].append({
                        'symbol': result.symbol,
                        'name': result.name,
                        'score': score,
                        'recommendation': determine_strategy_recommendation(score, strategy_name),
                        'metrics': {
                            'rsi': result.rsi,
                            'roe': result.roe_pct,
                            'eps_yoy': result.eps_yoy_pct,
                            'return': result.period_return_pct
                        }
                    })
            
            # Sort and limit to top 5
            strategy_top_picks[strategy_name].sort(key=lambda x: x['score'], reverse=True)
            strategy_top_picks[strategy_name] = strategy_top_picks[strategy_name][:5]
        
        # Get overall best matches
        best_matches = []
        for result in self.results:
            if result.best_score and result.best_score >= 70:
                best_matches.append({
                    'symbol': result.symbol,
                    'name': result.name,
                    'strategy': result.best_strategy,
                    'score': result.best_score,
                    'recommendation': result.recommendation,
                    'current_price': result.current_price,
                    'patterns': {
                        'three_up': result.three_up_pattern,
                        'cup_handle': result.cup_handle_valid
                    }
                })
        
        best_matches.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'total_analyzed': len(self.results),
            'total_symbols': len(self.watchlist.symbols),
            'strategies_used': len(all_strategies_used),
            'recommendations': {
                'STRONG BUY': len(recommendations.get('STRONG BUY', [])),
                'BUY': len(recommendations.get('BUY', [])),
                'HOLD': len(recommendations.get('HOLD', [])),
                'WEAK HOLD': len(recommendations.get('WEAK HOLD', [])),
                'AVOID': len(recommendations.get('AVOID', []))
            },
            'strategy_top_picks': strategy_top_picks,
            'best_matches': best_matches[:20],  # Top 20
            'pattern_counts': {
                'three_up': sum(1 for r in self.results if r.three_up_pattern),
                'cup_handle': sum(1 for r in self.results if r.cup_handle_valid)
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def _save_strategy_results(self, summary: Dict[str, Any]) -> Dict[str, str]:
        """Save analysis results with strategy information"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        watchlist_name = self.watchlist.name.replace(' ', '_').lower()
        output_files = {}
        
        if self.results:
            # Convert results to DataFrame
            df = pd.DataFrame([vars(r) for r in self.results])
            
            # Add strategy score columns
            for result in self.results:
                for strategy_name, score in result.strategy_scores.items():
                    col_name = f"score_{strategy_name.replace(' ', '_').lower()}"
                    if col_name not in df.columns:
                        df[col_name] = 0
                    idx = df[df['symbol'] == result.symbol].index[0]
                    df.at[idx, col_name] = score
            
            # Sort by best score
            df = df.sort_values('best_score', ascending=False)
            
            # Save CSV
            csv_file = self.output_dir / f"strategy_{watchlist_name}_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            output_files['csv'] = str(csv_file)
            
            # Save Excel with multiple sheets
            excel_file = self.output_dir / f"strategy_{watchlist_name}_{timestamp}.xlsx"
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # Overview sheet
                df.to_excel(writer, sheet_name='All Stocks', index=False)
                
                # Strong Buy sheet
                strong_buy = df[df['recommendation'] == 'STRONG BUY']
                if not strong_buy.empty:
                    strong_buy.to_excel(writer, sheet_name='Strong Buy', index=False)
                
                # Buy sheet
                buy = df[df['recommendation'] == 'BUY']
                if not buy.empty:
                    buy.to_excel(writer, sheet_name='Buy', index=False)
                
                # Strategy matches sheet
                if self.strategy_matches:
                    matches_df = pd.DataFrame([
                        {
                            'Symbol': m.symbol,
                            'Strategy': m.strategy_name,
                            'Score': m.score,
                            'Recommendation': m.recommendation,
                            'Name': m.details['name'],
                            'Sector': m.details['sector'],
                            'Price': m.details['current_price']
                        }
                        for m in self.strategy_matches[:50]  # Top 50
                    ])
                    matches_df.to_excel(writer, sheet_name='Strategy Matches', index=False)
                
                # Pattern sheet
                patterns = df[(df['three_up_pattern'] == True) | (df['cup_handle_valid'] == True)]
                if not patterns.empty:
                    patterns.to_excel(writer, sheet_name='Pattern Matches', index=False)
                
                # Summary sheet
                summary_data = []
                for strategy_name, picks in summary.get('strategy_top_picks', {}).items():
                    for pick in picks[:3]:  # Top 3 per strategy
                        summary_data.append({
                            'Strategy': strategy_name,
                            'Symbol': pick['symbol'],
                            'Score': pick['score'],
                            'Recommendation': pick['recommendation']
                        })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Strategy Summary', index=False)
            
            output_files['excel'] = str(excel_file)
            
            # Save JSON
            json_file = self.output_dir / f"strategy_{watchlist_name}_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump({
                    'watchlist': self.watchlist.to_dict(),
                    'summary': summary,
                    'results': [vars(r) for r in self.results],
                    'strategy_matches': [
                        {
                            'symbol': m.symbol,
                            'strategy': m.strategy_name,
                            'score': m.score,
                            'recommendation': m.recommendation,
                            'details': m.details
                        }
                        for m in self.strategy_matches[:100]  # Top 100
                    ]
                }, f, indent=2, default=str)
            output_files['json'] = str(json_file)
        
        # Generate strategy report
        report_file = self.output_dir / f"strategy_report_{watchlist_name}_{timestamp}.txt"
        self._write_strategy_report(report_file, summary)
        output_files['report'] = str(report_file)
        
        logger.info(f"Saved strategy analysis results for '{self.watchlist.name}'")
        
        return output_files
    
    def _write_strategy_report(self, file_path: Path, summary: Dict[str, Any]):
        """Write detailed strategy analysis report"""
        with open(file_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"STRATEGY-BASED ANALYSIS REPORT: {self.watchlist.name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Description: {self.watchlist.description or 'N/A'}\n")
            f.write(f"Schedule: {self.watchlist.analysis_schedule}\n")
            f.write(f"Recipients: {', '.join(self.watchlist.email_recipients) or 'None'}\n\n")
            
            # Summary statistics
            f.write("ANALYSIS SUMMARY:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Symbols: {summary['total_symbols']}\n")
            f.write(f"Analyzed: {summary['total_analyzed']}\n")
            f.write(f"Strategies Used: {summary['strategies_used']}\n\n")
            
            # Recommendation breakdown
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 40 + "\n")
            for rec, count in summary['recommendations'].items():
                f.write(f"  {rec:12} {count:3} stocks\n")
            f.write("\n")
            
            # Pattern summary
            f.write("PATTERN DETECTION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Three-Up Patterns:  {summary['pattern_counts']['three_up']}\n")
            f.write(f"  Cup & Handle:       {summary['pattern_counts']['cup_handle']}\n\n")
            
            # Best overall matches
            if summary['best_matches']:
                f.write("TOP STRATEGY MATCHES:\n")
                f.write("-" * 40 + "\n")
                for match in summary['best_matches'][:10]:
                    f.write(f"  {match['symbol']:6} Score: {match['score']:.1f} "
                           f"Strategy: {match['strategy'][:20]:20} "
                           f"{match['recommendation']}\n")
                f.write("\n")
            
            # Top picks by strategy
            f.write("TOP PICKS BY STRATEGY:\n")
            f.write("=" * 40 + "\n")
            for strategy_name, picks in summary['strategy_top_picks'].items():
                if picks:
                    f.write(f"\n{strategy_name}:\n")
                    f.write("-" * 30 + "\n")
                    for pick in picks[:3]:
                        f.write(f"  {pick['symbol']:6} {pick['name'][:20]:20} "
                               f"Score: {pick['score']:.1f} "
                               f"{pick['recommendation']}\n")

# =============================================================================
# ENHANCED EMAIL NOTIFIER
# =============================================================================

class EnhancedEmailNotifier(EmailNotifier):
    """Enhanced email notifier with strategy reporting"""
    
    def send_strategy_analysis_report(self, watchlist: Watchlist, 
                                     report_files: Dict[str, str],
                                     summary: Dict[str, Any]) -> bool:
        """Send strategy-based analysis report via email"""
        
        if not self.config.enabled:
            logger.info("Email notifications are disabled")
            return False
        
        if not self.config.is_configured():
            logger.error("Email is not properly configured")
            return False
        
        if not watchlist.email_recipients:
            logger.info(f"No email recipients for watchlist '{watchlist.name}'")
            return False
        
        try:
            # Create email message
            msg = MIMEMultipart('mixed')
            msg['Subject'] = f"Strategy Analysis Report: {watchlist.name} - {datetime.now().strftime('%Y-%m-%d')}"
            msg['From'] = f"{self.config.from_name} <{self.config.from_email}>"
            msg['To'] = ', '.join(watchlist.email_recipients)
            
            # Create enhanced HTML body
            html_body = self._create_strategy_html_report(watchlist, summary)
            msg.attach(MIMEText(html_body, 'html'))
            
            # Attach report files
            for file_type, file_path in report_files.items():
                if file_path and Path(file_path).exists():
                    self._attach_file(msg, file_path, file_type)
            
            # Send email
            success = self._send_email(msg, watchlist.email_recipients)
            
            if success:
                logger.info(f"Sent strategy analysis report for '{watchlist.name}'")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send strategy analysis report: {e}")
            return False
    
    def _create_strategy_html_report(self, watchlist: Watchlist, 
                                    summary: Dict[str, Any]) -> str:
        """Create enhanced HTML email body with strategy analysis"""
        
        best_matches = summary.get('best_matches', [])
        recommendations = summary.get('recommendations', {})
        strategy_picks = summary.get('strategy_top_picks', {})
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f5f5f5; padding: 20px; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; }}
                h1 {{ margin: 0; font-size: 28px; }}
                h2 {{ color: #333; margin-top: 30px; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
                .content {{ padding: 30px; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0; }}
                .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #667eea; }}
                .metric-label {{ color: #666; font-size: 12px; margin-top: 5px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th {{ background: #667eea; color: white; padding: 12px; text-align: left; }}
                td {{ padding: 10px; border-bottom: 1px solid #eee; }}
                tr:hover {{ background: #f8f9fa; }}
                .strong-buy {{ background: #28a745; color: white; padding: 3px 8px; border-radius: 4px; }}
                .buy {{ background: #5cb85c; color: white; padding: 3px 8px; border-radius: 4px; }}
                .hold {{ background: #ffc107; color: dark; padding: 3px 8px; border-radius: 4px; }}
                .avoid {{ background: #dc3545; color: white; padding: 3px 8px; border-radius: 4px; }}
                .score-high {{ color: #28a745; font-weight: bold; }}
                .score-med {{ color: #ffc107; font-weight: bold; }}
                .score-low {{ color: #dc3545; }}
                .pattern-badge {{ background: #17a2b8; color: white; padding: 2px 6px; border-radius: 3px; font-size: 11px; }}
                .footer {{ background: #f8f9fa; padding: 20px; text-align: center; border-radius: 0 0 10px 10px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Strategy Analysis Report: {watchlist.name}</h1>
                    <p style="margin: 10px 0 0 0; opacity: 0.9;">
                        {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
                    </p>
                </div>
                
                <div class="content">
                    <p><strong>Description:</strong> {watchlist.description or 'Custom watchlist analysis using multiple investment strategies'}</p>
                    
                    <h2>📈 Analysis Overview</h2>
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-value">{summary.get('total_analyzed', 0)}</div>
                            <div class="metric-label">Stocks Analyzed</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{summary.get('strategies_used', 0)}</div>
                            <div class="metric-label">Strategies Used</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{recommendations.get('STRONG BUY', 0) + recommendations.get('BUY', 0)}</div>
                            <div class="metric-label">Buy Signals</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{summary.get('pattern_counts', {}).get('three_up', 0) + summary.get('pattern_counts', {}).get('cup_handle', 0)}</div>
                            <div class="metric-label">Patterns Found</div>
                        </div>
                    </div>
                    
                    <h2>🎯 Recommendation Breakdown</h2>
                    <table>
                        <tr>
                            <th>Recommendation</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
        """
        
        total = summary.get('total_analyzed', 1) or 1
        for rec_type in ['STRONG BUY', 'BUY', 'HOLD', 'WEAK HOLD', 'AVOID']:
            count = recommendations.get(rec_type, 0)
            percentage = (count / total) * 100
            
            if rec_type == 'STRONG BUY':
                badge_class = 'strong-buy'
            elif rec_type == 'BUY':
                badge_class = 'buy'
            elif rec_type in ['HOLD', 'WEAK HOLD']:
                badge_class = 'hold'
            else:
                badge_class = 'avoid'
            
            html += f"""
                        <tr>
                            <td><span class="{badge_class}">{rec_type}</span></td>
                            <td>{count}</td>
                            <td>{percentage:.1f}%</td>
                        </tr>
            """
        
        html += """
                    </table>
                    
                    <h2>🏆 Top Strategy Matches</h2>
                    <table>
                        <tr>
                            <th>Symbol</th>
                            <th>Name</th>
                            <th>Strategy</th>
                            <th>Score</th>
                            <th>Recommendation</th>
                            <th>Price</th>
                        </tr>
        """
        
        for match in best_matches[:10]:
            score = match['score']
            if score >= 80:
                score_class = 'score-high'
            elif score >= 65:
                score_class = 'score-med'
            else:
                score_class = 'score-low'
            
            patterns = []
            if match.get('patterns', {}).get('three_up'):
                patterns.append('<span class="pattern-badge">3-Up</span>')
            if match.get('patterns', {}).get('cup_handle'):
                patterns.append('<span class="pattern-badge">C&H</span>')
            
            html += f"""
                        <tr>
                            <td><strong>{match['symbol']}</strong> {' '.join(patterns)}</td>
                            <td>{match.get('name', 'N/A')[:25]}</td>
                            <td>{match['strategy'][:20]}</td>
                            <td class="{score_class}">{score:.1f}</td>
                            <td>{match['recommendation']}</td>
                            <td>${match.get('current_price', 0):.2f}</td>
                        </tr>
            """
        
        html += """
                    </table>
        """
        
        # Add strategy-specific top picks
        if strategy_picks:
            html += """
                    <h2>📋 Top Picks by Strategy</h2>
            """
            
            for strategy_name, picks in list(strategy_picks.items())[:3]:  # Top 3 strategies
                if picks:
                    html += f"""
                    <h3 style="color: #667eea; margin-top: 20px;">{strategy_name}</h3>
                    <table style="margin-top: 10px;">
                        <tr>
                            <th>Symbol</th>
                            <th>Score</th>
                            <th>RSI</th>
                            <th>ROE %</th>
                            <th>EPS YoY %</th>
                        </tr>
                    """
                    
                    for pick in picks[:3]:  # Top 3 per strategy
                        metrics = pick.get('metrics', {})
                        html += f"""
                        <tr>
                            <td><strong>{pick['symbol']}</strong></td>
                            <td>{pick['score']:.1f}</td>
                            <td>{metrics.get('rsi', 'N/A'):.1f if metrics.get('rsi') else 'N/A'}</td>
                            <td>{metrics.get('roe', 'N/A'):.1f if metrics.get('roe') else 'N/A'}</td>
                            <td>{metrics.get('eps_yoy', 'N/A'):.1f if metrics.get('eps_yoy') else 'N/A'}</td>
                        </tr>
                        """
                    
                    html += "</table>"
        
        # Footer
        html += f"""
                </div>
                
                <div class="footer">
                    <p><strong>Disclaimer:</strong> This analysis is for informational purposes only and should not be considered as investment advice.</p>
                    <p>Generated by Stock Analysis System v{__version__} | {__copyright__}</p>
                    <p>Full detailed reports are attached to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html

# =============================================================================
# PUBLIC API WITH STRATEGY SUPPORT
# =============================================================================

@performance_monitor("analyze_watchlist_with_strategies")
def analyze_watchlist(watchlist: Watchlist, send_email: bool = True) -> bool:
    """
    Analyze a watchlist using investment strategies and optionally send email report
    """
    try:
        # Create strategy analyzer
        analyzer = StrategyWatchlistAnalyzer(watchlist)
        
        # Run analysis
        results = analyzer.analyze()
        
        # Send email if requested and recipients exist
        if send_email and watchlist.email_recipients:
            notifier = EnhancedEmailNotifier()
            email_sent = notifier.send_strategy_analysis_report(
                watchlist,
                results['output_files'],
                results['summary']
            )
            
            if email_sent:
                logger.info(f"Email report sent for watchlist '{watchlist.name}'")
                
                # Update database
                manager = get_watchlist_manager()
                manager.record_analysis(
                    watchlist.id,
                    results['summary'],
                    results['output_files'].get('excel'),
                    email_sent=True
                )
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to analyze watchlist '{watchlist.name}': {e}")
        return False

@performance_monitor("analyze_all_watchlists")
def analyze_all_watchlists(schedule: str = None, send_emails: bool = True) -> int:
    """Analyze all watchlists based on schedule"""
    manager = get_watchlist_manager()
    
    if schedule:
        watchlists = manager.get_watchlists_for_schedule(schedule)
    else:
        watchlists = manager.get_all_watchlists(active_only=True)
    
    analyzed = 0
    
    for watchlist in watchlists:
        if manager.should_analyze_watchlist(watchlist):
            logger.info(f"Analyzing watchlist '{watchlist.name}' with strategies...")
            
            success = analyze_watchlist(watchlist, send_email=send_emails)
            
            if success:
                analyzed += 1
                logger.info(f"Successfully analyzed watchlist '{watchlist.name}'")
            else:
                logger.error(f"Failed to analyze watchlist '{watchlist.name}'")
    
    logger.info(f"Analyzed {analyzed} watchlists with strategy scoring")
    return analyzed

def main():
    """Main entry point for strategy-based technical analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Strategy-Based Technical Analysis with Watchlist Support",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--watchlist", help="Name of watchlist to analyze")
    parser.add_argument("--schedule", choices=["daily", "weekly", "monthly"],
                       help="Analyze all watchlists with this schedule")
    parser.add_argument("--all", action="store_true",
                       help="Analyze all due watchlists")
    parser.add_argument("--no-email", action="store_true",
                       help="Don't send email reports")
    parser.add_argument("--strategy", help="Use specific strategy (e.g., 'Aggressive Growth')")
    parser.add_argument("--strategy-category", 
                       choices=["growth", "value", "dividend", "momentum", "risk", "market", "sector"],
                       help="Use all strategies from a category")
    
    args = parser.parse_args()
    
    # Analyze specific watchlist
    if args.watchlist:
        manager = get_watchlist_manager()
        watchlist = manager.get_watchlist(name=args.watchlist)
        
        if not watchlist:
            print(f"Watchlist '{args.watchlist}' not found")
            sys.exit(1)
        
        # Add strategy configuration if specified
        if args.strategy:
            watchlist.analysis_config = watchlist.analysis_config or {}
            watchlist.analysis_config['strategies'] = [args.strategy]
        elif args.strategy_category:
            watchlist.analysis_config = watchlist.analysis_config or {}
            watchlist.analysis_config['strategy_category'] = args.strategy_category
        
        success = analyze_watchlist(watchlist, send_email=not args.no_email)
        sys.exit(0 if success else 1)
    
    # Analyze by schedule
    if args.schedule:
        count = analyze_all_watchlists(
            schedule=args.schedule,
            send_emails=not args.no_email
        )
        print(f"Analyzed {count} watchlists with strategy scoring")
        sys.exit(0 if count > 0 else 1)
    
    # Analyze all due watchlists
    if args.all:
        count = analyze_all_watchlists(send_emails=not args.no_email)
        print(f"Analyzed {count} watchlists with strategy scoring")
        sys.exit(0 if count > 0 else 1)
    
    # Default: show help
    parser.print_help()

if __name__ == "__main__":
    main()