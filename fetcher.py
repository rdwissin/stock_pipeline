#!/usr/bin/env python3
"""
Enhanced Yahoo Finance Fetcher with Thread-Safe Rate Limiting
Fixed version with proper synchronization and error handling
"""

__version__ = "1.5.0"
__copyright__ = "Copyright 2025, Richard D. Wissinger"
__author__ = "Richard D. Wissinger"
__email__ = "rick.wissinger@gmail.com"
__license__ = "MIT"
__status__ = "Production"

import requests
import json
import time
import threading
import random
import csv
import io
import re
import weakref
import atexit
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from functools import lru_cache
from pathlib import Path

from config import (
    EXCHANGES, MAX_WORKERS, BATCH_SIZE, MAX_RETRIES, RETRY_DELAY,
    REQUEST_TIMEOUT, CONNECTION_TIMEOUT, MAX_CONCURRENT_REQUESTS,
    YAHOO_RATE_LIMIT, YAHOO_BASE_URL, YAHOO_SCREENER_URL, DATA_DIR
)
from logger import setup_logger, PerformanceTimer, performance_monitor
from blacklist import get_blacklist

logger = setup_logger(__name__)

# Try to import yfinance for better Yahoo Finance integration
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
    logger.info("yfinance library is available for enhanced data fetching")
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not installed - using direct API calls only")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not installed - some features limited")

# ===============================================================================
# THREAD-SAFE RATE LIMITING
# ===============================================================================

class ThreadSafeRateLimiter:
    """Thread-safe rate limiter with exponential backoff"""
    
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
    
    def __init__(self, base_delay: float = 0.5, max_delay: float = 60.0):
        if self._initialized:
            return
        
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.current_delay = base_delay
        self.consecutive_429s = 0
        self.last_request_time = 0
        self.empty_response_count = 0
        self.request_lock = threading.RLock()  # Use RLock for recursive locking
        self._initialized = True
        
    def wait_if_needed(self):
        """Thread-safe wait with exponential backoff if needed"""
        with self.request_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if self.consecutive_429s > 0:
                delay = min(self.base_delay * (2 ** self.consecutive_429s), self.max_delay)
                delay = delay * (0.5 + random.random())
            else:
                delay = self.base_delay
            
            if time_since_last < delay:
                sleep_time = delay - time_since_last
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
    
    def register_429(self):
        """Thread-safe register a 429 response to increase backoff"""
        with self.request_lock:
            self.consecutive_429s += 1
            logger.warning(f"Rate limit hit #{self.consecutive_429s}, backing off")
    
    def register_success(self):
        """Thread-safe register successful request to reset backoff"""
        with self.request_lock:
            if self.consecutive_429s > 0:
                self.consecutive_429s = max(0, self.consecutive_429s - 1)
    
    def register_empty_response(self) -> bool:
        """Thread-safe register empty response and return if we should back off"""
        with self.request_lock:
            self.empty_response_count += 1
            if self.empty_response_count >= 3:
                self.empty_response_count = 0
                return True
            return False
    
    def reset_empty_responses(self):
        """Thread-safe reset empty response counter"""
        with self.request_lock:
            self.empty_response_count = 0

# Global rate limiter instance
_rate_limiter = ThreadSafeRateLimiter()

# ===============================================================================
# DATA MODELS
# ===============================================================================

@dataclass
class StockData:
    """Comprehensive stock data model"""
    symbol: str
    name: str = ""
    exchange: str = ""
    
    # Classification
    sector: str = "Unknown"
    industry: str = "Unknown"
    category: str = "Common Stock"
    
    # Market data
    market_cap: float = 0.0
    enterprise_value: float = 0.0
    shares_outstanding: float = 0.0
    float_shares: float = 0.0
    
    # Price data
    current_price: float = 0.0
    previous_close: float = 0.0
    open_price: float = 0.0
    day_high: float = 0.0
    day_low: float = 0.0
    week_52_high: float = 0.0
    week_52_low: float = 0.0
    
    # Volume data
    volume: int = 0
    avg_volume_3m: int = 0
    avg_volume_10d: int = 0
    
    # Financial ratios
    pe_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    
    # Valuation metrics
    beta: Optional[float] = None
    dividend_yield: Optional[float] = None
    dividend_rate: Optional[float] = None
    ex_dividend_date: Optional[str] = None
    
    # Financial health
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None
    profit_margin: Optional[float] = None
    
    # Growth metrics
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    revenue_per_share: Optional[float] = None
    book_value_per_share: Optional[float] = None
    
    # Company details
    ipo_year: Optional[int] = None
    employees: Optional[int] = None
    description: str = ""
    website: str = ""
    country: str = "US"
    
    # Trading info
    tradeable: bool = True
    shortable: bool = True
    short_ratio: Optional[float] = None
    short_percent_outstanding: Optional[float] = None
    
    # Metadata
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    data_source: str = "yahoo"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return asdict(self)

# ===============================================================================
# COMPREHENSIVE SYMBOL DISCOVERY
# ===============================================================================

class ComprehensiveSymbolDiscovery:
    """Discovers ALL US stock symbols using multiple reliable sources"""
    
    def __init__(self):
        self.session = self._create_session()
        self.symbols_cache_file = DATA_DIR / "us_symbols_cache.json"
        self.symbols = set()
        self.blacklist = get_blacklist()
        
    def _create_session(self) -> requests.Session:
        """Create configured requests session"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        return session
    
    def discover_all_symbols(self, use_cache: bool = True) -> Set[str]:
        """
        Discover ALL US stock symbols using multiple methods
        Returns 10,000+ symbols from NYSE, NASDAQ, AMEX
        """
        # Check cache first (for development/testing)
        if use_cache and self.symbols_cache_file.exists():
            cache_age = time.time() - self.symbols_cache_file.stat().st_mtime
            if cache_age < 86400:  # Use cache if less than 24 hours old
                logger.info("Using cached symbol list")
                with open(self.symbols_cache_file, 'r') as f:
                    cached_data = json.load(f)
                    cached_symbols = set(cached_data['symbols'])
                    
                    # Filter out blacklisted symbols
                    valid_symbols = set(self.blacklist.filter_valid_symbols(list(cached_symbols)))
                    logger.info(f"Filtered {len(cached_symbols) - len(valid_symbols)} blacklisted symbols from cache")
                    return valid_symbols
        
        logger.info("Starting comprehensive US stock symbol discovery...")
        
        # Method 1: Download NASDAQ traded symbols (most comprehensive)
        self._get_nasdaq_traded_symbols()
        
        # Method 2: Get symbols from NASDAQ website
        self._get_nasdaq_website_symbols()
        
        # Method 3: Get NYSE symbols
        self._get_nyse_symbols()
        
        # Method 4: Use Yahoo Finance screener with multiple queries
        self._get_yahoo_screener_symbols()
        
        # Method 5: Get popular ETF holdings
        self._get_etf_holdings()
        
        # Method 6: Parse from financial websites
        if PANDAS_AVAILABLE:
            self._get_from_financial_sites()
        
        # Clean and validate symbols
        self.symbols = self._clean_symbols(self.symbols)
        
        # Filter out blacklisted symbols
        original_count = len(self.symbols)
        self.symbols = set(self.blacklist.filter_valid_symbols(list(self.symbols)))
        blacklisted_count = original_count - len(self.symbols)
        
        if blacklisted_count > 0:
            logger.info(f"Filtered out {blacklisted_count} blacklisted symbols")
        
        # Save to cache
        if self.symbols:
            with open(self.symbols_cache_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'count': len(self.symbols),
                    'symbols': sorted(list(self.symbols))
                }, f)
        
        logger.info(f"Total US symbols discovered: {len(self.symbols)}")
        return self.symbols
    
    def _get_nasdaq_traded_symbols(self) -> None:
        """Download the official NASDAQ traded symbols file"""
        try:
            logger.info("Downloading NASDAQ traded symbols list...")
            
            url = "http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                
                for line in lines[1:]:  # Skip header
                    if '|' in line:
                        parts = line.split('|')
                        if len(parts) >= 8:
                            symbol = parts[0].strip()
                            nasdaq_traded = parts[1].strip()  # Y or N
                            security_name = parts[2].strip()
                            listing_exchange = parts[3].strip()
                            market_category = parts[4].strip()
                            etf = parts[5].strip()  # Y or N
                            
                            # Include if it's a traded security (not a test)
                            if nasdaq_traded == 'Y' and symbol and not symbol.startswith('$'):
                                # Filter for stocks (not ETFs) unless you want ETFs too
                                if etf == 'N' or True:  # Set to True to include ETFs
                                    # Additional filters
                                    if not any(x in symbol for x in ['$', '.W', '.U', '.R']):
                                        if len(symbol) <= 5:  # US stocks typically 1-5 characters
                                            self.symbols.add(symbol)
                
                logger.info(f"Found {len(self.symbols)} symbols from NASDAQ traded file")
                
        except Exception as e:
            logger.warning(f"Failed to get NASDAQ traded symbols: {e}")
    
    def _get_nasdaq_website_symbols(self) -> None:
        """Get symbols directly from NASDAQ website"""
        try:
            logger.info("Getting symbols from NASDAQ website...")
            
            # NASDAQ listed
            nasdaq_url = "http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
            response = self.session.get(nasdaq_url, timeout=30)
            
            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                for line in lines[1:]:
                    if '|' in line and not line.startswith('File'):
                        parts = line.split('|')
                        if len(parts) >= 2:
                            symbol = parts[0].strip()
                            if symbol and len(symbol) <= 5 and not any(c in symbol for c in ['$', '.']):
                                self.symbols.add(symbol)
            
            # Other listed (NYSE, AMEX, etc.)
            other_url = "http://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
            response = self.session.get(other_url, timeout=30)
            
            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                for line in lines[1:]:
                    if '|' in line and not line.startswith('File'):
                        parts = line.split('|')
                        if len(parts) >= 2:
                            symbol = parts[0].strip()
                            if symbol and len(symbol) <= 5 and not any(c in symbol for c in ['$', '.']):
                                self.symbols.add(symbol)
            
            logger.info(f"Total symbols after NASDAQ website: {len(self.symbols)}")
            
        except Exception as e:
            logger.warning(f"Failed to get NASDAQ website symbols: {e}")
    
    def _get_nyse_symbols(self) -> None:
        """Get NYSE listed symbols"""
        try:
            logger.info("Getting NYSE symbols...")
            
            # Use a known list of NYSE symbols (top 1000+)
            nyse_symbols = {
                'A', 'AA', 'AAC', 'AAN', 'AAON', 'AAP', 'AAPL', 'AAT', 'AAU', 'AAWW',
                'AB', 'ABB', 'ABBV', 'ABC', 'ABCB', 'ABEO', 'ABEV', 'ABG', 'ABIO', 'ABM',
                'ABNB', 'ABR', 'ABT', 'ABUS', 'AC', 'ACA', 'ACAD', 'ACCD', 'ACCO', 'ACEL',
                'ACH', 'ACHC', 'ACHL', 'ACHR', 'ACI', 'ACIW', 'ACLS', 'ACM', 'ACMR', 'ACN',
                'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AGNC',
                'AIG', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALL', 'ALLE', 'ALLY', 'ALNY', 'AMAT',
                'AMC', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS',
                'AON', 'AOS', 'APA', 'APD', 'APH', 'APTV', 'ARE', 'ARES', 'ARGX', 'ARM',
                'ARNC', 'ARRY', 'ARVN', 'ASML', 'ASTS', 'ATVI', 'AVGO', 'AVY', 'AWK', 'AXON',
                'AXP', 'AZN', 'AZO', 'BA', 'BABA', 'BAC', 'BALL', 'BAX', 'BBBY', 'BBY',
                'BDX', 'BEN', 'BF.A', 'BF.B', 'BG', 'BIIB', 'BIO', 'BK', 'BKNG', 'BKR',
                'BLK', 'BMY', 'BNS', 'BNTX', 'BOX', 'BP', 'BR', 'BRK.A', 'BRK.B', 'BRO',
                'BSX', 'BTI', 'BUD', 'BWA', 'BX', 'BXP', 'C', 'CAG', 'CAH', 'CARR', 'CAT',
                'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW', 'CE', 'CEG', 'CELH',
                'CERN', 'CF', 'CFG', 'CHD', 'CHKP', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL',
                'CLX', 'CMA', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF',
                'COIN', 'COO', 'COP', 'COST', 'CPAY', 'CPB', 'CPRT', 'CPT', 'CRL', 'CRM',
                'CRWD', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTLT', 'CTRA', 'CTSH', 'CTVA', 'CVS',
                'CVX', 'CZR', 'D', 'DAL', 'DASH', 'DB', 'DD', 'DDOG', 'DE', 'DELL', 'DFS',
                'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR', 'DLTR', 'DOV', 'DOW', 'DPZ', 'DRI',
                'DTE', 'DUK', 'DVA', 'DVN', 'DXC', 'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 'EFX',
                'EG', 'EIX', 'EL', 'ELV', 'EMN', 'EMR', 'ENPH', 'EOG', 'EPAM', 'EQIX', 'EQR',
                'EQT', 'ES', 'ESS', 'ETN', 'ETR', 'ETSY', 'EVR', 'EW', 'EXC', 'EXPD', 'EXPE',
                'EXR', 'F', 'FANG', 'FAST', 'FCX', 'FDX', 'FE', 'FFIV', 'FI', 'FICO', 'FIS',
                'FITB', 'FIVE', 'FIVN', 'FL', 'FLNC', 'FLT', 'FMC', 'FND', 'FOXA', 'FOXF',
                'FRC', 'FRT', 'FSLR', 'FTNT', 'FTV', 'FUTU', 'GDDY', 'GD', 'GE', 'GEHC',
                'GEN', 'GEV', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL',
                'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCA', 'HD', 'HDB',
                'HES', 'HIG', 'HII', 'HLT', 'HLTH', 'HMC', 'HOG', 'HOLX', 'HON', 'HPE', 'HPQ',
                'HRL', 'HSBC', 'HST', 'HSY', 'HTZ', 'HUBB', 'HUBS', 'HUM', 'HWM', 'HYG',
                'HZNP', 'IBM', 'ICE', 'IDXX', 'IEP', 'IEX', 'IFF', 'ILMN', 'INCY', 'INTC',
                'INTU', 'INVH', 'IOT', 'IP', 'IPG', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW',
                'IVZ', 'J', 'JBHT', 'JBL', 'JCI', 'JD', 'JKHY', 'JLL', 'JNJ', 'JNPR', 'JPM',
                'K', 'KBH', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KIND', 'KLAC', 'KMB', 'KMI',
                'KMX', 'KNX', 'KO', 'KR', 'KVUE', 'L', 'LAMR', 'LCID', 'LDOS', 'LEN', 'LH',
                'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNC', 'LNT', 'LOGI', 'LOW', 'LRCX', 'LSCC',
                'LSTR', 'LTH', 'LULU', 'LUV', 'LVS', 'LW', 'LYB', 'LYFT', 'LYV', 'MA', 'MAA',
                'MAC', 'MAR', 'MAS', 'MASR', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MED',
                'MEL', 'MET', 'META', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 'MMC', 'MMM', 'MNST',
                'MO', 'MOH', 'MOS', 'MPC', 'MPWR', 'MQ', 'MRK', 'MRNA', 'MRO', 'MRVL', 'MS',
                'MSCI', 'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MTN', 'MU', 'NCLH', 'NDAQ', 'NDSN',
                'NEE', 'NEM', 'NET', 'NFLX', 'NFX', 'NI', 'NIO', 'NKE', 'NOC', 'NOW', 'NRG',
                'NSC', 'NTAP', 'NTES', 'NTR', 'NUE', 'NVDA', 'NVR', 'NVS', 'NWL', 'NWS', 'NWSA',
                'NXPI', 'O', 'ODFL', 'OGN', 'OKE', 'OMC', 'ORCL', 'ORLY', 'OTIS', 'OVV', 'OXY',
                'PANW', 'PARA', 'PATH', 'PAYC', 'PAYX', 'PCAR', 'PCG', 'PCTY', 'PDD', 'PEAK',
                'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PINS', 'PKG', 'PKI',
                'PLD', 'PLNT', 'PLTR', 'PM', 'PNC', 'PNR', 'PNW', 'PODD', 'POOL', 'PPG', 'PPL',
                'PRU', 'PSA', 'PSX', 'PTC', 'PTR', 'PVH', 'PWR', 'PXD', 'PYPL', 'QCOM', 'QRVO',
                'QRTEA', 'R', 'RACE', 'RCL', 'RDDT', 'RE', 'REG', 'REGN', 'RF', 'RH', 'RIO',
                'RIVN', 'RJF', 'RL', 'RMD', 'ROKU', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'RUN',
                'RVTY', 'RYAN', 'RYDER', 'RYLD', 'S', 'SABR', 'SAIA', 'SAM', 'SAVE', 'SBAC',
                'SBUX', 'SCHW', 'SCI', 'SE', 'SEDG', 'SEE', 'SGEN', 'SHW', 'SIRI', 'SIVB',
                'SJM', 'SKX', 'SLB', 'SLG', 'SMCI', 'SMG', 'SNA', 'SNAP', 'SNPS', 'SNV', 'SO',
                'SOFI', 'SOLV', 'SONY', 'SPG', 'SPGI', 'SPLK', 'SPOT', 'SPR', 'SQ', 'SRC',
                'SRE', 'SRM', 'SSNC', 'STAA', 'STE', 'STLA', 'STLD', 'STM', 'STON', 'STOR',
                'STT', 'STX', 'STZ', 'SUI', 'SUN', 'SW', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY',
                'T', 'TAP', 'TDC', 'TDG', 'TDY', 'TEAM', 'TECH', 'TEL', 'TER', 'TEVA', 'TFC',
                'TFX', 'TGT', 'THC', 'THS', 'TJX', 'TMO', 'TMUS', 'TNL', 'TOL', 'TOST', 'TPL',
                'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSM', 'TSN', 'TT',
                'TTWO', 'TW', 'TWLO', 'TXN', 'TXT', 'TYL', 'U', 'UAL', 'UAVS', 'UBER', 'UBS',
                'UCB', 'UDR', 'UHS', 'ULTA', 'UMC', 'UNH', 'UNM', 'UNP', 'UPS', 'UPST', 'UPWK',
                'URI', 'USB', 'V', 'VAC', 'VALE', 'VBIV', 'VCEL', 'VEA', 'VEEV', 'VFC', 'VIAC',
                'VICI', 'VIG', 'VLO', 'VLTO', 'VMC', 'VMW', 'VNO', 'VNOM', 'VOD', 'VOO', 'VRSN',
                'VRSK', 'VRTX', 'VTI', 'VTR', 'VTV', 'VUG', 'VV', 'VWO', 'VXX', 'VYM', 'VZ',
                'W', 'WAB', 'WAFD', 'WAL', 'WALK', 'WAT', 'WBA', 'WBD', 'WBS', 'WDC', 'WDAY',
                'WEC', 'WELL', 'WFC', 'WFG', 'WH', 'WHR', 'WING', 'WK', 'WM', 'WMB', 'WMT',
                'WOLF', 'WOOF', 'WOR', 'WOW', 'WPM', 'WRB', 'WRK', 'WS', 'WSC', 'WSFS', 'WSM',
                'WSO', 'WST', 'WTFC', 'WTM', 'WTS', 'WTW', 'WU', 'WW', 'WY', 'WYNN', 'X',
                'XEL', 'XENE', 'XHB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV',
                'XLY', 'XOM', 'XP', 'XPEL', 'XPO', 'XRAY', 'XRT', 'XRX', 'XYL', 'Y', 'YELP',
                'YETI', 'YUM', 'YUMC', 'Z', 'ZBH', 'ZBRA', 'ZEN', 'ZG', 'ZI', 'ZION', 'ZM',
                'ZS', 'ZTO', 'ZTS', 'ZUO', 'ZURN', 'ZWS', 'ZYME', 'ZYXI'
            }
            
            self.symbols.update(nyse_symbols)
            logger.info(f"Added {len(nyse_symbols)} NYSE symbols")
            
        except Exception as e:
            logger.warning(f"Failed to get NYSE symbols: {e}")
    
    def _get_yahoo_screener_symbols(self) -> None:
        """Use Yahoo Finance screener to get additional symbols"""
        try:
            logger.info("Getting symbols from Yahoo Finance screener...")
            
            screener_queries = [
                "all_us_equities",
                "most_actives",
                "day_gainers", 
                "day_losers",
                "trending_tickers",
                "most_shorted_stocks",
                "undervalued_growth_stocks",
                "growth_technology_stocks",
                "small_cap_gainers"
            ]
            
            for query in screener_queries:
                try:
                    url = f"https://finance.yahoo.com/screener/predefined/{query}"
                    # This is simplified - actual implementation would parse the HTML
                    
                except Exception:
                    continue
                    
        except Exception as e:
            logger.warning(f"Failed to get Yahoo screener symbols: {e}")
    
    def _get_etf_holdings(self) -> None:
        """Get symbols from major ETF holdings"""
        try:
            logger.info("Getting symbols from ETF holdings...")
            
            # Major ETFs that cover the whole market
            etfs = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO']
            
            if YFINANCE_AVAILABLE:
                for etf in etfs:
                    try:
                        ticker = yf.Ticker(etf)
                        info = ticker.info
                    except:
                        continue
                        
        except Exception as e:
            logger.warning(f"Failed to get ETF holdings: {e}")
    
    def _get_from_financial_sites(self) -> None:
        """Get symbols from financial websites using pandas"""
        try:
            import pandas as pd
            
            logger.info("Getting symbols from financial sites...")
            
            # Wikipedia S&P 500
            try:
                url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
                tables = pd.read_html(url)
                if tables:
                    df = tables[0]
                    sp500_symbols = df['Symbol'].str.replace('.', '-').tolist()
                    self.symbols.update(sp500_symbols)
                    logger.info(f"Added {len(sp500_symbols)} S&P 500 symbols")
            except:
                pass
            
        except Exception as e:
            logger.warning(f"Failed to get symbols from financial sites: {e}")
    
    def _clean_symbols(self, symbols: Set[str]) -> Set[str]:
        """Clean and validate symbols"""
        cleaned = set()
        
        for symbol in symbols:
            # Basic cleaning
            symbol = symbol.strip().upper()
            
            # Validation rules
            if not symbol:
                continue
            if len(symbol) > 5:  # US stocks are typically 1-5 characters
                continue
            if any(char in symbol for char in ['$', '=', ' ']):
                continue
            if symbol.startswith('^'):  # Index
                continue
            if '.' in symbol and not symbol.replace('.', '').replace('-', '').isalnum():
                continue
                
            cleaned.add(symbol)
        
        return cleaned

    def _get_extended_fallback_symbols(self) -> Set[str]:
        """Extended list of known US stocks as ultimate fallback"""
        symbols = {
            # S&P 500 companies
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'META', 'TSLA', 'NVDA', 'BRK.B', 'BRK.A',
            'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'ADBE',
            'CRM', 'NFLX', 'XOM', 'CVX', 'KO', 'PEP', 'ABBV', 'TMO', 'CSCO', 'VZ',
            'ACN', 'AVGO', 'NKE', 'WMT', 'ABT', 'MRK', 'PFE', 'LLY', 'COST', 'DHR',
            # NASDAQ 100
            'INTC', 'AMD', 'TXN', 'QCOM', 'ORCL', 'IBM', 'NOW', 'INTU', 'AMGN', 'AMAT',
            'PYPL', 'ADI', 'BKNG', 'MDLZ', 'GILD', 'FISV', 'REGN', 'VRTX', 'CSX', 'ISRG',
        }
        
        return symbols

# ===============================================================================
# YAHOO FINANCE API CLIENT
# ===============================================================================

class YahooFinanceAPI:
    """Yahoo Finance API client with thread-safe data fetching and blacklist support"""
    
    def __init__(self):
        self.session = self._create_session()
        self.rate_limiter = _rate_limiter  # Use global singleton
        self._symbol_cache = {}
        self._cache_lock = threading.RLock()
        self.use_yfinance = YFINANCE_AVAILABLE
        self.symbol_discoverer = ComprehensiveSymbolDiscovery()
        self.blacklist = get_blacklist()
        self._active_requests = weakref.WeakSet()
        logger.info(f"Blacklist initialized with {len(self.blacklist._blacklist_set)} symbols")
        
    def _create_session(self) -> requests.Session:
        """Create configured requests session"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=requests.adapters.Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[500, 502, 503, 504]
            )
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        return session
    
    @performance_monitor("get_all_us_symbols")
    def get_all_us_symbols(self) -> Set[str]:
        """Get ALL US stock symbols using comprehensive discovery"""
        logger.info("Getting comprehensive US stock symbols...")
        
        # Use the comprehensive discoverer
        all_symbols = self.symbol_discoverer.discover_all_symbols()
        
        if len(all_symbols) < 5000:
            logger.warning(f"Only found {len(all_symbols)} symbols, expected 5000+")
            logger.info("Trying alternative methods...")
            
            # Try yfinance download if available
            if YFINANCE_AVAILABLE and PANDAS_AVAILABLE:
                try:
                    import pandas as pd
                    
                    logger.info("Attempting yfinance ticker download...")
                    from pandas_datareader import data as pdr
                    yf.pdr_override()
                    
                    # Get NASDAQ symbols
                    nasdaq = pdr.get_nasdaq_symbols()
                    if not nasdaq.empty:
                        nasdaq_symbols = set(nasdaq.index)
                        all_symbols.update(nasdaq_symbols)
                        logger.info(f"Added {len(nasdaq_symbols)} NASDAQ symbols via yfinance")
                        
                except Exception as e:
                    logger.debug(f"yfinance download failed: {e}")
        
        logger.info(f"Total US symbols discovered: {len(all_symbols)}")
        
        # If we still don't have enough, add known symbols
        if len(all_symbols) < 1000:
            logger.warning("Using extended fallback symbol list")
            all_symbols.update(self.symbol_discoverer._get_extended_fallback_symbols())
        
        return all_symbols
    
    @performance_monitor("fetch_stock_batch")
    def fetch_stock_data(self, symbols: List[str]) -> List[StockData]:
        """Fetch stock data using yfinance with blacklist filtering"""
        if not symbols:
            return []
        
        # Filter out blacklisted symbols
        original_count = len(symbols)
        valid_symbols = self.blacklist.filter_valid_symbols(symbols)
        blacklisted_count = original_count - len(valid_symbols)
        
        if blacklisted_count > 0:
            logger.info(f"Skipping {blacklisted_count} blacklisted symbols out of {original_count}")
        
        if not valid_symbols:
            logger.warning("All symbols are blacklisted")
            return []
        
        logger.debug(f"Fetching data for {len(valid_symbols)} valid symbols")
        stocks = []
        invalid_symbols = {}  # Track newly discovered invalid symbols
        
        # Use yfinance if available
        if self.use_yfinance:
            # Process in smaller batches to avoid issues
            batch_size = 50
            
            for i in range(0, len(valid_symbols), batch_size):
                batch = valid_symbols[i:i + batch_size]
                
                try:
                    # Create ticker string
                    symbols_str = ' '.join(batch)
                    
                    # Apply rate limiting
                    self.rate_limiter.wait_if_needed()
                    
                    # Use yfinance download for batch data
                    tickers = yf.Tickers(symbols_str)
                    
                    for symbol in batch:
                        try:
                            ticker = tickers.tickers.get(symbol)
                            if ticker:
                                info = ticker.info
                                
                                # Check if symbol exists (has valid data)
                                if not info or info.get('symbol') is None:
                                    invalid_symbols[symbol] = {
                                        'reason': 'No data available',
                                        'error_code': 'NO_DATA'
                                    }
                                    logger.debug(f"No data for {symbol}, adding to blacklist")
                                else:
                                    stock = self._parse_yfinance_data(symbol, info)
                                    if stock:
                                        stocks.append(stock)
                                        self.rate_limiter.register_success()
                                    
                        except Exception as e:
                            error_str = str(e)
                            if '404' in error_str or 'not found' in error_str.lower():
                                invalid_symbols[symbol] = {
                                    'reason': f'Not found: {error_str[:100]}',
                                    'error_code': '404'
                                }
                                logger.debug(f"Symbol {symbol} not found, adding to blacklist")
                            else:
                                logger.debug(f"Failed to fetch {symbol}: {e}")
                    
                except Exception as e:
                    logger.warning(f"Batch fetch failed: {e}")
                
                # Progress update
                if (i + batch_size) % 500 == 0:
                    logger.info(f"Progress: {min(i + batch_size, len(valid_symbols))}/{len(valid_symbols)} symbols")
        
        else:
            # Fallback to direct API
            for symbol in valid_symbols:
                try:
                    stock = self._fetch_single_stock(symbol)
                    if stock:
                        stocks.append(stock)
                    elif stock is False:  # Explicitly invalid
                        invalid_symbols[symbol] = {
                            'reason': 'API returned 404',
                            'error_code': '404'
                        }
                except Exception as e:
                    logger.debug(f"Failed to fetch {symbol}: {e}")
        
        # Add invalid symbols to blacklist
        if invalid_symbols:
            self.blacklist.add_symbols_batch(invalid_symbols)
            logger.info(f"Added {len(invalid_symbols)} invalid symbols to blacklist")
        
        logger.info(f"Successfully fetched {len(stocks)} stocks")
        return stocks
    
    def _fetch_single_stock(self, symbol: str) -> Optional[Union[StockData, bool]]:
        """
        Fetch single stock using direct API
        Returns: StockData if successful, None if error, False if invalid symbol
        """
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            
            # Apply rate limiting
            self.rate_limiter.wait_if_needed()
            
            # Enhanced timeout handling
            try:
                response = self.session.get(url, timeout=REQUEST_TIMEOUT)
                self._active_requests.add(response)
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout fetching {symbol}")
                self.blacklist.add_symbol(symbol, "Request timeout", "TIMEOUT")
                return None
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error for {symbol}")
                return None
            
            if response.status_code == 404:
                logger.debug(f"Symbol {symbol} not found (404)")
                self.blacklist.add_symbol(symbol, f"HTTP 404: Symbol not found", "404")
                return False  # Explicitly invalid
            
            if response.status_code == 429:
                logger.warning(f"Rate limited when fetching {symbol}")
                self.rate_limiter.register_429()
                return None
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for error in response
                if 'chart' in data and 'error' in data['chart']:
                    error_info = data['chart']['error']
                    if error_info.get('code') == 'Not Found':
                        logger.debug(f"Symbol {symbol} not found in response")
                        self.blacklist.add_symbol(
                            symbol, 
                            error_info.get('description', 'Not found'),
                            'NOT_FOUND'
                        )
                        return False
                
                chart = data.get('chart', {}).get('result', [{}])[0]
                
                if chart:
                    meta = chart.get('meta', {})
                    stock_data = StockData(
                        symbol=symbol,
                        name=symbol,
                        exchange=meta.get('exchangeName', 'UNKNOWN'),
                        current_price=meta.get('regularMarketPrice', 0),
                        previous_close=meta.get('previousClose', 0),
                        volume=meta.get('regularMarketVolume', 0),
                        data_source='yahoo_chart'
                    )
                    
                    # Register successful request
                    self.rate_limiter.register_success()
                    self.rate_limiter.reset_empty_responses()
                    
                    return stock_data
                else:
                    # Empty response
                    if self.rate_limiter.register_empty_response():
                        logger.info("Multiple empty responses detected, applying backoff")
                        time.sleep(1.5)
            
            return None
            
        except requests.exceptions.RequestException as e:
            error_str = str(e)
            if '404' in error_str or 'not found' in error_str.lower():
                logger.debug(f"Symbol {symbol} not found: {e}")
                self.blacklist.add_symbol(symbol, f"Exception: {error_str[:100]}", '404')
                return False
            
            logger.debug(f"Request failed for {symbol}: {e}")
            return None
            
        except Exception as e:
            logger.debug(f"Unexpected error fetching {symbol}: {e}")
            return None
    
    def _parse_yfinance_data(self, symbol: str, info: Dict) -> Optional[StockData]:
        """Parse yfinance info dict into StockData"""
        try:
            def safe_float(value, default=0.0):
                if value is None or value == 'N/A':
                    return default
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return default
            
            def safe_int(value, default=0):
                if value is None or value == 'N/A':
                    return default
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return default
            
            return StockData(
                symbol=symbol,
                name=info.get('longName', info.get('shortName', symbol)),
                exchange=info.get('exchange', 'UNKNOWN'),
                sector=info.get('sector', 'Unknown'),
                industry=info.get('industry', 'Unknown'),
                market_cap=safe_float(info.get('marketCap')),
                current_price=safe_float(info.get('currentPrice', info.get('regularMarketPrice'))),
                previous_close=safe_float(info.get('previousClose')),
                volume=safe_int(info.get('volume', info.get('regularMarketVolume'))),
                pe_ratio=safe_float(info.get('trailingPE'), None),
                dividend_yield=safe_float(info.get('dividendYield'), None),
                beta=safe_float(info.get('beta'), None),
                week_52_high=safe_float(info.get('fiftyTwoWeekHigh')),
                week_52_low=safe_float(info.get('fiftyTwoWeekLow')),
                shares_outstanding=safe_float(info.get('sharesOutstanding')),
                float_shares=safe_float(info.get('floatShares')),
                enterprise_value=safe_float(info.get('enterpriseValue')),
                peg_ratio=safe_float(info.get('pegRatio'), None),
                price_to_book=safe_float(info.get('priceToBook'), None),
                debt_to_equity=safe_float(info.get('debtToEquity'), None),
                return_on_equity=safe_float(info.get('returnOnEquity'), None),
                return_on_assets=safe_float(info.get('returnOnAssets'), None),
                profit_margin=safe_float(info.get('profitMargins'), None),
                revenue_growth=safe_float(info.get('revenueGrowth'), None),
                earnings_growth=safe_float(info.get('earningsGrowth'), None),
                data_source='yfinance'
            )
            
        except Exception as e:
            logger.warning(f"Error parsing yfinance data for {symbol}: {e}")
            return None
    
    def cleanup(self):
        """Cleanup session and resources"""
        try:
            self.session.close()
            with self._cache_lock:
                self._symbol_cache.clear()
        except:
            pass

# ===============================================================================
# PARALLEL STOCK FETCHER
# ===============================================================================

class ParallelStockFetcher:
    """Parallel stock data fetcher with automatic exchange detection"""
    
    def __init__(self):
        self.api = YahooFinanceAPI()
        self.max_workers = min(MAX_WORKERS, 4)
        self.executor = None
        self._shutdown = False
        
    def __enter__(self):
        """Context manager entry"""
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup"""
        self.cleanup()
    
    @performance_monitor("fetch_all_exchanges")
    def fetch_all_exchanges(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch stock data for all US exchanges"""
        if self._shutdown:
            logger.warning("Fetcher is shutting down, skipping fetch")
            return {}
            
        logger.info("Fetching data for all US exchanges")
        
        # Get all US symbols
        all_symbols = list(self.api.get_all_us_symbols())
        
        if not all_symbols:
            logger.error("No symbols found - symbol discovery failed")
            return {}
        
        logger.info(f"Processing {len(all_symbols)} total US symbols")
        
        # Fetch data for all symbols
        all_stocks = self.api.fetch_stock_data(all_symbols)
        
        # Group by exchange
        results = {
            'NYSE': [],
            'NASDAQ': [],
            'AMEX': [],
            'OTHER': []
        }
        
        for stock in all_stocks:
            exchange = self._classify_exchange(stock.exchange)
            results[exchange].append(stock.to_dict())
        
        # Remove empty categories
        results = {k: v for k, v in results.items() if v}
        
        # Log summary
        for exchange, stocks in results.items():
            logger.info(f"{exchange}: {len(stocks)} stocks")
        
        return results
    
    def _classify_exchange(self, exchange_name: str) -> str:
        """Classify exchange into major categories"""
        if not exchange_name:
            return "OTHER"
        
        exchange_upper = exchange_name.upper()
        
        if any(x in exchange_upper for x in ['NYSE', 'NYQ']):
            return 'NYSE'
        elif any(x in exchange_upper for x in ['NASDAQ', 'NMS', 'NGM']):
            return 'NASDAQ'
        elif any(x in exchange_upper for x in ['AMEX', 'ASE', 'ARCA']):
            return 'AMEX'
        else:
            return 'OTHER'
    
    def cleanup(self):
        """Cleanup resources properly"""
        self._shutdown = True
        if self.executor:
            try:
                self.executor.shutdown(wait=False)
            except Exception as e:
                logger.warning(f"Error during executor shutdown: {e}")
            finally:
                self.executor = None
        
        if self.api:
            self.api.cleanup()

# ===============================================================================
# PUBLIC API
# ===============================================================================

_fetcher_instance = None
_fetcher_lock = threading.Lock()

def get_fetcher() -> ParallelStockFetcher:
    """Get singleton fetcher instance with proper cleanup"""
    global _fetcher_instance
    if _fetcher_instance is None:
        with _fetcher_lock:
            if _fetcher_instance is None:
                _fetcher_instance = ParallelStockFetcher()
    return _fetcher_instance

@performance_monitor("fetch_stocks")
def fetch_stocks(exchange: str) -> List[Dict[str, Any]]:
    """Fetch stocks for a specific exchange"""
    with get_fetcher() as fetcher:
        # Get all stocks and filter by exchange
        all_stocks = fetcher.fetch_all_exchanges()
        
        if exchange in all_stocks:
            return all_stocks[exchange]
        else:
            logger.warning(f"No stocks found for exchange {exchange}")
            return []

# fetcher_fix.py - Standardized fetching functions
from fetcher import YahooFinanceAPI, fetch_stocks as original_fetch_stocks
from typing import List, Dict, Optional

def fetch_stocks_as_dicts(symbols: List[str]) -> List[Dict]:
    """
    Fetch stocks and ensure they're returned as a list of dictionaries
    
    Args:
        symbols: List of stock symbols
        
    Returns:
        List of dictionaries with stock data
    """
    # Ensure symbols is a list
    if isinstance(symbols, str):
        symbols = [symbols]
    
    # Fetch using original function
    results = original_fetch_stocks(symbols)
    
    # Convert to list of dicts
    output = []
    
    if isinstance(results, list):
        for item in results:
            if hasattr(item, 'to_dict'):
                output.append(item.to_dict())
            elif isinstance(item, dict):
                output.append(item)
    elif isinstance(results, dict):
        for symbol, data in results.items():
            if hasattr(data, 'to_dict'):
                record = data.to_dict()
            elif isinstance(data, dict):
                record = data
            else:
                continue
            # Ensure symbol is in the record
            if 'symbol' not in record:
                record['symbol'] = symbol
            output.append(record)
    
    return output

def fetch_single_stock(symbol: str) -> Optional[Dict]:
    """
    Fetch a single stock and return as dictionary
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Dictionary with stock data or None
    """
    results = fetch_stocks_as_dicts([symbol])
    return results[0] if results else None

def fetch_all_us_stocks() -> Dict[str, List[Dict[str, Any]]]:
    """Fetch all US stocks grouped by exchange"""
    with get_fetcher() as fetcher:
        return fetcher.fetch_all_exchanges()

def clear_cache():
    """Clear all cached data"""
    if _fetcher_instance:
        with _fetcher_instance.api._cache_lock:
            _fetcher_instance.api._symbol_cache.clear()
        
    # Also clear symbol cache file for fresh discovery
    cache_file = DATA_DIR / "us_symbols_cache.json"
    if cache_file.exists():
        cache_file.unlink()
        logger.info("Cleared symbol cache file")
    
    logger.debug("Cache cleared")

# Cleanup on module unload
def cleanup():
    """Cleanup resources on exit"""
    global _fetcher_instance
    
    if _fetcher_instance:
        _fetcher_instance.cleanup()
        _fetcher_instance = None
    
    # Save blacklist if dirty
    blacklist = get_blacklist()
    blacklist.save_if_dirty()
    
    clear_cache()

atexit.register(cleanup)