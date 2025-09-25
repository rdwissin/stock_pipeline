# config.py - Enhanced configuration with Yahoo Finance as primary provider

__version__ = "1.5.0"
__copyright__ = "Copyright 2025, Richard D. Wissinger"
__author__ = "Richard D. Wissinger"
__email__ = "rick.wissinger@gmail.com"
__license__ = "MIT"  # or your chosen license
__status__ = "Production"

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Primary Provider: Yahoo Finance (No API key required)
PRIMARY_PROVIDER = "yahoo"
YAHOO_BASE_URL = "https://query1.finance.yahoo.com/v1"
YAHOO_SCREENER_URL = "https://query2.finance.yahoo.com/v1/finance/screener"

# Target exchanges and their Yahoo Finance identifiers
EXCHANGES = {
    "NYSE": "us_market",
    "NASDAQ": "us_market", 
    "AMEX": "us_market"
}

# Stock symbols to fetch from each exchange (Yahoo provides all US stocks together)
EXCHANGE_SYMBOLS = {
    "NYSE": [],  # Will be populated dynamically
    "NASDAQ": [],
    "AMEX": []
}

# Optional Secondary Providers (require API keys)
SECONDARY_PROVIDERS = {
    "financial_modeling_prep": {
        "enabled": bool(os.getenv("FMP_API_KEY")),
        "api_key": os.getenv("FMP_API_KEY"),
        "base_url": "https://financialmodelingprep.com/api/v3",
        "rate_limit": 5,
        "tier": "free"  # free, basic, premium
    },
    "alpha_vantage": {
        "enabled": bool(os.getenv("AV_API_KEY")),
        "api_key": os.getenv("AV_API_KEY"),
        "base_url": "https://www.alphavantage.co/query",
        "rate_limit": 5,
        "tier": "free"
    },
    "iex_cloud": {
        "enabled": bool(os.getenv("IEX_API_KEY")),
        "api_key": os.getenv("IEX_API_KEY"),
        "base_url": "https://cloud.iexapis.com/v1",
        "rate_limit": 100,
        "tier": "free"
    },
    "polygon": {
        "enabled": bool(os.getenv("POLYGON_API_KEY")),
        "api_key": os.getenv("POLYGON_API_KEY"),
        "base_url": "https://api.polygon.io/v3",
        "rate_limit": 5,
        "tier": "free"
    }
}

# Parallel Processing Configuration
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))  # CPU cores to use
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "50"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))  # Symbols per batch
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # Database insert batch size

# Request Configuration
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))  # Base delay in seconds
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
CONNECTION_TIMEOUT = int(os.getenv("CONNECTION_TIMEOUT", "10"))

# Rate Limiting (per provider)
REQUESTS_PER_MINUTE = int(os.getenv("REQUESTS_PER_MINUTE", "60"))  # Yahoo is generous
YAHOO_RATE_LIMIT = 60 / REQUESTS_PER_MINUTE if REQUESTS_PER_MINUTE > 0 else 0

# Directory Configuration - macOS optimized
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"
BACKUP_DIR = BASE_DIR / "backups"

# Create directories if they don't exist
for directory in [DATA_DIR, LOGS_DIR, CACHE_DIR, BACKUP_DIR]:
    directory.mkdir(exist_ok=True)

# Database Configuration
DB_PATH = DATA_DIR / "stocks_enhanced.db"
BACKUP_DB_PATH = BACKUP_DIR / f"stocks_backup_{os.getenv('USER', 'user')}.db"

# Logging Configuration
LOG_LEVELS = {
    "TRACE": 5,     # Most detailed - API calls, timing
    "DEBUG": 10,    # Detailed - Function entry/exit
    "INFO": 20,     # General - Progress updates
    "WARNING": 30,  # Issues - Retries, data anomalies
    "ERROR": 40,    # Problems - Failed operations
    "CRITICAL": 50  # Fatal - Pipeline stoppers
}

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
ENABLE_PERFORMANCE_LOGGING = os.getenv("ENABLE_PERFORMANCE_LOGGING", "true").lower() == "true"
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "true").lower() == "true"
LOG_TO_CONSOLE = os.getenv("LOG_TO_CONSOLE", "true").lower() == "true"

# Memory tracking for logger.py
ENABLE_MEMORY_TRACKING = os.getenv("ENABLE_MEMORY_TRACKING", "true").lower() == "true"
LOG_ROTATION_COUNT = int(os.getenv("LOG_ROTATION_COUNT", "10"))
MAX_LOG_SIZE_MB = int(os.getenv("MAX_LOG_SIZE_MB", "100"))

# Email Alert Configuration (optional)
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
EMAIL_USER = os.getenv("EMAIL_USER", "")
EMAIL_PASS = os.getenv("EMAIL_PASS", "")
ALERT_RECIPIENTS = os.getenv("ALERT_RECIPIENTS", "").split(",") if os.getenv("ALERT_RECIPIENTS") else []
SEND_SUCCESS_ALERTS = os.getenv("SEND_SUCCESS_ALERTS", "true").lower() == "true"

# Data Collection Settings
COLLECT_HISTORICAL_PRICES = os.getenv("COLLECT_HISTORICAL_PRICES", "true").lower() == "true"
HISTORICAL_DAYS = int(os.getenv("HISTORICAL_DAYS", "30"))  # Days of price history
COLLECT_FINANCIAL_RATIOS = os.getenv("COLLECT_FINANCIAL_RATIOS", "true").lower() == "true"
COLLECT_ANALYST_DATA = os.getenv("COLLECT_ANALYST_DATA", "false").lower() == "true"

# Performance and Memory Settings
MAX_MEMORY_MB = int(os.getenv("MAX_MEMORY_MB", "2048"))  # Max memory usage
ENABLE_COMPRESSION = os.getenv("ENABLE_COMPRESSION", "true").lower() == "true"
AUTO_VACUUM_DB = os.getenv("AUTO_VACUUM_DB", "true").lower() == "true"

# Data Quality Settings
MIN_MARKET_CAP = float(os.getenv("MIN_MARKET_CAP", "0"))  # Filter by market cap
EXCLUDE_OTCBB = os.getenv("EXCLUDE_OTCBB", "true").lower() == "true"
EXCLUDE_PINK_SHEETS = os.getenv("EXCLUDE_PINK_SHEETS", "true").lower() == "true"

# Cache Settings
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
CACHE_DURATION_HOURS = int(os.getenv("CACHE_DURATION_HOURS", "6"))
MAX_CACHE_SIZE_MB = int(os.getenv("MAX_CACHE_SIZE_MB", "100"))

# Configuration for historical data limits
MAX_YEARS_HISTORY = int(os.getenv("MAX_YEARS_HISTORY", "10"))  # Extended to 10 years for comprehensive analysis
MAX_RETRIES_ON_LOCK = int(os.getenv("MAX_RETRIES_ON_LOCK", "10"))  # Increased retries with exponential backoff
BASE_RETRY_DELAY = float(os.getenv("MIN_MARKET_CAP", "0.1"))   # Base delay for exponential backoff
MAX_RETRY_DELAY = float(os.getenv("MIN_MARKET_CAP", "30.0"))   # Maximum delay between retries
