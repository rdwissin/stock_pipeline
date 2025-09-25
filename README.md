# 📊 Enhanced Stock Data Pipeline v1.5.1

A robust, production-grade stock data collection and analysis system for the complete US stock market. Features comprehensive data fetching from Yahoo Finance, advanced technical analysis, automated scheduling, and intelligent blacklist management.

**Version 1.5.1** - Fixed scheduler bug for better reliability  
**Author:** Richard D. Wissinger  
**License:** MIT

---

## 🌟 Key Features

### Core Capabilities
- **Complete US Market Coverage**: Automatically fetches ALL stocks from NYSE, NASDAQ, and AMEX
- **No API Keys Required**: Uses Yahoo Finance public endpoints
- **Maximum Historical Data**: Fetches up to 10 years of price history
- **Clean Interrupt Handling**: Ctrl-C saves progress and exits gracefully
- **Automated Scheduling**: Built-in cron/launchd scheduler installer
- **Smart Blacklist**: Auto-blacklists invalid symbols to prevent repeated failures
- **Email Alerts**: Get notified of pipeline failures or important events
- **Comprehensive Logging**: Detailed logs with rotation and performance metrics
- **Database Management**: SQLite with automatic backups and optimization
- **Resource Monitoring**: Tracks memory and CPU usage in real-time

### Data Collection
- Real-time stock prices and market data
- Company fundamentals (P/E, Market Cap, Dividends, etc.)
- 10-year historical price data (OHLCV)
- Financial ratios and metrics
- Sector and industry classification
- Technical indicators (RSA, MACD, Bollinger Bands)

### Analysis Tools
- Technical analysis with 20+ indicators
- Strategy backtesting framework
- Watchlist management system
- Market sector analysis
- Performance analytics
- Data quality monitoring

---

## 🚀 Quick Start

### 1. Prerequisites
```bash
# macOS (via Homebrew)
brew install python@3.13

# Linux
sudo apt update && sudo apt install python3.13 python3.13-venv python3-pip

# Verify installation
python3 --version  # Should show 3.13+
```

### 2. Installation
```bash
# Clone repository
git clone https://github.com/yourusername/stock_pipeline.git
cd stock_pipeline

# Run setup script (creates venv, installs dependencies, configures environment)
./setup.sh

# Or manual setup
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### 3. Configuration
Edit `.env` file with your settings:
```bash
# Core Settings
MAX_WORKERS=8                    # Parallel processing threads
BATCH_SIZE=100                   # Stocks per batch
MAX_CONCURRENT_REQUESTS=50       # Yahoo Finance rate limiting

# Email Alerts (optional)
EMAIL_USER=your.email@gmail.com
EMAIL_PASS=your_app_password
ALERT_RECIPIENTS=recipient@example.com

# Historical Data
COLLECT_HISTORICAL_PRICES=true
HISTORICAL_DAYS=30              # For daily updates
MAX_YEARS_HISTORY=10            # Maximum historical data
```

### 4. Run Pipeline
```bash
# Normal run (fetches today's data + recent history)
./run_pipeline.sh

# Force run with complete historical data
./run_pipeline.sh --force --full-history

# Debug mode with sequential processing
./run_pipeline.sh --debug --sequential
```

---

## 🗓️ Automated Scheduling (v1.5.1)

### macOS Setup

The installer now correctly handles schedule selection:

```bash
# Run the installer
./install_cron.sh

# Select scheduler:
# 1) launchd (Recommended for macOS)
# 2) crontab (Traditional Unix)

# Choose schedule:
# 1) Default (6 PM weekdays)
# 2) Market close (4 PM weekdays)
# 3) After hours (8 PM weekdays)
# 4) Early morning (6 AM weekdays)
# 5) Lunch time (12 PM weekdays)
# 6) Daily (6 PM every day)
# 7) Twice daily (9 AM, 6 PM)
# 8) Custom schedule

# The installer will configure everything automatically
```

### Manual Crontab Setup
```bash
# Edit crontab
crontab -e

# Add daily update at 6 PM weekdays
0 18 * * 1-5 cd /path/to/stock_pipeline && ./run_pipeline.sh

# Add weekly full history update (Sunday 8 PM)
0 20 * * 0 cd /path/to/stock_pipeline && ./run_pipeline.sh --full-history
```

### Scheduler Management
```bash
# Check status
./install_cron.sh --status

# Uninstall
./install_cron.sh --uninstall

# View logs
tail -f logs/launchd.log  # macOS launchd
tail -f logs/cron.log     # crontab
```

---

## 📁 Project Structure

```
stock_pipeline/
├── main.py                 # Main pipeline orchestrator (v1.4.0)
├── fetcher.py             # Yahoo Finance data fetcher
├── storage.py             # Database operations (Python 3.13 compatible)
├── historical_fetcher.py  # Historical data collector
├── technical_analysis.py  # Technical indicators
├── strategies.py          # Trading strategy definitions
├── watchlist_manager.py   # Watchlist management
├── blacklist.py          # Invalid symbol management
├── alerts.py             # Email notification system
├── config.py             # Configuration management
├── logger.py             # Enhanced logging system
│
├── run_pipeline.sh       # Main runner script (v1.4.0)
├── install_cron.sh       # Scheduler installer (v1.5.1 - Fixed)
├── monitor_pipeline.sh   # Status monitoring
├── maintain_pipeline.sh  # Maintenance utilities
├── setup.sh             # Initial setup script
│
├── data/                # Data directory
│   ├── stocks_enhanced.db    # Main database
│   ├── blacklist.json       # Blacklisted symbols
│   └── watchlists/          # Watchlist files
│
├── logs/                # Log directory
│   ├── pipeline_*.log       # Pipeline execution logs
│   ├── errors_*.log         # Error logs
│   └── performance_*.log    # Performance metrics
│
├── reports/            # Analysis reports
└── backups/           # Database backups
```

---

## 💾 Database Schema

### Main Tables

#### stocks
```sql
CREATE TABLE stocks (
    id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    snapshot_date TEXT NOT NULL,
    exchange TEXT,
    name TEXT,
    sector TEXT,
    industry TEXT,
    market_cap REAL,
    current_price REAL,
    previous_close REAL,
    volume INTEGER,
    avg_volume INTEGER,
    pe_ratio REAL,
    dividend_yield REAL,
    beta REAL,
    high_52week REAL,
    low_52week REAL,
    -- Additional fields...
    UNIQUE(symbol, snapshot_date)
);
```

#### historical_prices
```sql
CREATE TABLE historical_prices (
    symbol TEXT,
    date TEXT,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    PRIMARY KEY (symbol, date)
);
```

---

## 📊 Usage Examples

### Basic Pipeline Operations
```bash
# Run complete pipeline
python main.py

# Force refresh all data
python main.py --force

# Fetch maximum historical data
python main.py --full-history

# Skip historical data
python main.py --no-history

# Debug mode
python main.py --debug
```

### Historical Data Management
```bash
# Fetch 10 years for all stocks
python historical_fetcher.py --all

# Update recent 5 days only
python historical_fetcher.py --recent 5

# Specific symbols
python historical_fetcher.py --symbols AAPL MSFT GOOGL
```

### Analysis and Reports
```bash
# Analyze watchlist
python analyze_data.py watchlist "Tech Stocks"

# Generate market report
python analyze_data.py report --format excel

# Run strategy analysis
python analyze_data.py strategy "Value Investing"
```

### Monitoring
```bash
# Check pipeline status
./monitor_pipeline.sh

# View real-time logs
tail -f logs/pipeline_*.log

# Check database stats
sqlite3 data/stocks_enhanced.db "SELECT COUNT(*) FROM stocks;"
```

---

## 🐍 Python 3.13 Support

Full compatibility with Python 3.13.7+:

```bash
# Automatic migration
python migrate_to_python313.py

# Test compatibility
python test_python313_compatibility.py
```

**Fixed Issues**:
- ✅ sqlite3 weak reference errors resolved
- ✅ Enhanced connection pooling
- ✅ Improved thread safety
- ✅ Better memory management

---

## 🛠️ Troubleshooting

### Common Issues

#### Scheduler Not Running (Fixed in v1.5.1)
```bash
# The schedule selection bug has been fixed
# If you still have issues:
./install_cron.sh --uninstall
./install_cron.sh  # Reinstall with fixed version
```

#### Rate Limiting (429 Errors)
```bash
# Reduce concurrent requests
echo "MAX_CONCURRENT_REQUESTS=25" >> .env
echo "RETRY_DELAY=2.0" >> .env

# Use sequential mode
python main.py --sequential
```

#### Memory Issues
```bash
# Reduce batch sizes
echo "BATCH_SIZE=50" >> .env
echo "MAX_WORKERS=4" >> .env

# Monitor memory
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
```

#### Database Locked
```bash
# Enable WAL mode
sqlite3 data/stocks_enhanced.db "PRAGMA journal_mode=WAL;"

# Increase timeout
sqlite3 data/stocks_enhanced.db "PRAGMA busy_timeout=30000;"
```

---

## 📚 API Reference

### Main Pipeline
```python
from main import EnhancedPipeline

pipeline = EnhancedPipeline()
success = pipeline.run_pipeline(
    force=False,           # Force refresh
    parallel=True,         # Use parallel processing
    fetch_historical=True, # Fetch historical data
    historical_mode="full" # "full" or "recent"
)
```

### Data Fetching
```python
from fetcher import fetch_all_us_stocks

# Get all US stocks
all_stocks = fetch_all_us_stocks()
# Returns: {'NYSE': [...], 'NASDAQ': [...], 'AMEX': [...]}
```

### Database Operations
```python
from storage import get_database

db = get_database()
overview = db.get_market_overview()
snapshot = db.get_latest_snapshot()
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Yahoo Finance for providing free stock market data
- The Python community for excellent libraries
- Contributors and users of this project

---

## 📞 Contact

**Author:** Richard D. Wissinger  
**Email:** rick.wissinger@gmail.com  
**Project Link:** [https://github.com/yourusername/stock_pipeline](https://github.com/yourusername/stock_pipeline)

---

## 📈 Version History

- **v1.5.1** (2025-09-21): Fixed scheduler installation bug, improved schedule selection
- **v1.5.0** (2025-09-20): Added automated scheduling with launchd/cron support
- **v1.4.0** (2025-09-18): Enhanced signal handling, Python 3.13 support
- **v1.3.0** (2025-09-15): Added complete US market fetching
- **v1.2.0** (2025-09-10): Improved blacklist management
- **v1.1.0** (2025-09-05): Added email alerts
- **v1.0.0** (2025-09-01): Initial release