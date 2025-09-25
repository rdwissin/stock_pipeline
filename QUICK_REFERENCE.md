# 📋 QUICK REFERENCE - Stock Pipeline v1.5.1

Quick commands and troubleshooting guide for the Enhanced Stock Data Pipeline.

---

## 🚀 Essential Commands

### Daily Operations
```bash
# Run pipeline (normal)
./run_pipeline.sh

# Force run (ignore schedule checks)
./run_pipeline.sh --force

# Run with full historical data
./run_pipeline.sh --full-history

# Debug mode
./run_pipeline.sh --debug

# Sequential mode (no parallelism)
./run_pipeline.sh --sequential
```

### Pipeline Management
```bash
# Check status
./monitor_pipeline.sh

# Install scheduler (FIXED in v1.5.1)
./install_cron.sh

# Check scheduler status
./install_cron.sh --status

# Uninstall scheduler
./install_cron.sh --uninstall

# Database maintenance
./maintain_pipeline.sh optimize-db
./maintain_pipeline.sh backup-db
./maintain_pipeline.sh clean-logs 30
```

---

## ⚙️ Configuration Quick Reference

### Environment Variables (.env)
```bash
# Performance Tuning
MAX_WORKERS=8                    # CPU cores to use
BATCH_SIZE=100                   # Stocks per batch
MAX_CONCURRENT_REQUESTS=50       # Yahoo API limit

# Email Alerts
EMAIL_USER=your.email@gmail.com
EMAIL_PASS=app_specific_password
ALERT_RECIPIENTS=user1@example.com,user2@example.com

# Data Collection
COLLECT_HISTORICAL_PRICES=true
HISTORICAL_DAYS=30               # For daily updates
MAX_YEARS_HISTORY=10             # Maximum history

# Logging
LOG_LEVEL=INFO                   # DEBUG, INFO, WARNING, ERROR
ENABLE_PERFORMANCE_LOGGING=true
```

---

## 📊 Database Queries

### Quick Stats
```bash
# Total stocks
sqlite3 data/stocks_enhanced.db "SELECT COUNT(DISTINCT symbol) FROM stocks;"

# Latest snapshot date
sqlite3 data/stocks_enhanced.db "SELECT MAX(snapshot_date) FROM stocks;"

# Historical records count
sqlite3 data/stocks_enhanced.db "SELECT COUNT(*) FROM historical_prices;"

# Top gainers today
sqlite3 data/stocks_enhanced.db "
SELECT symbol, name, 
       ((current_price - previous_close) / previous_close * 100) as pct_change
FROM stocks 
WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM stocks)
ORDER BY pct_change DESC 
LIMIT 10;"
```

### Market Overview
```sql
-- Sector breakdown
SELECT sector, COUNT(*) as count, 
       ROUND(AVG(market_cap/1e9), 2) as avg_market_cap_b
FROM stocks 
WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM stocks)
GROUP BY sector 
ORDER BY count DESC;

-- Exchange summary
SELECT exchange, COUNT(*) as total_stocks,
       ROUND(SUM(market_cap)/1e12, 2) as total_market_cap_t
FROM stocks
WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM stocks)
GROUP BY exchange;
```

---

## 🐍 Python One-Liners

### Quick Analysis
```python
# Market overview
python -c "from storage import get_database; db = get_database(); print(db.get_market_overview())"

# List watchlists
python -c "from watchlist_manager import WatchlistManager; m = WatchlistManager(); print([w.name for w in m.get_all_watchlists()])"

# Blacklist size
python -c "from blacklist import get_blacklist; print(f'Blacklisted: {len(get_blacklist()._blacklist_set)}')"

# Quick stock analysis
python -c "from technical_analysis import analyze_single_symbol; analyze_single_symbol('AAPL')"

# Strategy matches
python -c "from strategies import find_strategy_matches; print(find_strategy_matches('Quality Value')[:5])"

# Database stats
python -c "import sqlite3; conn = sqlite3.connect('data/stocks_enhanced.db'); c = conn.cursor(); c.execute('SELECT COUNT(*) FROM stocks'); print(f'Total records: {c.fetchone()[0]}')"
```

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### Scheduler Installation Fixed (v1.5.1)
```bash
# The bug where schedule selection failed has been fixed
# To reinstall with the fixed version:
./install_cron.sh --uninstall
./install_cron.sh

# Verify it's working:
./install_cron.sh --status
```

#### Scheduler Not Running (macOS)
```bash
# For launchd:
launchctl list | grep stockpipeline     # Check if loaded

# If not loaded:
launchctl load ~/Library/LaunchAgents/com.stockpipeline.updater.plist

# For crontab:
crontab -l                               # Check if scheduled

# Check scheduler logs:
tail -f logs/launchd.log                # launchd logs
tail -f logs/cron.log                   # cron logs
```

#### 429 Rate Limiting
```bash
# Reduce request rate
echo "MAX_CONCURRENT_REQUESTS=25" >> .env
echo "RETRY_DELAY=2.0" >> .env

# Use sequential mode
python main.py --sequential
```

#### Memory Issues
```bash
# Check memory usage
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"

# Reduce memory usage
echo "MAX_WORKERS=4" >> .env
echo "BATCH_SIZE=50" >> .env

# Set memory limit
python main.py --memory-limit 2048
```

#### Database Locked
```bash
# Enable WAL mode (recommended)
sqlite3 data/stocks_enhanced.db "PRAGMA journal_mode=WAL;"

# Increase timeout
sqlite3 data/stocks_enhanced.db "PRAGMA busy_timeout=30000;"

# Kill stuck processes
pkill -f "python.*main.py"
```

#### Email Not Working
```python
# Test email configuration
from alerts import send_alert
send_alert("Test", "Test message from stock pipeline")

# Check configuration
from config import EMAIL_USER, EMAIL_PASS, ALERT_RECIPIENTS
print(f"Email: {EMAIL_USER}")
print(f"Recipients: {ALERT_RECIPIENTS}")
```

#### Missing Data
```bash
# Re-collect all data
python main.py --force

# Update historical
python historical_fetcher.py --all --years 10

# Check blacklist
python blacklist.py show

# Clear blacklist
python blacklist.py cleanup --days 30
```

---

## ⏰ Automation Setup (v1.5.1 - FIXED)

### 🍎 macOS - Two Options

#### Option 1: launchd (Recommended)
```bash
# Run installer and select launchd
./install_cron.sh
# Select: 1 (launchd)
# Choose schedule: 1-8
# Confirm installation
# Done! Schedule will work correctly now.

# Verify it's working
launchctl list | grep stockpipeline
```

#### Option 2: crontab
```bash
# Run installer and select crontab
./install_cron.sh
# Select: 2 (crontab)
# Choose schedule: 1-8
# Done!

# Verify
crontab -l
```

### 🐧 Linux - crontab
```bash
# Run installer (auto-selects crontab on Linux)
./install_cron.sh
# Choose schedule: 1-8
# Done!
```

### Schedule Options (All Fixed)
1. **Default (6 PM weekdays)** - Best for after-market analysis
2. **Market close (4 PM weekdays)** - Immediate post-market data
3. **After hours (8 PM weekdays)** - Complete after-hours data
4. **Early morning (6 AM weekdays)** - Pre-market preparation
5. **Lunch time (12 PM weekdays)** - Mid-day update
6. **Daily (6 PM every day)** - Include weekends
7. **Twice daily (9 AM, 6 PM)** - Morning and evening updates
8. **Custom schedule** - Define your own

---

## 📈 Performance Optimization

### System Resources
```bash
# High Performance (16+ cores, 32GB RAM)
cat >> .env << EOF
MAX_WORKERS=16
MAX_CONCURRENT_REQUESTS=100
BATCH_SIZE=200
MAX_MEMORY_MB=8192
EOF

# Medium (8 cores, 16GB RAM)
cat >> .env << EOF
MAX_WORKERS=8
MAX_CONCURRENT_REQUESTS=50
BATCH_SIZE=100
MAX_MEMORY_MB=4096
EOF

# Low Resource (4 cores, 8GB RAM)
cat >> .env << EOF
MAX_WORKERS=4
MAX_CONCURRENT_REQUESTS=25
BATCH_SIZE=50
MAX_MEMORY_MB=2048
EOF
```

---

## 🛡️ Maintenance Commands

### Daily Maintenance
```bash
# Check pipeline health
./monitor_pipeline.sh

# View today's logs
tail -f logs/pipeline_$(date +%Y%m%d)*.log

# Check for errors
grep ERROR logs/pipeline_$(date +%Y%m%d)*.log
```

### Weekly Maintenance
```bash
# Optimize database
sqlite3 data/stocks_enhanced.db "VACUUM; ANALYZE;"

# Backup database
./maintain_pipeline.sh backup-db

# Clean old logs (>30 days)
./maintain_pipeline.sh clean-logs 30
```

### Monthly Maintenance
```bash
# Full system check
./maintain_pipeline.sh full-check

# Clean old snapshots (keep 90 days)
python -c "from storage import cleanup_old_snapshots; cleanup_old_snapshots(90)"

# Review blacklist
python blacklist.py stats
python blacklist.py cleanup --days 180
```

---

## 🚨 Emergency Procedures

### Pipeline Stuck
```bash
# Kill all Python processes
pkill -f python

# Clear locks
rm -f data/*.lock

# Reset and restart
./run_pipeline.sh --force
```

### Database Corruption
```bash
# Check integrity
sqlite3 data/stocks_enhanced.db "PRAGMA integrity_check;"

# Restore from backup
cp backups/latest/stocks_enhanced.db data/

# Rebuild if needed
mv data/stocks_enhanced.db data/stocks_enhanced.corrupt
python main.py --force --full-history
```

### Complete Reset
```bash
# Backup current data
mkdir -p backups/manual_$(date +%Y%m%d)
cp -r data/* backups/manual_$(date +%Y%m%d)/

# Clean everything
rm -rf data/*.db logs/*.log

# Reinitialize
python main.py --force --full-history
```

---

## 📞 Support

For issues or questions:
- Check logs: `logs/errors_*.log`
- Review documentation: `README.md`
- Author: Richard D. Wissinger (rick.wissinger@gmail.com)

---

**Version:** 1.5.1 | **Updated:** 2025-09-21 | **Status:** Production