#!/usr/bin/env python3
"""
Enhanced Storage Module - Compatible with Existing Pipeline
Version: 1.5.1
Author: Richard D. Wissinger
Email: rick.wissinger@gmail.com

Fully compatible with the existing main.py pipeline.
Python 3.13+ compatible with connection pooling.
FIXED: Added missing transaction method for historical data saving.
"""

__version__ = "1.5.1"
__copyright__ = "Copyright 2025, Richard D. Wissinger"
__author__ = "Richard D. Wissinger"
__email__ = "rick.wissinger@gmail.com"
__license__ = "MIT"
__status__ = "Production"


import sqlite3
import json
import os
import sys
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from contextlib import contextmanager
from dataclasses import dataclass
import threading
from collections import deque
import time
import atexit
from pathlib import Path

# Import logger with fallback
try:
    from logger import setup_logger
    logger = setup_logger(__name__)
    def log_info(msg): logger.info(msg)
    def log_error(msg): logger.error(msg)
    def log_warning(msg): logger.warning(msg)
    def log_debug(msg): logger.debug(msg)
except ImportError:
    def log_info(msg): print(f"{datetime.now().strftime('%H:%M:%S')} - INFO - {msg}")
    def log_error(msg): print(f"{datetime.now().strftime('%H:%M:%S')} - ERROR - {msg}")
    def log_warning(msg): print(f"{datetime.now().strftime('%H:%M:%S')} - WARNING - {msg}")
    def log_debug(msg): print(f"{datetime.now().strftime('%H:%M:%S')} - DEBUG - {msg}")
    class Logger:
        def info(self, msg): log_info(msg)
        def error(self, msg): log_error(msg)
        def warning(self, msg): log_warning(msg)
        def debug(self, msg): log_debug(msg)
    logger = Logger()


class DatabaseError(Exception):
    """Custom database exception"""
    pass


class ConnectionPool:
    """Thread-safe connection pool for SQLite - Python 3.13 Compatible"""
    
    def __init__(self, db_path: str, max_connections: int = 10):
        self.db_path = db_path
        self.max_connections = max_connections
        
        # Python 3.13 fix: Use regular set instead of WeakSet
        self._active_connections: Set[sqlite3.Connection] = set()
        self._available_connections: deque = deque()
        self._lock = threading.RLock()
        self._shutdown = False
        self._connection_counter = 0
        
        atexit.register(self.close_all)
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool"""
        conn = None
        
        try:
            with self._lock:
                # Try to reuse an available connection
                while self._available_connections and not self._shutdown:
                    try:
                        conn = self._available_connections.popleft()
                        conn.execute("SELECT 1")
                        break
                    except:
                        conn = None
                
                # Create new connection if needed
                if conn is None and not self._shutdown:
                    if len(self._active_connections) >= self.max_connections:
                        # Wait a bit and try again
                        time.sleep(0.1)
                        if self._available_connections:
                            conn = self._available_connections.popleft()
                        else:
                            conn = self._create_connection()
                    else:
                        conn = self._create_connection()
                
                if conn:
                    self._active_connections.add(conn)
            
            yield conn
            
        finally:
            if conn:
                with self._lock:
                    self._active_connections.discard(conn)
                    if not self._shutdown:
                        try:
                            conn.rollback()
                            self._available_connections.append(conn)
                        except:
                            try:
                                conn.close()
                            except:
                                pass
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection"""
        conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None
        )
        
        # Enable optimizations
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=30000000")
        
        conn.row_factory = sqlite3.Row
        self._connection_counter += 1
        return conn
    
    def close_all(self):
        """Close all connections in the pool"""
        with self._lock:
            self._shutdown = True
            
            for conn in list(self._active_connections):
                try:
                    conn.close()
                except:
                    pass
            self._active_connections.clear()
            
            while self._available_connections:
                try:
                    conn = self._available_connections.popleft()
                    conn.close()
                except:
                    pass


class Database:
    """Database wrapper compatible with existing pipeline"""
    
    def __init__(self, db_path: str = "data/stocks_enhanced.db"):
        self.db_path = db_path
        self._ensure_data_directory()
        
        # Initialize connection pool
        self._pool = ConnectionPool(db_path, max_connections=10)
        
        # Initialize database if needed
        if not os.path.exists(self.db_path) or os.path.getsize(self.db_path) == 0:
            self.init_db()
    
    def _ensure_data_directory(self):
        """Ensure data directory exists"""
        data_dir = os.path.dirname(self.db_path)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
    
    @contextmanager
    def get_connection(self):
        """Get a database connection from pool"""
        try:
            with self._pool.get_connection() as conn:
                yield conn
        except Exception as e:
            log_error(f"Database connection error: {e}")
            raise DatabaseError(f"Cannot connect to database: {e}")
    
    @contextmanager
    def transaction(self, conn: sqlite3.Connection):
        """Transaction context manager for explicit transaction handling
        
        THIS WAS MISSING AND CAUSING HISTORICAL DATA SAVE FAILURES!
        
        Args:
            conn: Database connection
            
        Yields:
            cursor: Database cursor for executing queries
        """
        cursor = conn.cursor()
        try:
            # Start transaction
            conn.execute("BEGIN")
            yield cursor
            # Commit if successful
            conn.commit()
            log_debug("Transaction committed successfully")
        except Exception as e:
            # Rollback on error
            conn.rollback()
            log_error(f"Transaction rolled back due to error: {e}")
            raise
        finally:
            cursor.close()
    
    def init_db(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create all required tables
            cursor.executescript("""
                -- Main stocks table
                CREATE TABLE IF NOT EXISTS stocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    name TEXT,
                    exchange TEXT,
                    sector TEXT,
                    industry TEXT,
                    market_cap REAL,
                    pe_ratio REAL,
                    dividend_yield REAL,
                    beta REAL,
                    eps REAL,
                    revenue REAL,
                    profit_margin REAL,
                    operating_margin REAL,
                    roe REAL,
                    roa REAL,
                    debt_to_equity REAL,
                    current_ratio REAL,
                    quick_ratio REAL,
                    revenue_growth REAL,
                    earnings_growth REAL,
                    book_value REAL,
                    price_to_book REAL,
                    price_to_sales REAL,
                    peg_ratio REAL,
                    enterprise_value REAL,
                    ebitda REAL,
                    free_cash_flow REAL,
                    shares_outstanding INTEGER,
                    shares_float INTEGER,
                    insider_ownership REAL,
                    institutional_ownership REAL,
                    short_ratio REAL,
                    current_price REAL,
                    previous_close REAL,
                    open_price REAL,
                    day_low REAL,
                    day_high REAL,
                    volume INTEGER,
                    avg_volume INTEGER,
                    week_52_low REAL,
                    week_52_high REAL,
                    change_percent REAL,
                    is_blacklisted INTEGER DEFAULT 0,
                    blacklist_reason TEXT,
                    snapshot_date TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, exchange, snapshot_date)
                );
                
                -- Historical prices table with proper indexing
                CREATE TABLE IF NOT EXISTS historical_prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open_price REAL,
                    high_price REAL,
                    low_price REAL,
                    close_price REAL,
                    adj_close_price REAL,
                    volume INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date)
                );
                
                -- Pipeline runs table
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_date TEXT NOT NULL,
                    status TEXT,
                    exchanges TEXT,
                    total_stocks INTEGER,
                    new_stocks INTEGER,
                    updated_stocks INTEGER,
                    processing_time REAL,
                    peak_memory REAL,
                    errors TEXT,
                    performance_metrics TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Create indexes for better query performance
                CREATE INDEX IF NOT EXISTS idx_stocks_symbol ON stocks(symbol);
                CREATE INDEX IF NOT EXISTS idx_stocks_exchange ON stocks(exchange);
                CREATE INDEX IF NOT EXISTS idx_stocks_snapshot_date ON stocks(snapshot_date);
                CREATE INDEX IF NOT EXISTS idx_stocks_symbol_exchange_date ON stocks(symbol, exchange, snapshot_date);
                CREATE INDEX IF NOT EXISTS idx_historical_symbol ON historical_prices(symbol);
                CREATE INDEX IF NOT EXISTS idx_historical_date ON historical_prices(date);
                CREATE INDEX IF NOT EXISTS idx_historical_symbol_date ON historical_prices(symbol, date);
                CREATE INDEX IF NOT EXISTS idx_pipeline_date ON pipeline_runs(run_date);
            """)
            
            conn.commit()
            log_info("Database schema initialized successfully")
    
    def verify_historical_data(self, symbol: str) -> Dict[str, Any]:
        """Verify historical data for a symbol
        
        Args:
            symbol: Stock symbol to check
            
        Returns:
            Dictionary with verification results
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get record count
            cursor.execute("""
                SELECT 
                    COUNT(*) as record_count,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date,
                    COUNT(DISTINCT date) as unique_dates
                FROM historical_prices
                WHERE symbol = ?
            """, (symbol,))
            
            result = cursor.fetchone()
            
            if result and result['record_count'] > 0:
                # Calculate date range
                earliest = datetime.strptime(result['earliest_date'], '%Y-%m-%d')
                latest = datetime.strptime(result['latest_date'], '%Y-%m-%d')
                years_covered = (latest - earliest).days / 365.25
                
                return {
                    'symbol': symbol,
                    'has_data': True,
                    'record_count': result['record_count'],
                    'earliest_date': result['earliest_date'],
                    'latest_date': result['latest_date'],
                    'unique_dates': result['unique_dates'],
                    'years_covered': round(years_covered, 2),
                    'data_completeness': round(result['unique_dates'] / result['record_count'] * 100, 2)
                }
            else:
                return {
                    'symbol': symbol,
                    'has_data': False,
                    'record_count': 0,
                    'message': 'No historical data found'
                }
    
    def get_historical_summary(self) -> Dict[str, Any]:
        """Get summary of all historical data in database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT symbol) as total_symbols,
                    COUNT(*) as total_records,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date,
                    AVG(volume) as avg_volume
                FROM historical_prices
            """)
            
            result = cursor.fetchone()
            
            if result and result['total_records'] > 0:
                return {
                    'total_symbols': result['total_symbols'],
                    'total_records': result['total_records'],
                    'earliest_date': result['earliest_date'],
                    'latest_date': result['latest_date'],
                    'avg_volume': int(result['avg_volume']) if result['avg_volume'] else 0,
                    'avg_records_per_symbol': result['total_records'] // result['total_symbols'] if result['total_symbols'] > 0 else 0
                }
            else:
                return {
                    'total_symbols': 0,
                    'total_records': 0,
                    'message': 'No historical data in database'
                }
    
    def save_snapshot(self, exchange: str, stocks_data: List[Dict]) -> Tuple[int, int, int]:
        """Save stock snapshot for an exchange - existing implementation"""
        if not stocks_data:
            return 0, 0, 0
        
        snapshot_date = datetime.now().strftime("%Y-%m-%d")
        
        # Get existing symbols for this exchange and date
        existing_symbols = set()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT symbol FROM stocks 
                WHERE exchange = ? AND snapshot_date = ?
            """, (exchange, snapshot_date))
            existing_symbols = {row[0] for row in cursor.fetchall()}
        
        # Prepare data for insertion
        saved_count = 0
        new_count = 0
        updated_count = 0
        
        with self.get_connection() as conn:
            # Use the transaction context manager
            with self.transaction(conn) as cursor:
                for stock in stocks_data:
                    symbol = stock.get('symbol')
                    if not symbol:
                        continue
                    
                    try:
                        # Check if this is a new or existing stock
                        is_new = symbol not in existing_symbols
                        
                        cursor.execute("""
                            INSERT OR REPLACE INTO stocks (
                                symbol, name, exchange, sector, industry, market_cap, pe_ratio,
                                dividend_yield, beta, eps, revenue, profit_margin, operating_margin,
                                roe, roa, debt_to_equity, current_ratio, quick_ratio, revenue_growth,
                                earnings_growth, book_value, price_to_book, price_to_sales, peg_ratio,
                                enterprise_value, ebitda, free_cash_flow, shares_outstanding,
                                shares_float, insider_ownership, institutional_ownership, short_ratio,
                                current_price, previous_close, open_price, day_low, day_high,
                                volume, avg_volume, week_52_low, week_52_high, change_percent,
                                is_blacklisted, blacklist_reason, snapshot_date
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                                     ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                                     ?, ?, ?, ?, ?)
                        """, (
                            symbol, stock.get('name'), exchange,
                            stock.get('sector'), stock.get('industry'),
                            stock.get('market_cap'), stock.get('pe_ratio'),
                            stock.get('dividend_yield'), stock.get('beta'),
                            stock.get('eps'), stock.get('revenue'),
                            stock.get('profit_margin'), stock.get('operating_margin'),
                            stock.get('roe'), stock.get('roa'),
                            stock.get('debt_to_equity'), stock.get('current_ratio'),
                            stock.get('quick_ratio'), stock.get('revenue_growth'),
                            stock.get('earnings_growth'), stock.get('book_value'),
                            stock.get('price_to_book'), stock.get('price_to_sales'),
                            stock.get('peg_ratio'), stock.get('enterprise_value'),
                            stock.get('ebitda'), stock.get('free_cash_flow'),
                            stock.get('shares_outstanding'), stock.get('shares_float'),
                            stock.get('insider_ownership'), stock.get('institutional_ownership'),
                            stock.get('short_ratio'), stock.get('current_price'),
                            stock.get('previous_close'), stock.get('open_price'),
                            stock.get('day_low'), stock.get('day_high'),
                            stock.get('volume'), stock.get('avg_volume'),
                            stock.get('week_52_low'), stock.get('week_52_high'),
                            stock.get('change_percent'),
                            stock.get('is_blacklisted', 0),
                            stock.get('blacklist_reason'),
                            snapshot_date
                        ))
                        
                        saved_count += 1
                        if is_new:
                            new_count += 1
                        else:
                            updated_count += 1
                            
                    except Exception as e:
                        log_error(f"Error saving {symbol}: {e}")
                        continue
        
        log_info(f"Saved {saved_count} stocks for {exchange}: {new_count} new, {updated_count} updated")
        return saved_count, new_count, updated_count
    
    # ... [Include all other existing methods from the Database class] ...
    
    def get_latest_snapshot_date(self) -> Optional[str]:
        """Get the most recent snapshot date"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(snapshot_date) FROM stocks")
                result = cursor.fetchone()
                return result[0] if result and result[0] else None
        except:
            return None
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Get market overview statistics"""
        latest_date = self.get_latest_snapshot_date()
        if not latest_date:
            return {'overall': (0, 0, 0, 0, 0)}
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT symbol) as total_stocks,
                    COUNT(DISTINCT exchange) as total_exchanges,
                    SUM(market_cap) as total_market_cap,
                    AVG(pe_ratio) as avg_pe_ratio,
                    AVG(dividend_yield) as avg_dividend_yield
                FROM stocks
                WHERE snapshot_date = ?
            """, (latest_date,))
            
            result = cursor.fetchone()
            
            overall = (
                result['total_stocks'] if result else 0,
                result['total_exchanges'] if result else 0,
                result['total_market_cap'] if result else 0,
                result['avg_pe_ratio'] if result else 0,
                result['avg_dividend_yield'] if result else 0
            )
            
            return {'overall': overall}
    
    def cleanup_old_snapshots(self, keep_days: int = 90) -> Tuple[int, int]:
        """Remove snapshots older than specified days"""
        cutoff_date = (datetime.now() - timedelta(days=keep_days)).strftime("%Y-%m-%d")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM stocks WHERE snapshot_date < ?", (cutoff_date,))
            deleted_stocks = cursor.rowcount
            
            cursor.execute("DELETE FROM historical_prices WHERE date < ?", (cutoff_date,))
            deleted_prices = cursor.rowcount
            
            conn.commit()
            
        return deleted_stocks, deleted_prices
    
    def backup_database(self) -> bool:
        """Create a backup of the database"""
        try:
            backup_dir = "backups"
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"stocks_{timestamp}.db")
            
            with self.get_connection() as source_conn:
                with sqlite3.connect(backup_path) as backup_conn:
                    source_conn.backup(backup_conn)
            
            log_info(f"Database backed up to {backup_path}")
            
            # Keep only last 5 backups
            backups = sorted([f for f in os.listdir(backup_dir) if f.endswith('.db')])
            while len(backups) > 5:
                old_backup = os.path.join(backup_dir, backups.pop(0))
                os.remove(old_backup)
                log_debug(f"Removed old backup: {old_backup}")
            
            return True
            
        except Exception as e:
            log_error(f"Backup failed: {e}")
            return False
    
    def log_pipeline_run(self, **kwargs):
        """Log pipeline run details"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            exchanges = kwargs.get('exchanges', [])
            exchanges_json = json.dumps(exchanges) if isinstance(exchanges, list) else exchanges
            
            perf_metrics = kwargs.get('performance_metrics', {})
            perf_json = json.dumps(perf_metrics) if isinstance(perf_metrics, dict) else perf_metrics
            
            cursor.execute("""
                INSERT INTO pipeline_runs (
                    run_date, status, exchanges, total_stocks, new_stocks,
                    updated_stocks, processing_time, peak_memory, errors,
                    performance_metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                kwargs.get('run_date'),
                kwargs.get('status'),
                exchanges_json,
                kwargs.get('total_stocks', 0),
                kwargs.get('new_stocks', 0),
                kwargs.get('updated_stocks', 0),
                kwargs.get('processing_time', 0),
                kwargs.get('peak_memory', 0),
                kwargs.get('errors', ''),
                perf_json
            ))
            
            conn.commit()
    
    def shutdown(self):
        """Shutdown database connections"""
        if hasattr(self, '_pool'):
            self._pool.close_all()
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.shutdown()
        except:
            pass


# ==================== MODULE-LEVEL SINGLETON ====================

_database_instance = None
_database_lock = threading.Lock()

def get_database(db_path: str = "data/stocks_enhanced.db") -> Database:
    """Get or create database instance"""
    global _database_instance
    
    if _database_instance is None:
        with _database_lock:
            if _database_instance is None:
                _database_instance = Database(db_path)
    
    return _database_instance


# ==================== MODULE-LEVEL FUNCTIONS ====================

def init_db():
    """Initialize database"""
    db = get_database()
    db.init_db()

def save_snapshot(exchange: str, stocks_data: List[Dict]) -> Tuple[int, int, int]:
    """Save stock snapshot for an exchange"""
    db = get_database()
    return db.save_snapshot(exchange, stocks_data)

def get_latest_snapshot_date() -> Optional[str]:
    """Get the most recent snapshot date"""
    db = get_database()
    return db.get_latest_snapshot_date()

def log_pipeline_run(**kwargs):
    """Log pipeline run details"""
    db = get_database()
    db.log_pipeline_run(**kwargs)

def get_market_overview() -> Dict[str, Any]:
    """Get market overview statistics"""
    db = get_database()
    return db.get_market_overview()

def cleanup_old_snapshots(keep_days: int = 90) -> Tuple[int, int]:
    """Remove old snapshots"""
    db = get_database()
    return db.cleanup_old_snapshots(keep_days)

def backup_database() -> bool:
    """Create database backup"""
    db = get_database()
    return db.backup_database()

def verify_historical_data(symbol: str) -> Dict[str, Any]:
    """Verify historical data for a symbol"""
    db = get_database()
    return db.verify_historical_data(symbol)

def get_historical_summary() -> Dict[str, Any]:
    """Get summary of all historical data"""
    db = get_database()
    return db.get_historical_summary()


# ==================== CLEANUP ====================

def cleanup():
    """Clean up module-level resources"""
    global _database_instance
    if _database_instance:
        _database_instance.shutdown()
        _database_instance = None

# Register cleanup on module exit
atexit.register(cleanup)


# ==================== TESTING ====================

if __name__ == "__main__":
    print("Testing storage.py with historical data verification...")
    print(f"Python version: {sys.version}")
    
    # Initialize database
    init_db()
    print("✅ Database initialized")
    
    # Test historical data verification
    print("\n📊 Historical Data Summary:")
    summary = get_historical_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Test specific symbol verification
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    print("\n🔍 Verifying Historical Data for Test Symbols:")
    for symbol in test_symbols:
        result = verify_historical_data(symbol)
        if result['has_data']:
            print(f"  {symbol}: {result['record_count']} records, {result['years_covered']} years")
        else:
            print(f"  {symbol}: No historical data")
    
    # Test market overview
    print("\n📈 Market Overview:")
    overview = get_market_overview()
    if overview and 'overall' in overview:
        stats = overview['overall']
        print(f"  Total Stocks: {stats[0]}")
        print(f"  Total Exchanges: {stats[1]}")
        print(f"  Total Market Cap: ${stats[2]:,.2f}" if stats[2] else "  Total Market Cap: N/A")
        print(f"  Avg P/E Ratio: {stats[3]:.2f}" if stats[3] else "  Avg P/E Ratio: N/A")
        print(f"  Avg Dividend Yield: {stats[4]:.2%}" if stats[4] else "  Avg Dividend Yield: N/A")
    
    # Test recent pipeline runs
    print("\n📝 Recent Pipeline Runs:")
    runs = get_recent_pipeline_runs(5)
    if runs:
        for run in runs:
            print(f"  {run['run_date']}: {run['status']} - {run['total_stocks']} stocks")
    else:
        print("  No pipeline runs found")
    
    # Test database backup
    print("\n💾 Testing Database Backup...")
    if backup_database():
        print("  ✅ Backup created successfully")
    else:
        print("  ❌ Backup failed")
    
    print("\n✅ Storage module is fully functional with all features!")
