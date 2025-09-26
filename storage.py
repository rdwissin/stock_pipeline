#!/usr/bin/env python3
"""
Enhanced database operations with thread-safe connection pooling and proper cleanup
Fixed version with connection lifecycle management and transaction safety
"""

__version__ = "1.5.1"
__copyright__ = "Copyright 2025, Richard D. Wissinger"
__author__ = "Richard D. Wissinger"
__email__ = "rick.wissinger@gmail.com"
__license__ = "MIT"
__status__ = "Production"

import sqlite3
import os
import threading
import time
import atexit
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import json
import pickle

from config import (
    DB_PATH, BACKUP_DB_PATH, CHUNK_SIZE, MAX_WORKERS, 
    AUTO_VACUUM_DB, ENABLE_COMPRESSION, MAX_MEMORY_MB
)
from logger import setup_logger, PerformanceTimer, performance_monitor

logger = setup_logger(__name__)

# ===============================================================================
# CUSTOM EXCEPTIONS
# ===============================================================================

class DatabaseError(Exception):
    """Custom exception for database operations"""
    pass

# ===============================================================================
# CONNECTION MANAGEMENT
# ===============================================================================

class ConnectionInfo:
    """Track connection metadata for cleanup"""
    def __init__(self, conn: sqlite3.Connection):
        self.connection = conn
        self.created_at = time.time()
        self.last_used = time.time()
        self.thread_id = threading.get_ident()
        self.transaction_depth = 0
        self.is_closed = False
    
    def touch(self):
        """Update last used time"""
        self.last_used = time.time()
    
    def age(self) -> float:
        """Get age in seconds"""
        return time.time() - self.created_at
    
    def idle_time(self) -> float:
        """Get idle time in seconds"""
        return time.time() - self.last_used

    def close(self):
        """Close the connection safely"""
        if not self.is_closed:
            try:
                if self.transaction_depth > 0:
                    self.connection.rollback()
                # Store the connection id before closing
                conn_id = id(self.connection)
                self.connection.close()
                self.is_closed = True
                # Return the conn_id so it can be removed from tracking
                return conn_id
            except:
                pass
        return None

# ===============================================================================
# ENHANCED DATABASE CLASS
# ===============================================================================

class EnhancedStockDatabase:
    """Enhanced database manager with proper connection pooling and cleanup"""
    
    def __init__(self):
        self.db_path = DB_PATH
        self.lock = threading.RLock()
        self._connection_pool: Dict[str, ConnectionInfo] = {}
        self._init_complete = False
        self._cleanup_thread = None
        self._shutdown = False
        self._shutdown_event = threading.Event()
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        # Register cleanup on exit
        atexit.register(self.shutdown)
        
        # Use regular set for Python 3.13+ compatibility (sqlite3 connections no longer support weak refs)
        self._active_connections = set()

    def _start_cleanup_thread(self):
        """Start background thread for connection cleanup"""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_worker,
                daemon=True,
                name="DB-Cleanup"
            )
            self._cleanup_thread.start()
            logger.debug("Started database connection cleanup thread")
    
    def _cleanup_worker(self):
        """Background worker to clean up stale connections"""
        while not self._shutdown:
            try:
                # Use event wait with timeout for clean shutdown
                if self._shutdown_event.wait(timeout=30):
                    break
                self.cleanup_stale_connections()
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
    
    def cleanup_stale_connections(self, max_age_seconds: int = 3600, max_idle_seconds: int = 300):
        """Clean up stale and idle connections"""
        with self.lock:
            current_threads = set(str(t.ident) for t in threading.enumerate() if t.ident)
            to_remove = []
            
            for thread_id, conn_info in self._connection_pool.items():
                # Remove connections from dead threads
                if thread_id not in current_threads:
                    logger.debug(f"Cleaning connection from dead thread {thread_id}")
                    to_remove.append(thread_id)
                # Remove old connections
                elif conn_info.age() > max_age_seconds:
                    logger.debug(f"Cleaning old connection from thread {thread_id} (age: {conn_info.age():.1f}s)")
                    to_remove.append(thread_id)
                # Remove idle connections
                elif conn_info.idle_time() > max_idle_seconds and conn_info.transaction_depth == 0:
                    logger.debug(f"Cleaning idle connection from thread {thread_id} (idle: {conn_info.idle_time():.1f}s)")
                    to_remove.append(thread_id)
            
            for thread_id in to_remove:
                try:
                    conn_info = self._connection_pool[thread_id]
                    conn_id = conn_info.close()  # Get the connection id when closing
                    if conn_id and conn_id in self._active_connections:
                        self._active_connections.discard(conn_id)  # Remove from tracking
                    del self._connection_pool[thread_id]
                except Exception as e:
                    logger.warning(f"Error closing connection for thread {thread_id}: {e}")

            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} database connections")

    def close_all_connections(self):
        """Close all connections in the pool"""
        with self.lock:
            for thread_id, conn_info in self._connection_pool.items():
                conn_id = conn_info.close()
                if conn_id and conn_id in self._active_connections:
                    self._active_connections.discard(conn_id)
            self._connection_pool.clear()
            self._active_connections.clear()  # Clear the tracking set
            logger.info("Closed all database connections")

    def shutdown(self):
        """Shutdown database manager and cleanup resources"""
        logger.debug("Shutting down database manager")
        self._shutdown = True
        self._shutdown_event.set()
        
        self.close_all_connections()
        
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)

    @contextmanager
    def get_connection(self, thread_id: str = None):
        """Get thread-safe database connection with proper cleanup"""
        if thread_id is None:
            thread_id = str(threading.get_ident())
        
        conn_info = None
        
        try:
            with self.lock:
                # Get or create connection for this thread
                if thread_id not in self._connection_pool:
                    conn = sqlite3.connect(
                        self.db_path,
                        timeout=30.0,
                        check_same_thread=False,
                        isolation_level=None  # Autocommit mode
                    )
                    
                    # Apply optimizations
                    self._optimize_connection(conn)
                    
                    # Create connection info
                    conn_info = ConnectionInfo(conn)
                    self._connection_pool[thread_id] = conn_info

                    # Track connection without weak reference
                    self._active_connections.add(id(conn))  # Store connection id instead of connection itself
                    
                    logger.trace(f"Created new connection for thread {thread_id}")
                else:
                    conn_info = self._connection_pool[thread_id]
                    conn_info.touch()
            
            yield conn_info.connection
            
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            # Rollback if in transaction
            if conn_info and conn_info.transaction_depth > 0:
                try:
                    conn_info.connection.rollback()
                    conn_info.transaction_depth = 0
                except:
                    pass
            raise DatabaseError(f"Cannot connect to database: {e}")
        finally:
            # Update last used time
            if conn_info:
                conn_info.touch()
    
    @contextmanager
    def transaction(self, conn: sqlite3.Connection = None):
        """Fixed context manager for safe transactions with proper rollback"""
        if conn is None:
            # Get connection for current thread
            thread_id = str(threading.get_ident())
            with self.get_connection(thread_id) as conn:
                # Use the transaction with this connection
                with self._transaction_context(conn, thread_id) as cursor:
                    yield cursor
        else:
            thread_id = str(threading.get_ident())
            with self._transaction_context(conn, thread_id) as cursor:
                yield cursor
    
    @contextmanager
    def _transaction_context(self, conn: sqlite3.Connection, thread_id: str):
        """Helper context manager for transaction handling"""
        cursor = conn.cursor()
        conn_info = self._connection_pool.get(thread_id)
        
        try:
            # Start transaction
            cursor.execute("BEGIN TRANSACTION")
            if conn_info:
                conn_info.transaction_depth += 1
            
            yield cursor
            
            # Commit on success
            cursor.execute("COMMIT")
            if conn_info:
                conn_info.transaction_depth = max(0, conn_info.transaction_depth - 1)
            
        except Exception as e:
            # Rollback on any error
            try:
                cursor.execute("ROLLBACK")
                if conn_info:
                    conn_info.transaction_depth = max(0, conn_info.transaction_depth - 1)
            except:
                pass
            raise e
        finally:
            cursor.close()
    
    def _optimize_connection(self, conn: sqlite3.Connection):
        """Apply performance optimizations to connection"""
        optimizations = [
            ("PRAGMA journal_mode=WAL", "WAL mode for better concurrency"),
            ("PRAGMA synchronous=NORMAL", "Normal sync for better performance"),
            ("PRAGMA cache_size=20000", "Larger cache for better performance"),
            ("PRAGMA temp_store=MEMORY", "Memory temp storage"),
            ("PRAGMA mmap_size=268435456", "256MB memory-mapped I/O"),
            ("PRAGMA optimize", "Query planner optimization"),
        ]
        
        if AUTO_VACUUM_DB:
            optimizations.append(("PRAGMA auto_vacuum=INCREMENTAL", "Incremental auto-vacuum"))
        
        for pragma, description in optimizations:
            try:
                conn.execute(pragma)
                logger.trace(f"Applied optimization: {description}")
            except sqlite3.Error as e:
                logger.warning(f"Failed to apply {description}: {e}")
    
    @performance_monitor("init_database")
    def init_db(self) -> None:
        """Initialize database with comprehensive schema including blacklist tracking"""
        if self._init_complete:
            return
            
        logger.info("Initializing enhanced stock database with blacklist support")
        
        with self.get_connection() as conn:
            with self.transaction(conn) as cursor:
                # Main enhanced stocks table with blacklist status
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS stocks (
                        -- Primary Keys
                        symbol TEXT NOT NULL,
                        snapshot_date TEXT NOT NULL,
                        
                        -- Basic Information
                        name TEXT NOT NULL,
                        exchange TEXT NOT NULL,
                        
                        -- Classification
                        sector TEXT,
                        industry TEXT,
                        category TEXT,
                        
                        -- Market Data
                        market_cap REAL DEFAULT 0,
                        enterprise_value REAL,
                        shares_outstanding REAL,
                        float_shares REAL,
                        
                        -- Price Data
                        current_price REAL,
                        previous_close REAL,
                        open_price REAL,
                        day_high REAL,
                        day_low REAL,
                        week_52_high REAL,
                        week_52_low REAL,
                        
                        -- Volume Data
                        volume INTEGER DEFAULT 0,
                        avg_volume_3m INTEGER DEFAULT 0,
                        avg_volume_10d INTEGER DEFAULT 0,
                        
                        -- Financial Ratios
                        pe_ratio REAL,
                        peg_ratio REAL,
                        pb_ratio REAL,
                        ps_ratio REAL,
                        price_to_book REAL,
                        
                        -- Valuation Metrics
                        beta REAL,
                        dividend_yield REAL,
                        dividend_rate REAL,
                        ex_dividend_date TEXT,
                        
                        -- Financial Health
                        debt_to_equity REAL,
                        current_ratio REAL,
                        quick_ratio REAL,
                        return_on_equity REAL,
                        return_on_assets REAL,
                        profit_margin REAL,
                        
                        -- Growth Metrics
                        revenue_growth REAL,
                        earnings_growth REAL,
                        revenue_per_share REAL,
                        book_value_per_share REAL,
                        
                        -- Company Details
                        ipo_year INTEGER,
                        employees INTEGER,
                        description TEXT,
                        website TEXT,
                        country TEXT DEFAULT 'US',
                        
                        -- Trading Info
                        tradeable BOOLEAN DEFAULT 1,
                        shortable BOOLEAN DEFAULT 1,
                        short_ratio REAL,
                        short_percent_outstanding REAL,
                        
                        -- BLACKLIST STATUS
                        is_blacklisted BOOLEAN DEFAULT 0,
                        blacklist_date TEXT,
                        blacklist_reason TEXT,
                        
                        -- Metadata
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_updated TEXT,
                        data_source TEXT DEFAULT 'yahoo',
                        
                        PRIMARY KEY (symbol, snapshot_date)
                    )
                """)
                
                # Blacklist tracking table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS blacklist (
                        symbol TEXT PRIMARY KEY,
                        added_date TEXT NOT NULL,
                        reason TEXT,
                        error_code TEXT,
                        retry_count INTEGER DEFAULT 0,
                        last_retry TEXT,
                        is_active BOOLEAN DEFAULT 1,
                        removed_date TEXT,
                        removal_reason TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Blacklist history table for audit trail
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS blacklist_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        action TEXT NOT NULL, -- 'ADDED', 'RETRIED', 'REMOVED'
                        action_date TEXT NOT NULL,
                        reason TEXT,
                        error_code TEXT,
                        details TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Historical prices table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS historical_prices (
                        symbol TEXT NOT NULL,
                        date TEXT NOT NULL,
                        open_price REAL,
                        high_price REAL,
                        low_price REAL,
                        close_price REAL,
                        adj_close_price REAL,
                        volume INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (symbol, date),
                        FOREIGN KEY (symbol) REFERENCES stocks(symbol)
                    )
                """)
                
                # Pipeline execution tracking
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS pipeline_runs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_date TEXT NOT NULL,
                        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        end_time TIMESTAMP,
                        status TEXT NOT NULL,
                        exchanges_processed TEXT,
                        total_stocks INTEGER DEFAULT 0,
                        total_new_stocks INTEGER DEFAULT 0,
                        total_updated_stocks INTEGER DEFAULT 0,
                        blacklisted_stocks INTEGER DEFAULT 0,
                        processing_time_seconds REAL,
                        memory_peak_mb REAL,
                        errors TEXT,
                        performance_metrics TEXT,
                        UNIQUE(run_date)
                    )
                """)
                
                # Data quality tracking
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS data_quality (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        snapshot_date TEXT NOT NULL,
                        symbol TEXT,
                        issue_type TEXT NOT NULL,
                        issue_description TEXT,
                        severity TEXT DEFAULT 'WARNING',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create comprehensive indexes for performance
                self._create_indexes(cursor)
                
                logger.info("Database schema with blacklist support initialized successfully")
                self._init_complete = True
    
    def _create_indexes(self, cursor: sqlite3.Cursor):
        """Create optimized indexes for fast queries including blacklist indexes"""
        indexes = [
            # Primary lookup indexes
            ("idx_stocks_symbol_date", "stocks", "symbol, snapshot_date DESC"),
            ("idx_stocks_exchange_date", "stocks", "exchange, snapshot_date DESC"),
            ("idx_stocks_sector_date", "stocks", "sector, snapshot_date DESC"),
            ("idx_stocks_industry_date", "stocks", "industry, snapshot_date DESC"),
            
            # Market data indexes
            ("idx_stocks_market_cap", "stocks", "market_cap DESC, snapshot_date DESC"),
            ("idx_stocks_volume", "stocks", "volume DESC, snapshot_date DESC"),
            ("idx_stocks_price", "stocks", "current_price DESC, snapshot_date DESC"),
            
            # Blacklist indexes
            ("idx_stocks_blacklist", "stocks", "is_blacklisted, snapshot_date DESC"),
            ("idx_stocks_blacklist_date", "stocks", "blacklist_date DESC"),
            
            # Ratio analysis indexes
            ("idx_stocks_pe_ratio", "stocks", "pe_ratio ASC, snapshot_date DESC"),
            ("idx_stocks_pb_ratio", "stocks", "pb_ratio ASC, snapshot_date DESC"),
            ("idx_stocks_dividend_yield", "stocks", "dividend_yield DESC, snapshot_date DESC"),
            
            # Performance tracking indexes
            ("idx_stocks_beta", "stocks", "beta DESC, snapshot_date DESC"),
            ("idx_stocks_52w_performance", "stocks", "week_52_high DESC, week_52_low ASC"),
            
            # Trading info indexes
            ("idx_stocks_tradeable", "stocks", "tradeable, snapshot_date DESC"),
            ("idx_stocks_country", "stocks", "country, snapshot_date DESC"),
            
            # Blacklist table indexes
            ("idx_blacklist_active", "blacklist", "is_active, added_date DESC"),
            ("idx_blacklist_error_code", "blacklist", "error_code, added_date DESC"),
            ("idx_blacklist_retry", "blacklist", "retry_count, last_retry DESC"),
            
            # Blacklist history indexes
            ("idx_blacklist_hist_symbol", "blacklist_history", "symbol, action_date DESC"),
            ("idx_blacklist_hist_action", "blacklist_history", "action, action_date DESC"),
            
            # Historical prices indexes
            ("idx_hist_symbol_date", "historical_prices", "symbol, date DESC"),
            ("idx_hist_date_volume", "historical_prices", "date DESC, volume DESC"),
            
            # Pipeline tracking indexes
            ("idx_pipeline_runs_date", "pipeline_runs", "run_date DESC"),
            ("idx_pipeline_runs_status", "pipeline_runs", "status, run_date DESC"),
            
            # Data quality indexes
            ("idx_quality_date", "data_quality", "snapshot_date DESC"),
            ("idx_quality_symbol", "data_quality", "symbol, snapshot_date DESC"),
            ("idx_quality_severity", "data_quality", "severity, snapshot_date DESC"),
        ]
        
        for index_name, table_name, columns in indexes:
            try:
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name} 
                    ON {table_name}({columns})
                """)
                logger.trace(f"Created index: {index_name}")
            except sqlite3.Error as e:
                logger.warning(f"Failed to create index {index_name}: {e}")
    
    @performance_monitor("sync_blacklist")
    def sync_blacklist_to_database(self) -> Tuple[int, int]:
        """Sync blacklist from blacklist.py to database with proper transaction handling"""
        from blacklist import get_blacklist
        
        blacklist = get_blacklist()
        blacklist_data = blacklist._blacklist_data
        
        added_count = 0
        removed_count = 0
        
        with self.get_connection() as conn:
            with self.transaction(conn) as cursor:
                # Get current database blacklist
                cursor.execute("SELECT symbol FROM blacklist WHERE is_active = 1")
                db_blacklisted = set(row[0] for row in cursor.fetchall())
                
                # Get file blacklist
                file_blacklisted = set(blacklist_data.get('symbols', {}).keys())
                
                # Add new blacklisted symbols
                to_add = file_blacklisted - db_blacklisted
                for symbol in to_add:
                    info = blacklist_data['symbols'][symbol]
                    cursor.execute("""
                        INSERT OR REPLACE INTO blacklist 
                        (symbol, added_date, reason, error_code, retry_count, last_retry, is_active)
                        VALUES (?, ?, ?, ?, ?, ?, 1)
                    """, (
                        symbol,
                        info.get('added_date', datetime.now().isoformat()),
                        info.get('reason', 'Unknown'),
                        info.get('error_code', 'UNKNOWN'),
                        info.get('retry_count', 0),
                        info.get('last_retry')
                    ))
                    
                    # Add to history
                    cursor.execute("""
                        INSERT INTO blacklist_history 
                        (symbol, action, action_date, reason, error_code)
                        VALUES (?, 'ADDED', ?, ?, ?)
                    """, (
                        symbol,
                        datetime.now().isoformat(),
                        info.get('reason', 'Unknown'),
                        info.get('error_code', 'UNKNOWN')
                    ))
                    
                    # Update stocks table
                    cursor.execute("""
                        UPDATE stocks 
                        SET is_blacklisted = 1, 
                            blacklist_date = ?,
                            blacklist_reason = ?
                        WHERE symbol = ?
                    """, (
                        datetime.now().isoformat(),
                        info.get('reason', 'Unknown'),
                        symbol
                    ))
                    
                    added_count += 1
                
                # Remove symbols no longer blacklisted
                to_remove = db_blacklisted - file_blacklisted
                for symbol in to_remove:
                    cursor.execute("""
                        UPDATE blacklist 
                        SET is_active = 0, 
                            removed_date = ?,
                            removal_reason = 'Removed from file blacklist'
                        WHERE symbol = ?
                    """, (datetime.now().isoformat(), symbol))
                    
                    # Add to history
                    cursor.execute("""
                        INSERT INTO blacklist_history 
                        (symbol, action, action_date, reason)
                        VALUES (?, 'REMOVED', ?, 'Removed from file blacklist')
                    """, (symbol, datetime.now().isoformat()))
                    
                    # Update stocks table
                    cursor.execute("""
                        UPDATE stocks 
                        SET is_blacklisted = 0,
                            blacklist_date = NULL,
                            blacklist_reason = NULL
                        WHERE symbol = ?
                    """, (symbol,))
                    
                    removed_count += 1
                
                logger.info(f"Synced blacklist: {added_count} added, {removed_count} removed")
        
        return added_count, removed_count
    
    @performance_monitor("save_stock_snapshot")
    def save_snapshot(self, exchange: str, stocks_data: List[Dict[str, Any]]) -> Tuple[int, int, int]:
        """
        Save stock data snapshot with proper transaction handling
        Returns: (total_saved, new_stocks, updated_stocks)
        """
        if not stocks_data:
            logger.warning(f"No data to save for {exchange}")
            return 0, 0, 0
        
        # Import blacklist
        from blacklist import get_blacklist
        blacklist = get_blacklist()
        
        logger.info(f"Saving snapshot for {exchange}: {len(stocks_data)} stocks")
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        saved_count = 0
        new_count = 0
        updated_count = 0
        blacklisted_count = 0
        quality_issues = []
        
        with self.get_connection() as conn:
            # Process in chunks for better performance
            chunks = [stocks_data[i:i + CHUNK_SIZE] for i in range(0, len(stocks_data), CHUNK_SIZE)]
            
            for chunk_num, chunk in enumerate(chunks):
                logger.trace(f"Processing chunk {chunk_num + 1}/{len(chunks)}")
                
                with self.transaction(conn) as cursor:
                    for stock in chunk:
                        # Check if symbol is blacklisted
                        symbol = stock.get('symbol', '').strip().upper()
                        is_blacklisted = blacklist.is_blacklisted(symbol)
                        
                        if is_blacklisted:
                            blacklisted_count += 1
                            stock['is_blacklisted'] = True
                            stock['blacklist_reason'] = 'In blacklist'
                        
                        result = self._save_single_stock(cursor, stock, date_str, is_blacklisted)
                        if result:
                            saved_count += 1
                            if result == 'new':
                                new_count += 1
                            else:
                                updated_count += 1
                            
                            # Data quality checks
                            issues = self._validate_stock_data(stock, date_str)
                            quality_issues.extend(issues)
                    
                    logger.trace(f"Committed chunk {chunk_num + 1}")
            
            # Save data quality issues
            if quality_issues:
                self._save_quality_issues(conn, quality_issues)
        
        logger.info(f"Saved {saved_count} stocks for {exchange} on {date_str} "
                   f"(New: {new_count}, Updated: {updated_count}, Blacklisted: {blacklisted_count})")
        
        if quality_issues:
            logger.warning(f"Found {len(quality_issues)} data quality issues")
        
        return saved_count, new_count, updated_count
    
    def _save_single_stock(self, cursor: sqlite3.Cursor, stock: Dict[str, Any], 
                          date_str: str, is_blacklisted: bool = False) -> Optional[str]:
        """Save a single stock record with blacklist status"""
        try:
            symbol = stock.get('symbol', '').strip().upper()
            if not symbol:
                return None
            
            # Check if record exists
            cursor.execute(
                "SELECT COUNT(*) FROM stocks WHERE symbol = ? AND snapshot_date = ?",
                (symbol, date_str)
            )
            exists = cursor.fetchone()[0] > 0
            
            # Prepare data with blacklist fields
            stock_data = self._prepare_stock_data(stock, date_str)
            stock_data['is_blacklisted'] = is_blacklisted
            if is_blacklisted:
                stock_data['blacklist_date'] = datetime.now().isoformat()
                stock_data['blacklist_reason'] = stock.get('blacklist_reason', 'In blacklist')
            
            if exists:
                # Update existing record
                update_fields = []
                update_values = []
                
                for field, value in stock_data.items():
                    if field not in ['symbol', 'snapshot_date', 'created_at']:
                        update_fields.append(f"{field} = ?")
                        update_values.append(value)
                
                update_values.extend([symbol, date_str])
                
                cursor.execute(f"""
                    UPDATE stocks 
                    SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP
                    WHERE symbol = ? AND snapshot_date = ?
                """, update_values)
                
                return 'updated'
            else:
                # Insert new record
                fields = list(stock_data.keys())
                placeholders = ', '.join(['?' for _ in fields])
                values = list(stock_data.values())
                
                cursor.execute(f"""
                    INSERT INTO stocks ({', '.join(fields)})
                    VALUES ({placeholders})
                """, values)
                
                return 'new'
                
        except sqlite3.Error as e:
            symbol = stock.get('symbol', 'unknown')
            logger.warning(f"Error saving stock {symbol}: {e}")
            return None
    
    def _prepare_stock_data(self, stock: Dict[str, Any], date_str: str) -> Dict[str, Any]:
        """Prepare stock data for database insertion with all fields including blacklist"""
        # Define all possible fields with defaults
        field_defaults = {
            'symbol': '',
            'snapshot_date': date_str,
            'name': '',
            'exchange': '',
            'sector': 'Unknown',
            'industry': 'Unknown',
            'category': 'Common Stock',
            'market_cap': 0.0,
            'enterprise_value': None,
            'shares_outstanding': None,
            'float_shares': None,
            'current_price': None,
            'previous_close': None,
            'open_price': None,
            'day_high': None,
            'day_low': None,
            'week_52_high': None,
            'week_52_low': None,
            'volume': 0,
            'avg_volume_3m': 0,
            'avg_volume_10d': 0,
            'pe_ratio': None,
            'peg_ratio': None,
            'pb_ratio': None,
            'ps_ratio': None,
            'price_to_book': None,
            'beta': None,
            'dividend_yield': None,
            'dividend_rate': None,
            'ex_dividend_date': None,
            'debt_to_equity': None,
            'current_ratio': None,
            'quick_ratio': None,
            'return_on_equity': None,
            'return_on_assets': None,
            'profit_margin': None,
            'revenue_growth': None,
            'earnings_growth': None,
            'revenue_per_share': None,
            'book_value_per_share': None,
            'ipo_year': None,
            'employees': None,
            'description': '',
            'website': '',
            'country': 'US',
            'tradeable': True,
            'shortable': True,
            'short_ratio': None,
            'short_percent_outstanding': None,
            'is_blacklisted': False,
            'blacklist_date': None,
            'blacklist_reason': None,
            'last_updated': stock.get('last_updated', datetime.now().isoformat()),
            'data_source': stock.get('data_source', 'yahoo')
        }
        
        # Merge provided data with defaults
        result = {}
        for field, default in field_defaults.items():
            value = stock.get(field, default)
            
            # Handle None values and type conversion
            if value is None:
                result[field] = None
            elif field in ['symbol', 'name'] and not value:
                # Required fields
                result[field] = stock.get(field, default) or default
            else:
                result[field] = value
        
        return result
    
    def _validate_stock_data(self, stock: Dict[str, Any], date_str: str) -> List[Dict[str, str]]:
        """Validate stock data and return quality issues"""
        issues = []
        symbol = stock.get('symbol', 'unknown')
        
        # Check for missing critical data
        if not stock.get('name'):
            issues.append({
                'symbol': symbol,
                'snapshot_date': date_str,
                'issue_type': 'missing_name',
                'issue_description': 'Company name is missing',
                'severity': 'WARNING'
            })
        
        # Check for unrealistic market cap
        market_cap = stock.get('market_cap', 0)
        if market_cap and (market_cap > 10e12 or market_cap < 0):  # >$10T or negative
            issues.append({
                'symbol': symbol,
                'snapshot_date': date_str,
                'issue_type': 'unrealistic_market_cap',
                'issue_description': f'Market cap seems unrealistic: ${market_cap:,.0f}',
                'severity': 'ERROR'
            })
        
        # Check for missing price data
        if not stock.get('current_price'):
            issues.append({
                'symbol': symbol,
                'snapshot_date': date_str,
                'issue_type': 'missing_price',
                'issue_description': 'Current price is missing',
                'severity': 'WARNING'
            })
        
        # Check if blacklisted
        if stock.get('is_blacklisted'):
            issues.append({
                'symbol': symbol,
                'snapshot_date': date_str,
                'issue_type': 'blacklisted',
                'issue_description': f'Symbol is blacklisted: {stock.get("blacklist_reason", "Unknown reason")}',
                'severity': 'INFO'
            })
        
        return issues
    
    def _save_quality_issues(self, conn: sqlite3.Connection, issues: List[Dict[str, str]]):
        """Save data quality issues to database with proper transaction"""
        with self.transaction(conn) as cursor:
            for issue in issues:
                cursor.execute("""
                    INSERT INTO data_quality 
                    (snapshot_date, symbol, issue_type, issue_description, severity)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    issue['snapshot_date'], issue['symbol'], issue['issue_type'],
                    issue['issue_description'], issue['severity']
                ))
            
            logger.debug(f"Saved {len(issues)} data quality issues")

# ===============================================================================
# PUBLIC FUNCTIONS
# ===============================================================================

@performance_monitor("get_blacklisted_stocks")
def get_blacklisted_stocks(snapshot_date: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all blacklisted stocks with details"""
    db = get_database()
    
    if not snapshot_date:
        snapshot_date = get_latest_snapshot_date()
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    s.symbol,
                    s.name,
                    s.exchange,
                    s.blacklist_date,
                    s.blacklist_reason,
                    b.error_code,
                    b.retry_count,
                    b.last_retry
                FROM stocks s
                LEFT JOIN blacklist b ON s.symbol = b.symbol
                WHERE s.is_blacklisted = 1 
                  AND s.snapshot_date = ?
                ORDER BY s.blacklist_date DESC
            """, (snapshot_date,))
            
            columns = [description[0] for description in cursor.description]
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            return results
            
        except sqlite3.Error as e:
            logger.error(f"Error getting blacklisted stocks: {e}")
            return []

@performance_monitor("query_by_symbol")
def query_by_symbol(symbol: str, limit: int = 10, exclude_blacklisted: bool = True) -> List[Tuple]:
    """Query stock data by symbol with proper parameterization"""
    db = get_database()
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        try:
            if exclude_blacklisted:
                cursor.execute("""
                    SELECT * FROM stocks
                    WHERE symbol = ? AND is_blacklisted = 0
                    ORDER BY snapshot_date DESC
                    LIMIT ?
                """, (symbol.upper(), limit))
            else:
                cursor.execute("""
                    SELECT * FROM stocks
                    WHERE symbol = ?
                    ORDER BY snapshot_date DESC
                    LIMIT ?
                """, (symbol.upper(), limit))
            
            results = cursor.fetchall()
            logger.debug(f"Found {len(results)} records for symbol {symbol}")
            return results
            
        except sqlite3.Error as e:
            logger.error(f"Database error querying symbol {symbol}: {e}")
            return []

@performance_monitor("get_latest_snapshot_date")
def get_latest_snapshot_date() -> Optional[str]:
    """Get the most recent snapshot date"""
    db = get_database()
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT MAX(snapshot_date) FROM stocks")
            result = cursor.fetchone()
            return result[0] if result[0] else None
        except sqlite3.Error as e:
            logger.error(f"Error getting latest snapshot date: {e}")
            return None

@performance_monitor("get_market_overview")
def get_market_overview() -> Dict[str, Any]:
    """Get market overview statistics"""
    db = get_database()
    latest_date = get_latest_snapshot_date()
    
    if not latest_date:
        return {}
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        try:
            # Get overall market stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_stocks,
                    COUNT(CASE WHEN market_cap > 0 THEN 1 END) as stocks_with_cap,
                    SUM(CASE WHEN market_cap > 0 THEN market_cap END) as total_market_cap,
                    AVG(CASE WHEN pe_ratio > 0 AND pe_ratio < 1000 THEN pe_ratio END) as avg_pe,
                    AVG(CASE WHEN dividend_yield > 0 THEN dividend_yield END) as avg_dividend
                FROM stocks
                WHERE snapshot_date = ?
            """, (latest_date,))
            
            overall = cursor.fetchone()
            
            return {'overall': overall}
            
        except sqlite3.Error as e:
            logger.error(f"Error getting market overview: {e}")
            return {}

@performance_monitor("cleanup_old_data")
def cleanup_old_snapshots(keep_days: int = 30) -> Tuple[int, int]:
    """Clean up old snapshot data beyond specified days with transaction safety"""
    db = get_database()
    cutoff_date = datetime.now() - timedelta(days=keep_days)
    cutoff_str = cutoff_date.strftime("%Y-%m-%d")
    
    with db.get_connection() as conn:
        with db.transaction(conn) as cursor:
            # Clean stocks data
            cursor.execute("DELETE FROM stocks WHERE snapshot_date < ?", (cutoff_str,))
            stocks_deleted = cursor.rowcount
            
            # Clean historical prices
            cursor.execute("DELETE FROM historical_prices WHERE date < ?", (cutoff_str,))
            prices_deleted = cursor.rowcount
            
            # Clean data quality records
            cursor.execute("DELETE FROM data_quality WHERE snapshot_date < ?", (cutoff_str,))
            quality_deleted = cursor.rowcount
            
            # Clean old blacklist history (keep 180 days)
            old_history_cutoff = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
            cursor.execute("DELETE FROM blacklist_history WHERE action_date < ?", (old_history_cutoff,))
            history_deleted = cursor.rowcount
            
            logger.info(f"Cleaned up old data: {stocks_deleted} stock records, "
                       f"{prices_deleted} price records, {quality_deleted} quality records, "
                       f"{history_deleted} blacklist history records")
            
            return stocks_deleted, prices_deleted

@performance_monitor("backup_database")
def backup_database() -> bool:
    """Create a backup of the database"""
    try:
        import shutil
        shutil.copy2(DB_PATH, BACKUP_DB_PATH)
        logger.info(f"Database backed up to {BACKUP_DB_PATH}")
        return True
    except Exception as e:
        logger.error(f"Failed to backup database: {e}")
        return False

@performance_monitor("log_pipeline_run")
def log_pipeline_run(run_date: str, status: str, exchanges: List[str], 
                    total_stocks: int = 0, new_stocks: int = 0, updated_stocks: int = 0,
                    blacklisted_stocks: int = 0, processing_time: float = 0, 
                    peak_memory: float = 0, errors: str = "", 
                    performance_metrics: Dict[str, Any] = None) -> None:
    """Log comprehensive pipeline run information including blacklist stats"""
    db = get_database()
    
    with db.get_connection() as conn:
        with db.transaction(conn) as cursor:
            metrics_json = json.dumps(performance_metrics) if performance_metrics else ""
            
            cursor.execute("""
                INSERT OR REPLACE INTO pipeline_runs 
                (run_date, status, exchanges_processed, total_stocks, total_new_stocks,
                 total_updated_stocks, blacklisted_stocks, processing_time_seconds, 
                 memory_peak_mb, errors, performance_metrics, end_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                run_date, status, ",".join(exchanges), total_stocks, new_stocks,
                updated_stocks, blacklisted_stocks, processing_time, peak_memory, 
                errors, metrics_json
            ))
            
            logger.debug(f"Logged pipeline run for {run_date}")

# ===============================================================================
# DATABASE SINGLETON
# ===============================================================================

_db_instance = None
_db_lock = threading.Lock()

def get_database() -> EnhancedStockDatabase:
    """Get singleton database instance with thread safety"""
    global _db_instance
    if _db_instance is None:
        with _db_lock:
            if _db_instance is None:
                _db_instance = EnhancedStockDatabase()
    return _db_instance

@performance_monitor("init_db_wrapper")
def init_db() -> None:
    """Initialize database (public interface)"""
    db = get_database()
    db.init_db()
    # Sync blacklist on initialization
    db.sync_blacklist_to_database()

@performance_monitor("save_snapshot_wrapper")
def save_snapshot(exchange: str, data: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    """
    Save stock data snapshot (public interface)
    Returns: (total_saved, new_stocks, updated_stocks)
    """
    db = get_database()
    saved_count, new_count, updated_count = db.save_snapshot(exchange, data)
    return saved_count, new_count, updated_count

# Cleanup on module unload
@atexit.register
def cleanup_on_exit():
    """Cleanup database connections on exit"""
    global _db_instance
    if _db_instance:
        _db_instance.shutdown()
        _db_instance = None