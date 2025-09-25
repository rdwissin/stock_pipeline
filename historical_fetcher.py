#!/usr/bin/env python3
"""
Historical data fetcher with proper exponential backoff and database locking
Fixed version with improved retry logic and connection handling
"""

__version__ = "1.5.0"
__copyright__ = "Copyright 2025, Richard D. Wissinger"
__author__ = "Richard D. Wissinger"
__email__ = "rick.wissinger@gmail.com"
__license__ = "MIT"
__status__ = "Production"

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import time
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import sys
import signal
from pathlib import Path
import random
import math

# Add the project directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from logger import setup_logger, PerformanceTimer, performance_monitor
from storage import get_database
from blacklist import get_blacklist

logger = setup_logger(__name__)

# Configuration for historical data limits
from config import MAX_WORKERS, MAX_YEARS_HISTORY, MAX_RETRIES_ON_LOCK, BASE_RETRY_DELAY, MAX_RETRY_DELAY

class ExponentialBackoff:
    """Exponential backoff retry handler with jitter"""
    
    def __init__(self, base_delay: float = BASE_RETRY_DELAY, 
                 max_delay: float = MAX_RETRY_DELAY,
                 max_retries: int = MAX_RETRIES_ON_LOCK):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        self.attempt = 0
        
    def wait(self) -> float:
        """Calculate and apply exponential backoff with jitter"""
        if self.attempt >= self.max_retries:
            raise Exception(f"Max retries ({self.max_retries}) exceeded")
            
        # Exponential backoff: delay = base * (2 ^ attempt)
        delay = min(self.base_delay * (2 ** self.attempt), self.max_delay)
        
        # Add jitter (randomization) to prevent thundering herd
        jitter = random.uniform(0, delay * 0.3)  # Up to 30% jitter
        actual_delay = delay + jitter
        
        logger.debug(f"Retry attempt {self.attempt + 1}/{self.max_retries}, "
                    f"waiting {actual_delay:.3f}s")
        
        time.sleep(actual_delay)
        self.attempt += 1
        
        return actual_delay
    
    def reset(self):
        """Reset the backoff counter"""
        self.attempt = 0
    
    def can_retry(self) -> bool:
        """Check if more retries are available"""
        return self.attempt < self.max_retries

class HistoricalDataFetcher:
    """Fetch historical data with improved retry logic and connection handling"""
    
    def __init__(self, max_years: int = MAX_YEARS_HISTORY):
        """Initialize fetcher with configurable history limit"""
        self.max_years = max_years
        self.start_date = (datetime.now() - timedelta(days=365 * max_years)).strftime('%Y-%m-%d')
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.db = get_database()
        self.blacklist = get_blacklist()
        self.failed_symbols = []
        self.lock = threading.Lock()
        self.shutdown_event = None
        self.db_lock = threading.Lock()  # Additional lock for database operations
        self.rate_limiter = threading.Semaphore(5)  # Limit concurrent Yahoo requests
        
        logger.info(f"Historical fetcher initialized with {max_years}-year data collection")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        logger.info(f"Expected data volume: ~{max_years * 252} trading days per symbol")
        
    @performance_monitor("fetch_stock_history")
    def fetch_stock_history(self, symbol: str, period: str = None) -> Optional[pd.DataFrame]:
        """
        Fetch historical data with rate limiting and retry logic
        
        Args:
            symbol: Stock symbol
            period: Period string or None to use date range
        """
        # Check if shutdown requested
        if self.shutdown_event and self.shutdown_event.is_set():
            logger.debug(f"Shutdown requested, skipping {symbol}")
            return None
            
        # Check if symbol is blacklisted
        if self.blacklist.is_blacklisted(symbol):
            logger.debug(f"Skipping blacklisted symbol: {symbol}")
            return None
        
        # Apply rate limiting
        with self.rate_limiter:
            backoff = ExponentialBackoff(base_delay=0.5, max_delay=10.0, max_retries=3)
            
            while backoff.can_retry():
                try:
                    logger.debug(f"Fetching {self.max_years}-year history for {symbol}")
                    
                    # Use yfinance to get historical data
                    ticker = yf.Ticker(symbol)
                    
                    # For long-term data, use explicit date range for accuracy
                    hist = ticker.history(
                        start=self.start_date,
                        end=self.end_date,
                        auto_adjust=False,
                        actions=True,
                        prepost=False,
                        repair=True,
                        keepna=False,
                        raise_errors=False
                    )
                    
                    if hist.empty:
                        # Try with period as fallback
                        logger.debug(f"No data with date range, trying max period for {symbol}")
                        hist = ticker.history(
                            period="max",
                            auto_adjust=False,
                            actions=True,
                            repair=True,
                            keepna=False
                        )
                        
                        # If we got data, trim to our max_years limit
                        if not hist.empty:
                            cutoff_date = datetime.now() - timedelta(days=365 * self.max_years)
                            hist = hist[hist.index >= cutoff_date]
                    
                    if hist.empty:
                        logger.warning(f"No historical data available for {symbol}")
                        self.blacklist.add_symbol(symbol, "No historical data available", "NO_HISTORY")
                        with self.lock:
                            self.failed_symbols.append(symbol)
                        return None
                    
                    # Get actual date range
                    actual_start = hist.index[0]
                    actual_end = hist.index[-1]
                    days_of_history = (actual_end - actual_start).days
                    years_of_history = days_of_history / 365.25
                    
                    logger.debug(f"{symbol}: Retrieved {len(hist)} trading days "
                                f"({years_of_history:.1f} years)")
                    
                    # Clean data
                    original_len = len(hist)
                    hist = hist.dropna(subset=['Close'])
                    
                    if len(hist) < original_len:
                        logger.trace(f"{symbol}: Dropped {original_len - len(hist)} rows with missing Close")
                    
                    if hist.empty:
                        logger.warning(f"No valid historical data for {symbol} after cleaning")
                        return None
                        
                    # Reset index to get Date as column
                    hist = hist.reset_index()
                    hist['Symbol'] = symbol
                    
                    # Add basic technical indicators
                    if len(hist) >= 20:
                        hist['SMA_20'] = hist['Close'].rolling(window=20, min_periods=20).mean()
                    if len(hist) >= 50:
                        hist['SMA_50'] = hist['Close'].rolling(window=50, min_periods=50).mean()
                    if len(hist) >= 200:
                        hist['SMA_200'] = hist['Close'].rolling(window=200, min_periods=200).mean()
                    
                    # Add year-over-year metrics
                    if len(hist) >= 252:
                        hist['YoY_Return'] = hist['Close'].pct_change(periods=252) * 100
                    
                    logger.trace(f"Successfully fetched {len(hist)} days for {symbol}")
                    return hist
                    
                except Exception as e:
                    error_str = str(e)
                    
                    # Check if it's a 404 error
                    if '404' in error_str or 'not found' in error_str.lower():
                        self.blacklist.add_symbol(symbol, f"Fetch failed: {error_str[:100]}", "404")
                        logger.warning(f"Symbol {symbol} not found, blacklisted")
                        return None
                    
                    # Check if it's a rate limit error
                    if '429' in error_str or 'rate' in error_str.lower():
                        logger.warning(f"Rate limited for {symbol}, backing off")
                        backoff.wait()
                        continue
                    
                    # Other errors - retry with backoff
                    if backoff.can_retry():
                        logger.debug(f"Error fetching {symbol}: {e}, retrying...")
                        backoff.wait()
                        continue
                    else:
                        logger.warning(f"Failed to fetch history for {symbol} after retries: {e}")
                        with self.lock:
                            self.failed_symbols.append(symbol)
                        return None
            
            # If we exhausted retries
            logger.error(f"Failed to fetch {symbol} after all retries")
            with self.lock:
                self.failed_symbols.append(symbol)
            return None
    
    @performance_monitor("save_historical_data")
    def save_historical_data(self, symbol: str, data: pd.DataFrame, 
                           update_mode: str = "replace") -> int:
        """
        Save historical data with exponential backoff retry for database locks
        """
        if data is None or data.empty:
            return 0
            
        saved_count = 0
        backoff = ExponentialBackoff(base_delay=BASE_RETRY_DELAY, 
                                    max_delay=MAX_RETRY_DELAY,
                                    max_retries=MAX_RETRIES_ON_LOCK)
        
        while backoff.can_retry():
            try:
                with self.db_lock:
                    with self.db.get_connection() as conn:
                        # Set busy timeout to handle locks better
                        conn.execute("PRAGMA busy_timeout = 15000")  # 15 second timeout
                        
                        # Use transaction context manager from fixed storage.py
                        with self.db.transaction(conn) as cursor:
                            if update_mode == "replace":
                                # Delete existing data for this symbol
                                cursor.execute("DELETE FROM historical_prices WHERE symbol = ?", (symbol,))
                                deleted = cursor.rowcount
                                if deleted > 0:
                                    logger.trace(f"Cleared {deleted} existing records for {symbol}")
                            
                            # Prepare batch insert - larger batches for better performance
                            rows_to_insert = []
                            batch_size = 1000
                            
                            for _, row in data.iterrows():
                                # Check for shutdown periodically
                                if saved_count % 500 == 0 and self.shutdown_event and self.shutdown_event.is_set():
                                    if rows_to_insert:
                                        self._insert_batch(cursor, rows_to_insert)
                                    logger.info(f"Shutdown requested, saved {saved_count} records for {symbol}")
                                    return saved_count
                                
                                date_str = row['Date'].strftime('%Y-%m-%d') if hasattr(row['Date'], 'strftime') else str(row['Date'])[:10]
                                
                                row_data = (
                                    symbol,
                                    date_str,
                                    float(row['Open']) if pd.notna(row.get('Open')) else None,
                                    float(row['High']) if pd.notna(row.get('High')) else None,
                                    float(row['Low']) if pd.notna(row.get('Low')) else None,
                                    float(row['Close']) if pd.notna(row.get('Close')) else None,
                                    float(row.get('Adj Close', row['Close'])) if 'Adj Close' in row.index else float(row['Close']),
                                    int(row['Volume']) if pd.notna(row.get('Volume', 0)) else 0
                                )
                                
                                rows_to_insert.append(row_data)
                                saved_count += 1
                                
                                # Insert in batches
                                if len(rows_to_insert) >= batch_size:
                                    self._insert_batch(cursor, rows_to_insert)
                                    rows_to_insert = []
                            
                            # Insert remaining rows
                            if rows_to_insert:
                                self._insert_batch(cursor, rows_to_insert)
                            
                            # Log success
                            date_min = data['Date'].min()
                            date_max = data['Date'].max()
                            years_saved = (date_max - date_min).days / 365.25
                            logger.debug(f"Saved {saved_count} records for {symbol} ({years_saved:.1f} years)")
                            
                            return saved_count
                            
            except sqlite3.OperationalError as e:
                if "locked" in str(e):
                    if backoff.can_retry():
                        logger.warning(f"Database locked for {symbol}, retrying with backoff...")
                        backoff.wait()
                        continue
                    else:
                        logger.error(f"Failed to save {symbol} after {backoff.max_retries} retries: {e}")
                        return 0
                else:
                    logger.error(f"Database error for {symbol}: {e}")
                    return 0
                    
            except Exception as e:
                logger.error(f"Failed to save historical data for {symbol}: {e}")
                return 0
        
        return 0
    
    def _insert_batch(self, cursor: sqlite3.Cursor, rows: List[Tuple]):
        """Insert a batch of rows with proper parameterization"""
        cursor.executemany("""
            INSERT OR REPLACE INTO historical_prices
            (symbol, date, open_price, high_price, low_price, 
             close_price, adj_close_price, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
    
    @performance_monitor("fetch_all_historical")
    def fetch_all_historical_data(self, symbols: List[str], parallel: bool = True,
                                batch_size: int = 10, shutdown_event=None) -> Dict[str, int]:
        """
        Fetch historical data for all symbols with improved parallel processing
        """
        self.shutdown_event = shutdown_event
        
        # Filter out blacklisted symbols
        valid_symbols = self.blacklist.filter_valid_symbols(symbols)
        blacklisted_count = len(symbols) - len(valid_symbols)
        
        if blacklisted_count > 0:
            logger.info(f"Skipping {blacklisted_count} blacklisted symbols")
        
        logger.info("=" * 60)
        logger.info(f"FETCHING {self.max_years}-YEAR HISTORICAL DATA")
        logger.info(f"Processing {len(valid_symbols)} symbols")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        logger.info(f"Expected volume: ~{len(valid_symbols) * self.max_years * 252:,} total records")
        logger.info("=" * 60)
        
        results = {}
        self.failed_symbols = []
        
        if parallel:
            # Use adaptive worker count based on system resources
            safe_workers = min(MAX_WORKERS // 2, 4)
            
            with ThreadPoolExecutor(max_workers=safe_workers) as executor:
                future_to_symbol = {}
                
                # Submit tasks with rate limiting
                for i, symbol in enumerate(valid_symbols):
                    if shutdown_event and shutdown_event.is_set():
                        break
                    
                    # Stagger submissions to avoid overwhelming the API
                    if i > 0 and i % 10 == 0:
                        time.sleep(1)  # Pause every 10 symbols
                    
                    future = executor.submit(self._fetch_and_save_with_retry, symbol)
                    future_to_symbol[future] = symbol
                
                # Process completed futures
                completed = 0
                for future in as_completed(future_to_symbol):
                    if shutdown_event and shutdown_event.is_set():
                        logger.info("Shutdown requested, stopping historical fetch")
                        # Cancel remaining futures
                        for f in future_to_symbol:
                            f.cancel()
                        break
                        
                    symbol = future_to_symbol[future]
                    try:
                        saved_count = future.result(timeout=120)
                        results[symbol] = saved_count
                        completed += 1
                        
                        # Progress update
                        if completed % 25 == 0 or completed == len(valid_symbols):
                            success_rate = (completed - len(self.failed_symbols)) / completed * 100
                            total_records = sum(results.values())
                            avg_records = total_records / completed if completed > 0 else 0
                            logger.info(f"Progress: {completed}/{len(valid_symbols)} symbols "
                                      f"({success_rate:.1f}% success, "
                                      f"{total_records:,} total records, "
                                      f"{avg_records:.0f} avg/symbol)")
                            
                    except Exception as e:
                        logger.warning(f"Failed to process {symbol}: {e}")
                        results[symbol] = 0
        else:
            # Sequential processing with progress updates
            for i, symbol in enumerate(valid_symbols):
                if shutdown_event and shutdown_event.is_set():
                    logger.info("Shutdown requested")
                    break
                    
                saved_count = self._fetch_and_save_with_retry(symbol)
                results[symbol] = saved_count
                
                if (i + 1) % 25 == 0:
                    total_records = sum(results.values())
                    avg_records = total_records / (i + 1)
                    logger.info(f"Progress: {i + 1}/{len(valid_symbols)} symbols "
                              f"({total_records:,} records, {avg_records:.0f} avg/symbol)")
                
                # Rate limiting
                time.sleep(0.2)
        
        # Summary
        successful = sum(1 for v in results.values() if v > 0)
        total_records = sum(results.values())
        avg_records_per_symbol = total_records / successful if successful > 0 else 0
        
        logger.info("=" * 60)
        logger.info(f"{self.max_years}-YEAR HISTORICAL DATA FETCH COMPLETE")
        logger.info("=" * 60)
        logger.info(f"  Total symbols processed: {len(results)}")
        logger.info(f"  Successful downloads: {successful}")
        logger.info(f"  Failed downloads: {len(self.failed_symbols)}")
        logger.info(f"  Blacklisted symbols: {blacklisted_count}")
        logger.info(f"  Total records saved: {total_records:,}")
        logger.info(f"  Average records per symbol: {avg_records_per_symbol:.0f}")
        logger.info(f"  Data coverage: Up to {self.max_years} years per symbol")
        
        if self.failed_symbols:
            logger.info(f"  Failed symbols (first 10): {self.failed_symbols[:10]}")
        
        logger.info("=" * 60)
        
        # Save blacklist
        self.blacklist.save_if_dirty()
        
        return results
    
    def _fetch_and_save_with_retry(self, symbol: str) -> int:
        """Fetch and save with proper error handling"""
        try:
            # Fetch data
            data = self.fetch_stock_history(symbol)
            
            if data is not None and not data.empty:
                # Save with retry logic for locks
                saved = self.save_historical_data(symbol, data, update_mode="replace")
                return saved
            
            return 0
            
        except Exception as e:
            logger.debug(f"Error processing {symbol}: {e}")
            with self.lock:
                self.failed_symbols.append(symbol)
            return 0
    
    def update_recent_history(self, symbols: List[str], days: int = 5, shutdown_event=None) -> Dict[str, int]:
        """
        Update only recent history (for daily updates)
        """
        self.shutdown_event = shutdown_event
        
        # Filter out blacklisted symbols
        valid_symbols = self.blacklist.filter_valid_symbols(symbols)
        
        logger.info(f"Updating recent {days} days for {len(valid_symbols)} symbols")
        
        results = {}
        
        # Use fewer workers for updates
        safe_workers = min(MAX_WORKERS // 2, 4)
        
        with ThreadPoolExecutor(max_workers=safe_workers) as executor:
            future_to_symbol = {}
            
            for symbol in valid_symbols:
                if shutdown_event and shutdown_event.is_set():
                    break
                    
                time.sleep(0.05)  # Small delay between submissions
                future = executor.submit(self._update_recent_safe, symbol, days)
                future_to_symbol[future] = symbol
            
            for future in as_completed(future_to_symbol):
                if shutdown_event and shutdown_event.is_set():
                    break
                    
                symbol = future_to_symbol[future]
                try:
                    saved_count = future.result(timeout=30)
                    results[symbol] = saved_count
                except Exception as e:
                    logger.debug(f"Failed to update {symbol}: {e}")
                    results[symbol] = 0
        
        successful = sum(1 for v in results.values() if v > 0)
        logger.info(f"Updated recent history for {successful}/{len(valid_symbols)} symbols")
        
        self.blacklist.save_if_dirty()
        
        return results
    
    def _update_recent_safe(self, symbol: str, days: int) -> int:
        """Update recent history with safety checks and rate limiting"""
        with self.rate_limiter:
            try:
                ticker = yf.Ticker(symbol)
                
                # Get recent data
                data = ticker.history(
                    period=f"{days}d",
                    auto_adjust=False,
                    actions=True,
                    repair=True
                )
                
                if data is not None and not data.empty:
                    data = data.dropna(subset=['Close'])
                    if not data.empty:
                        data = data.reset_index()
                        data['Symbol'] = symbol
                        
                        # Use append mode to add new data
                        saved = self.save_historical_data(symbol, data, update_mode="append")
                        return saved
                
                return 0
                
            except Exception as e:
                logger.debug(f"Error updating {symbol}: {e}")
                return 0

# Database optimization for parallel operations
def optimize_database_for_parallel():
    """Optimize database settings for parallel operations"""
    db = get_database()
    with db.get_connection() as conn:
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=20000")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA busy_timeout=15000")
        conn.execute("PRAGMA mmap_size=536870912")
        logger.info("Database optimized for historical data operations")

def get_all_symbols_from_db() -> List[str]:
    """Get all unique symbols from the database"""
    db = get_database()
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        # Fixed query - check if column exists first
        cursor.execute("PRAGMA table_info(stocks)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'is_blacklisted' in columns:
            cursor.execute("""
                SELECT DISTINCT symbol FROM stocks 
                WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM stocks)
                  AND (is_blacklisted = 0 OR is_blacklisted IS NULL)
                ORDER BY symbol
            """)
        else:
            # Fallback without is_blacklisted column
            cursor.execute("""
                SELECT DISTINCT symbol FROM stocks 
                WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM stocks)
                ORDER BY symbol
            """)
        
        symbols = [row[0] for row in cursor.fetchall()]
    
    return symbols

def update_all_historical_data(period: str = "10y", batch_only: Optional[List[str]] = None,
                              shutdown_event=None, max_years: int = MAX_YEARS_HISTORY):
    """
    Main function to update historical data
    """
    logger.info("=" * 60)
    logger.info(f"{max_years}-YEAR HISTORICAL DATA UPDATE")
    logger.info(f"Version: {__version__}")
    logger.info("=" * 60)
    
    # Optimize database first
    optimize_database_for_parallel()
    
    logger.info(f"Fetching {max_years}-year historical data for all stocks")
    
    # Get symbols to process
    if batch_only:
        symbols = batch_only
        logger.info(f"Processing batch of {len(symbols)} symbols")
    else:
        symbols = get_all_symbols_from_db()
        
        if not symbols:
            logger.warning("No symbols found in database. Run the main pipeline first.")
            return {}
        
        logger.info(f"Found {len(symbols)} symbols in database")
    
    # Initialize fetcher
    fetcher = HistoricalDataFetcher(max_years=max_years)
    
    # Process in smaller batches for large datasets
    batch_size = 100
    all_results = {}
    
    for i in range(0, len(symbols), batch_size):
        if shutdown_event and shutdown_event.is_set():
            logger.info("Shutdown requested")
            break
            
        batch = symbols[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(symbols) + batch_size - 1) // batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} symbols)")
        
        # Fetch historical data for this batch
        results = fetcher.fetch_all_historical_data(
            batch, 
            parallel=True, 
            batch_size=10,
            shutdown_event=shutdown_event
        )
        all_results.update(results)
        
        # Pause between batches
        if i + batch_size < len(symbols):
            logger.info("Pausing between batches...")
            time.sleep(3)
    
    # Final summary
    total_successful = sum(1 for v in all_results.values() if v > 0)
    total_records = sum(all_results.values())
    avg_records = total_records / total_successful if total_successful > 0 else 0
    
    logger.info("=" * 60)
    logger.info(f"{max_years}-YEAR HISTORICAL DATA UPDATE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total symbols processed: {len(all_results)}")
    logger.info(f"Successful downloads: {total_successful}")
    logger.info(f"Failed downloads: {len(all_results) - total_successful}")
    logger.info(f"Total historical records: {total_records:,}")
    logger.info(f"Average records per symbol: {avg_records:.0f}")
    logger.info("=" * 60)
    
    return all_results

def update_recent_data_only(days: int = 5, shutdown_event=None):
    """Update only recent data (for daily updates)"""
    logger.info(f"Updating recent {days} days of historical data")
    logger.info(f"Version: {__version__}")
    
    # Optimize database first
    optimize_database_for_parallel()
    
    symbols = get_all_symbols_from_db()
    if not symbols:
        logger.warning("No symbols found in database")
        return {}
    
    fetcher = HistoricalDataFetcher(max_years=MAX_YEARS_HISTORY)
    results = fetcher.update_recent_history(symbols, days=days, shutdown_event=shutdown_event)
    
    return results

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description=f"Historical Stock Data Fetcher v{__version__}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{__copyright__}
Author: {__author__} ({__email__})

This fetcher provides up to {MAX_YEARS_HISTORY} years of historical data.
Includes exponential backoff retry logic and database locking fixes.

Examples:
  python historical_fetcher.py --all              # Fetch full history for all
  python historical_fetcher.py --recent 5         # Update last 5 days only
  python historical_fetcher.py --symbols AAPL MSFT  # Specific symbols
  python historical_fetcher.py --years 5          # Custom year limit
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help=f'Fetch historical data for all symbols ({MAX_YEARS_HISTORY}-year default)')
    parser.add_argument('--recent', type=int, metavar='DAYS',
                       help='Update only recent N days')
    parser.add_argument('--symbols', nargs='+',
                       help='Process only these symbols')
    parser.add_argument('--years', type=int, default=MAX_YEARS_HISTORY,
                       help=f'Number of years of history to fetch (default: {MAX_YEARS_HISTORY})')
    parser.add_argument('--version', action='store_true',
                       help='Show version information')
    
    args = parser.parse_args()
    
    if args.version:
        print(f"Historical Stock Data Fetcher v{__version__}")
        print(f"{__copyright__}")
        print(f"Data coverage: up to {MAX_YEARS_HISTORY} years")
        sys.exit(0)
    
    # Setup signal handler
    shutdown_event = threading.Event()
    
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}. Initiating clean shutdown...")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if args.recent:
            update_recent_data_only(days=args.recent, shutdown_event=shutdown_event)
        elif args.symbols:
            update_all_historical_data(
                batch_only=args.symbols, 
                shutdown_event=shutdown_event,
                max_years=args.years
            )
        elif args.all:
            update_all_historical_data(
                shutdown_event=shutdown_event,
                max_years=args.years
            )
        else:
            # Default: update recent 5 days
            update_recent_data_only(days=5, shutdown_event=shutdown_event)
            
    except KeyboardInterrupt:
        logger.info("Historical data fetch interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Historical data fetch failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
