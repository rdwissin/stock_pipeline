#!/usr/bin/env python3
"""
Enhanced Stock Data Pipeline - Main Module with Fixed Signal Handling
Compatible with Python 3.7+ and proper executor shutdown
"""

__version__ = "1.5.0"
__copyright__ = "Copyright 2025, Richard D. Wissinger"
__author__ = "Richard D. Wissinger"
__email__ = "rick.wissinger@gmail.com"
__license__ = "MIT"
__status__ = "Production"

import sys
import os
import time
import psutil
import threading
import signal
import atexit
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json

# ===============================================================================
# THREAD-SAFE EXECUTOR MANAGEMENT
# ===============================================================================

class ExecutorManager:
    """Thread-safe executor pool manager"""
    
    def __init__(self):
        self.executors = []
        self.lock = threading.Lock()
        self.shutdown_event = threading.Event()
        
    def register(self, executor):
        """Register an executor for management"""
        with self.lock:
            self.executors.append(executor)
    
    def unregister(self, executor):
        """Unregister an executor"""
        with self.lock:
            if executor in self.executors:
                self.executors.remove(executor)
    
    def shutdown_all(self):
        """Shutdown all registered executors"""
        with self.lock:
            self.shutdown_event.set()
            for executor in self.executors[:]:  # Copy list to avoid modification during iteration
                try:
                    if sys.version_info >= (3, 9):
                        executor.shutdown(wait=False, cancel_futures=True)
                    else:
                        executor.shutdown(wait=False)
                except Exception as e:
                    print(f"Warning: Error shutting down executor: {e}")
            self.executors.clear()
    
    def is_shutdown_requested(self):
        """Check if shutdown has been requested"""
        return self.shutdown_event.is_set()

# Global executor manager instance
executor_manager = ExecutorManager()

def signal_handler(signum, frame):
    """Handle interrupt signals for clean shutdown - compatible with Python 3.7+"""
    print(f"\n⚠️ Received signal {signum}. Initiating clean shutdown...")
    
    # Use the thread-safe manager
    executor_manager.shutdown_all()
    
    # Clean up resources
    try:
        from fetcher import cleanup
        cleanup()
    except Exception as e:
        print(f"Warning: Error during fetcher cleanup: {e}")
    
    # Save blacklist if dirty
    try:
        from blacklist import get_blacklist
        blacklist = get_blacklist()
        blacklist.save_if_dirty()
        print("✅ Blacklist saved")
    except Exception as e:
        print(f"Warning: Error saving blacklist: {e}")
    
    print("✅ Clean shutdown completed")
    sys.exit(130)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Import after signal handling is set up
from config import (
    EXCHANGES, MAX_WORKERS, PRIMARY_PROVIDER, SECONDARY_PROVIDERS,
    ENABLE_PERFORMANCE_LOGGING, MAX_MEMORY_MB, COLLECT_HISTORICAL_PRICES
)
from fetcher import fetch_stocks, fetch_all_us_stocks, clear_cache, get_fetcher
from storage import (
    init_db, save_snapshot, get_latest_snapshot_date, log_pipeline_run,
    get_market_overview, cleanup_old_snapshots, backup_database
)
from logger import setup_logger, PerformanceTimer, log_system_info, performance_monitor
from alerts import send_alert
from blacklist import get_blacklist

logger = setup_logger(__name__)

# ===============================================================================
# DATA CLASSES
# ===============================================================================

@dataclass
class ExchangeResult:
    """Result of processing a single exchange"""
    exchange: str
    success: bool
    stocks_count: int
    new_stocks: int
    updated_stocks: int
    processing_time: float
    error_message: str = ""
    memory_used_mb: float = 0.0

@dataclass
class PipelineMetrics:
    """Comprehensive pipeline execution metrics"""
    start_time: datetime
    end_time: datetime = None
    total_exchanges: int = 0
    successful_exchanges: int = 0
    failed_exchanges: int = 0
    total_stocks: int = 0
    new_stocks: int = 0
    updated_stocks: int = 0
    peak_memory_mb: float = 0.0
    cpu_usage_avg: float = 0.0
    exchange_results: List[ExchangeResult] = None
    historical_data_fetched: bool = False
    historical_records: int = 0
    blacklisted_symbols: int = 0
    technical_analysis_completed: bool = False

    def __post_init__(self):
        if self.exchange_results is None:
            self.exchange_results = []
    
    @property
    def duration(self) -> timedelta:
        end = self.end_time or datetime.now()
        return end - self.start_time
    
    @property
    def success_rate(self) -> float:
        if self.total_exchanges == 0:
            return 0.0
        return self.successful_exchanges / self.total_exchanges
    
    @property
    def processing_rate(self) -> float:
        """Stocks per second"""
        duration_seconds = self.duration.total_seconds()
        if duration_seconds == 0:
            return 0.0
        return self.total_stocks / duration_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration.total_seconds(),
            'total_exchanges': self.total_exchanges,
            'successful_exchanges': self.successful_exchanges,
            'failed_exchanges': self.failed_exchanges,
            'total_stocks': self.total_stocks,
            'new_stocks': self.new_stocks,
            'updated_stocks': self.updated_stocks,
            'success_rate': self.success_rate,
            'processing_rate': self.processing_rate,
            'peak_memory_mb': self.peak_memory_mb,
            'cpu_usage_avg': self.cpu_usage_avg,
            'historical_data_fetched': self.historical_data_fetched,
            'historical_records': self.historical_records,
            'blacklisted_symbols': self.blacklisted_symbols,
            'exchange_results': [
                {
                    'exchange': r.exchange,
                    'success': r.success,
                    'stocks_count': r.stocks_count,
                    'new_stocks': r.new_stocks,
                    'updated_stocks': r.updated_stocks,
                    'processing_time': r.processing_time,
                    'error_message': r.error_message,
                    'memory_used_mb': r.memory_used_mb
                }
                for r in self.exchange_results
            ]
        }

# ===============================================================================
# SYSTEM MONITORING
# ===============================================================================

class SystemMonitor:
    """Monitor system resources during pipeline execution"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.peak_memory = 0.0
        self.cpu_samples = []
        self.monitoring = False
        self.monitor_thread = None
        self.stop_event = threading.Event()
    
    def start_monitoring(self):
        """Start system monitoring in background thread"""
        self.monitoring = True
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.trace("Started system monitoring")
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return metrics"""
        self.monitoring = False
        self.stop_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0
        
        metrics = {
            'peak_memory_mb': self.peak_memory,
            'avg_cpu_percent': avg_cpu,
            'cpu_samples_count': len(self.cpu_samples)
        }
        
        logger.trace(f"System monitoring stopped: {metrics}")
        return metrics
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring and not self.stop_event.is_set() and not executor_manager.is_shutdown_requested():
            try:
                # Memory usage
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                self.peak_memory = max(self.peak_memory, memory_mb)
                
                # CPU usage
                cpu_percent = self.process.cpu_percent()
                self.cpu_samples.append(cpu_percent)
                
                # Memory warning
                if memory_mb > MAX_MEMORY_MB * 0.9:
                    logger.warning(f"High memory usage: {memory_mb:.1f}MB (limit: {MAX_MEMORY_MB}MB)")
                
                self.stop_event.wait(timeout=5)  # Sleep with interrupt capability
                
            except Exception as e:
                logger.debug(f"Error in monitoring loop: {e}")
                break

# ===============================================================================
# ENHANCED PIPELINE
# ===============================================================================

class EnhancedPipeline:
    """Enhanced stock data pipeline with complete US stock collection and historical data"""
    
    def __init__(self):
        self.metrics = None
        self.monitor = SystemMonitor()
        self.blacklist = get_blacklist()
        self.executor = None
        
    def __del__(self):
        """Cleanup on deletion"""
        self._shutdown_executor()
    
    def _create_executor(self, parallel: bool = True):
        """Create and register executor"""
        if parallel and not self.executor:
            self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
            executor_manager.register(self.executor)
    
    def _shutdown_executor(self):
        """Shutdown executor with version compatibility"""
        if self.executor:
            try:
                executor_manager.unregister(self.executor)
                if sys.version_info >= (3, 9):
                    self.executor.shutdown(wait=True, cancel_futures=False)
                else:
                    self.executor.shutdown(wait=True)
            except Exception as e:
                logger.warning(f"Error during executor shutdown: {e}")
            finally:
                self.executor = None
    
    @performance_monitor("should_run_today")
    def should_run_today(self, force: bool = False) -> Tuple[bool, str]:
        """
        Determine if pipeline should run today
        Returns: (should_run, reason)
        """
        if force:
            return True, "Forced execution"
        
        today = datetime.now()
        
        # Skip weekends (Saturday=5, Sunday=6)
        if today.weekday() >= 5:
            return False, "Weekend - markets are closed"
        
        # Check if already ran today
        today_str = today.strftime("%Y-%m-%d")
        latest_snapshot = get_latest_snapshot_date()
        
        if latest_snapshot == today_str:
            return False, f"Data already collected for {today_str}"
        
        # Check for holidays (simplified - could be enhanced with holiday API)
        return True, "Weekday - markets are open"
    
    @performance_monitor("fetch_all_us_stocks")
    def fetch_complete_us_market(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch ALL US stocks from all exchanges"""
        if executor_manager.is_shutdown_requested():
            logger.warning("Shutdown requested, skipping market fetch")
            return {}
            
        logger.info("Fetching complete US stock market data")
        
        # Get blacklist statistics
        blacklist_stats = self.blacklist.get_statistics()
        self.metrics.blacklisted_symbols = blacklist_stats['total_blacklisted']
        logger.info(f"Blacklist contains {self.metrics.blacklisted_symbols} invalid symbols")
        
        try:
            # Use the new comprehensive fetch function
            with PerformanceTimer(logger, "fetch_all_us_stocks"):
                all_stocks = fetch_all_us_stocks()
            
            # Log summary
            total = sum(len(stocks) for stocks in all_stocks.values())
            logger.info(f"Retrieved {total} total stocks across all exchanges")
            for exchange, stocks in all_stocks.items():
                logger.info(f"  {exchange}: {len(stocks)} stocks")
            
            # Update blacklist statistics
            blacklist_stats = self.blacklist.get_statistics()
            new_blacklisted = blacklist_stats['total_blacklisted'] - self.metrics.blacklisted_symbols
            if new_blacklisted > 0:
                logger.info(f"Added {new_blacklisted} new symbols to blacklist")
                self.metrics.blacklisted_symbols = blacklist_stats['total_blacklisted']
            
            return all_stocks
            
        except Exception as e:
            logger.error(f"Failed to fetch complete US market: {e}")
            return {}
    
    @performance_monitor("fetch_historical_data")
    def fetch_historical_data(self, mode: str = "recent", force: bool = False):
        """
        Fetch historical data for all stocks with interrupt handling
        
        Args:
            mode: 'full' for complete history, 'recent' for last few days
            force: Force full historical fetch even if data exists
        """
        if executor_manager.is_shutdown_requested():
            logger.warning("Shutdown requested, skipping historical data fetch")
            return
            
        logger.info(f"Starting historical data fetch (mode: {mode})")
        
        try:
            from historical_fetcher import update_all_historical_data, update_recent_data_only
            
            with PerformanceTimer(logger, "fetch_historical_data"):
                if executor_manager.is_shutdown_requested():
                    return
                    
                if mode == "full" or force:
                    logger.info("Fetching MAXIMUM historical data for all stocks")
                    results = update_all_historical_data(
                        period="max", 
                        shutdown_event=executor_manager.shutdown_event
                    )
                    self.metrics.historical_data_fetched = True
                    self.metrics.historical_records = sum(results.values()) if results else 0
                else:
                    logger.info("Updating recent historical data (last 5 days)")
                    results = update_recent_data_only(
                        days=5, 
                        shutdown_event=executor_manager.shutdown_event
                    )
                    self.metrics.historical_data_fetched = True
                    self.metrics.historical_records = sum(results.values()) if results else 0
                
                if results:
                    successful = sum(1 for v in results.values() if v > 0)
                    logger.info(f"Historical data fetched for {successful} symbols")
                    
        except ImportError:
            logger.warning("historical_fetcher module not found, skipping historical data")
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
    
    @performance_monitor("run_pipeline")
    def run_pipeline(self, force: bool = False, parallel: bool = True, 
                    fetch_historical: bool = None, historical_mode: str = "recent") -> bool:
        """
        Run the complete enhanced stock data pipeline
        
        Args:
            force: Force run even if data exists
            parallel: Use parallel processing
            fetch_historical: Fetch historical data (None=use config, True/False=override)
            historical_mode: 'full' or 'recent' for historical data
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 80)
        logger.info(f"🚀 STARTING ENHANCED STOCK DATA PIPELINE v{__version__}")
        logger.info("=" * 80)
        
        # Log system information
        log_system_info()
        
        # Initialize metrics and monitoring
        self.metrics = PipelineMetrics(
            start_time=datetime.now(),
            total_exchanges=3  # NYSE, NASDAQ, AMEX
        )
        self.monitor.start_monitoring()
        
        # Create executor if parallel processing
        self._create_executor(parallel)
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        try:
            # Check for shutdown
            if executor_manager.is_shutdown_requested():
                logger.warning("Shutdown requested at startup")
                return False
                
            # Pre-flight checks
            should_run, reason = self.should_run_today(force)
            if not should_run:
                logger.info(f"⭕ Pipeline skipped: {reason}")
                self._log_final_results("SKIPPED", reason)
                return True
            
            logger.info(f"✅ Pre-flight check passed: {reason}")
            
            # Initialize database
            with PerformanceTimer(logger, "database_initialization"):
                init_db()
            
            # Clean up old data (keep 90 days for historical analysis)
            if not executor_manager.is_shutdown_requested():
                with PerformanceTimer(logger, "cleanup_old_data"):
                    deleted_stocks, deleted_prices = cleanup_old_snapshots(90)
                    if deleted_stocks > 0:
                        logger.info(f"🧹 Cleaned up {deleted_stocks} old stock records, {deleted_prices} price records")
            
            # Clean up old blacklist entries
            self.blacklist.cleanup_old_entries(180)
            
            # Backup database
            if not executor_manager.is_shutdown_requested():
                with PerformanceTimer(logger, "database_backup"):
                    if backup_database():
                        logger.info("💾 Database backup completed")
            
            # Fetch ALL US stocks
            if not executor_manager.is_shutdown_requested():
                logger.info("📊 Fetching complete US stock market...")
                all_stocks_data = self.fetch_complete_us_market()
                
                if not all_stocks_data:
                    logger.error("Failed to fetch stock data")
                    self._log_final_results("FAILED", "No stock data retrieved")
                    return False
            else:
                logger.warning("Shutdown requested, skipping data fetch")
                return False
            
            # Process and save all stocks
            success = self._process_all_stocks(all_stocks_data, parallel)
            
            # Fetch historical data if enabled
            should_fetch_historical = fetch_historical if fetch_historical is not None else COLLECT_HISTORICAL_PRICES
            
            if should_fetch_historical and success and not executor_manager.is_shutdown_requested():
                logger.info("📈 Starting historical data collection...")
                
                # Determine if this is the first run (no historical data)
                from storage import get_database
                db = get_database()
                with db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM historical_prices")
                    historical_count = cursor.fetchone()[0]
                
                # If no historical data exists, do full fetch
                if historical_count == 0:
                    logger.info("No historical data found - fetching MAXIMUM history for all stocks")
                    self.fetch_historical_data(mode="full", force=True)
                else:
                    # Otherwise, use specified mode
                    self.fetch_historical_data(mode=historical_mode, force=force)

            # Run technical analysis if data collection was successful
            if success and not executor_manager.is_shutdown_requested():
                if os.getenv('RUN_TECHNICAL_ANALYSIS', 'true').lower() == 'true':
                    logger.info("📊 Starting technical analysis...")
                    
                    try:
                        from technical_analysis import run_daily_technical_analysis
                        
                        with PerformanceTimer(logger, "technical_analysis"):
                            analysis_success = run_daily_technical_analysis()
                            
                            if analysis_success:
                                logger.info("✅ Technical analysis completed successfully")
                                self.metrics.technical_analysis_completed = True
                            else:
                                logger.warning("⚠️ Technical analysis completed with issues")
                                
                    except ImportError:
                        logger.warning("Technical analysis module not available")
                    except Exception as e:
                        logger.error(f"Technical analysis failed: {e}")

            # Finalize metrics
            self.metrics.end_time = datetime.now()
            system_metrics = self.monitor.stop_monitoring()
            self.metrics.peak_memory_mb = system_metrics['peak_memory_mb']
            self.metrics.cpu_usage_avg = system_metrics['avg_cpu_percent']
            
            # Save blacklist if dirty
            self.blacklist.save_if_dirty()
            
            # Log final results
            status = "SUCCESS" if success else "PARTIAL" if self.metrics.successful_exchanges > 0 else "FAILED"
            self._log_final_results(status)
            
            # Send alerts if needed
            self._send_alerts_if_needed(success)
            
            return success
            
        except Exception as e:
            error_msg = f"Critical pipeline error: {e}"
            logger.critical(error_msg)
            self._log_final_results("FAILED", error_msg)
            send_alert("Pipeline Critical Failure", error_msg)
            return False
        finally:
            # Ensure monitoring is stopped
            if self.monitor.monitoring:
                self.monitor.stop_monitoring()
            
            # Save blacklist
            self.blacklist.save_if_dirty()
            
            # Shutdown executor properly
            self._shutdown_executor()
            
            # Clear cache to free memory
            clear_cache()
    
    @performance_monitor("process_all_stocks")
    def _process_all_stocks(self, all_stocks_data: Dict[str, List[Dict[str, Any]]], 
                          parallel: bool = True) -> bool:
        """Process and save all fetched stock data"""
        if executor_manager.is_shutdown_requested():
            logger.warning("Shutdown requested, skipping stock processing")
            return False
            
        logger.info(f"Processing {sum(len(stocks) for stocks in all_stocks_data.values())} total stocks")
        
        success = True
        
        for exchange, stocks_data in all_stocks_data.items():
            if executor_manager.is_shutdown_requested():
                logger.warning("Shutdown requested during processing")
                break
                
            if not stocks_data:
                logger.warning(f"No data for {exchange}")
                continue
                
            try:
                # Save to database
                with PerformanceTimer(logger, f"save_{exchange}_data"):
                    saved_count, new_count, updated_count = save_snapshot(exchange, stocks_data)
                
                # Record result
                result = ExchangeResult(
                    exchange=exchange,
                    success=True,
                    stocks_count=saved_count,
                    new_stocks=new_count,
                    updated_stocks=updated_count,
                    processing_time=0,  # Already tracked by PerformanceTimer
                    memory_used_mb=psutil.Process().memory_info().rss / 1024 / 1024
                )
                
                self.metrics.exchange_results.append(result)
                self.metrics.successful_exchanges += 1
                self.metrics.total_stocks += saved_count
                self.metrics.new_stocks += new_count
                self.metrics.updated_stocks += updated_count
                
                logger.info(f"✅ {exchange}: {saved_count} stocks "
                          f"({new_count} new, {updated_count} updated)")
                
            except Exception as e:
                logger.error(f"Failed to process {exchange}: {e}")
                self.metrics.failed_exchanges += 1
                result = ExchangeResult(
                    exchange=exchange,
                    success=False,
                    stocks_count=0,
                    new_stocks=0,
                    updated_stocks=0,
                    processing_time=0,
                    error_message=str(e)
                )
                self.metrics.exchange_results.append(result)
                success = False
        
        return success
    
    def _log_final_results(self, status: str, error_msg: str = ""):
        """Log comprehensive final results"""
        logger.info("=" * 80)
        logger.info("📊 PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 80)
        
        if self.metrics:
            logger.info(f"⏱️  Duration: {self.metrics.duration}")
            logger.info(f"📈 Success rate: {self.metrics.success_rate:.1%}")
            logger.info(f"🎯 Total stocks collected: {self.metrics.total_stocks:,}")
            logger.info(f"🆕 New stocks: {self.metrics.new_stocks:,}")
            logger.info(f"🔄 Updated stocks: {self.metrics.updated_stocks:,}")
            logger.info(f"🚫 Blacklisted symbols: {self.metrics.blacklisted_symbols:,}")
            logger.info(f"⚡ Processing rate: {self.metrics.processing_rate:.1f} stocks/second")
            logger.info(f"💾 Peak memory usage: {self.metrics.peak_memory_mb:.1f} MB")
            logger.info(f"🖥️  Average CPU usage: {self.metrics.cpu_usage_avg:.1f}%")
            
            if self.metrics.historical_data_fetched:
                logger.info(f"📈 Historical records saved: {self.metrics.historical_records:,}")
            
            # Exchange breakdown
            if self.metrics.exchange_results:
                logger.info("\n📋 EXCHANGE BREAKDOWN:")
                for result in self.metrics.exchange_results:
                    status_icon = "✅" if result.success else "❌"
                    logger.info(f"  {status_icon} {result.exchange:8}: "
                              f"{result.stocks_count:5,} stocks")
                    if not result.success and result.error_message:
                        logger.info(f"     Error: {result.error_message}")
            
            # Market overview
            try:
                overview = get_market_overview()
                if overview.get('overall'):
                    total_market_cap = overview['overall'][2] or 0
                    logger.info(f"\n💰 MARKET OVERVIEW:")
                    logger.info(f"  Total Market Cap: ${total_market_cap/1e12:.2f}T")
                    logger.info(f"  Average P/E Ratio: {overview['overall'][3]:.1f}" if overview['overall'][3] else "  Average P/E Ratio: N/A")
                    logger.info(f"  Average Dividend Yield: {overview['overall'][4]:.2f}%" if overview['overall'][4] else "  Average Dividend Yield: N/A")
            except Exception as e:
                logger.warning(f"Could not generate market overview: {e}")
        
        logger.info("=" * 80)
        
        # Log to database
        if self.metrics:
            exchanges_processed = [r.exchange for r in self.metrics.exchange_results if r.success]
            log_pipeline_run(
                run_date=datetime.now().strftime("%Y-%m-%d"),
                status=status,
                exchanges=exchanges_processed,
                total_stocks=self.metrics.total_stocks,
                new_stocks=self.metrics.new_stocks,
                updated_stocks=self.metrics.updated_stocks,
                processing_time=self.metrics.duration.total_seconds(),
                peak_memory=self.metrics.peak_memory_mb,
                errors=error_msg,
                performance_metrics=self.metrics.to_dict()
            )
    
    def _send_alerts_if_needed(self, success: bool):
        """Send alerts based on pipeline results"""
        from config import SEND_SUCCESS_ALERTS  # Import the config
        
        if self.metrics:
            if success and SEND_SUCCESS_ALERTS:  # Check config before sending success alerts
                # Send success notification
                if self.metrics.successful_exchanges == len(self.metrics.exchange_results):
                    # Complete success - all exchanges processed successfully
                    alert_message = f"""
                    ✅ Pipeline completed successfully!
                    
                    ⏱️  Duration: {self.metrics.duration}
                    📊 Total stocks collected: {self.metrics.total_stocks:,}
                    🆕 New stocks added: {self.metrics.new_stocks:,}
                    🔄 Stocks updated: {self.metrics.updated_stocks:,}
                    📈 Historical records saved: {self.metrics.historical_records:,}
                    
                    💾 Memory usage: {self.metrics.peak_memory_mb:.1f} MB
                    🏆 Success rate: {self.metrics.success_rate:.1%}
                    
                    Exchanges processed: {', '.join([r.exchange for r in self.metrics.exchange_results if r.success])}
                    
                    The data download and processing completed successfully.
                    """
                    send_alert("Pipeline Success", alert_message)
                    logger.info("Success notification sent")
                
                elif self.metrics.successful_exchanges > 0:
                    # Partial success - some exchanges succeeded
                    successful_exchanges = [r.exchange for r in self.metrics.exchange_results if r.success]
                    failed_exchanges = [r.exchange for r in self.metrics.exchange_results if not r.success]
                    
                    alert_message = f"""
                    ⚠️  Pipeline completed with partial success
                    
                    ✅ Success rate: {self.metrics.success_rate:.1%}
                    ⏱️  Duration: {self.metrics.duration}
                    📊 Total stocks collected: {self.metrics.total_stocks:,}
                    🆕 New stocks added: {self.metrics.new_stocks:,}
                    🔄 Stocks updated: {self.metrics.updated_stocks:,}
                    
                    ✅ Successful exchanges: {', '.join(successful_exchanges)}
                    ❌ Failed exchanges: {', '.join(failed_exchanges)}
                    
                    The pipeline completed but some exchanges had issues.
                    """
                    send_alert("Pipeline Partial Success", alert_message)
                    logger.info("Partial success notification sent")
            
            elif not success:  # Always send failure alerts
                # Handle failures (existing code remains the same)
                if self.metrics.successful_exchanges == 0:
                    # Complete failure
                    alert_message = f"""
                    🚨 Complete pipeline failure!
                    
                    ⏱️  Duration: {self.metrics.duration}
                    ❌ Failed exchanges: {', '.join([r.exchange for r in self.metrics.exchange_results if not r.success])}
                    📊 Total stocks collected: {self.metrics.total_stocks}
                    
                    Errors:
                    {chr(10).join([f"• {r.exchange}: {r.error_message}" for r in self.metrics.exchange_results if not r.success and r.error_message])}
                    """
                    send_alert("Pipeline Complete Failure", alert_message)
                    
                else:
                    # Partial failure
                    failed_exchanges = [r.exchange for r in self.metrics.exchange_results if not r.success]
                    alert_message = f"""
                    ⚠️  Partial pipeline failure
                    
                    ✅ Success rate: {self.metrics.success_rate:.1%}
                    📊 Total stocks collected: {self.metrics.total_stocks}
                    ❌ Failed exchanges: {', '.join(failed_exchanges)}
                    
                    The pipeline completed but some exchanges failed.
                    """
                    send_alert("Pipeline Partial Failure", alert_message)

def cleanup_on_exit():
    """Cleanup function for atexit"""
    try:
        logger.info("Performing final cleanup...")
        executor_manager.shutdown_all()
        clear_cache()
        blacklist = get_blacklist()
        blacklist.save_if_dirty()
        
        # Close all database connections
        from storage import get_database
        db = get_database()
        db.shutdown()
    except:
        pass

# Register cleanup on exit
atexit.register(cleanup_on_exit)

def print_version():
    """Print version information"""
    print(f"Enhanced Stock Data Pipeline v{__version__}")
    print(f"{__copyright__}")
    print(f"Author: {__author__} ({__email__})")
    print(f"License: {__license__}")
    print(f"Status: {__status__}")

def main():
    """Main entry point with enhanced argument handling"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description=f"Enhanced Stock Data Pipeline v{__version__} - Complete US Market Collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{__copyright__}
Author: {__author__} ({__email__})

Examples:
  python main.py                    # Normal run with recent historical data
  python main.py --force            # Force run even if data exists
  python main.py --full-history     # Fetch complete historical data
  python main.py --no-history       # Skip historical data fetch
  python main.py --debug            # Enable debug logging
  python main.py --sequential       # Disable parallel processing
  python main.py --version          # Show version information
        """
    )
    
    parser.add_argument("--version", action="store_true",
                       help="Show version information")
    parser.add_argument("--force", action="store_true", 
                       help="Force run even if data exists for today")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--trace", action="store_true",
                       help="Enable trace-level logging (very verbose)")
    parser.add_argument("--sequential", action="store_true",
                       help="Disable parallel processing")
    parser.add_argument("--memory-limit", type=int, metavar="MB",
                       help=f"Memory limit in MB (default: {MAX_MEMORY_MB})")
    parser.add_argument("--full-history", action="store_true",
                       help="Fetch complete historical data (all available)")
    parser.add_argument("--no-history", action="store_true",
                       help="Skip historical data fetching")
    parser.add_argument("--recent-history", type=int, metavar="DAYS",
                       help="Fetch only N days of recent history")
    
    args = parser.parse_args()
    
    # Show version and exit
    if args.version:
        print_version()
        sys.exit(0)
    
    # Configure logging level
    if args.trace:
        import logging
        logging.getLogger().setLevel(5)  # TRACE level
        logger.info("🔍 Trace logging enabled (very verbose)")
    elif args.debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🐛 Debug logging enabled")
    
    # Update memory limit if specified
    if args.memory_limit:
        import config
        config.MAX_MEMORY_MB = args.memory_limit
        logger.info(f"💾 Memory limit set to {args.memory_limit}MB")
    
    # Determine historical data mode
    fetch_historical = None
    historical_mode = "recent"
    
    if args.no_history:
        fetch_historical = False
    elif args.full_history:
        fetch_historical = True
        historical_mode = "full"
    elif args.recent_history:
        fetch_historical = True
        historical_mode = "recent"
    
    try:
        pipeline = EnhancedPipeline()
        success = pipeline.run_pipeline(
            force=args.force, 
            parallel=not args.sequential,
            fetch_historical=fetch_historical,
            historical_mode=historical_mode
        )
        
        exit_code = 0 if success else 1
        logger.info(f"🏁 Pipeline completed with exit code: {exit_code}")
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.info("⛔️ Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"💥 Unhandled error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()