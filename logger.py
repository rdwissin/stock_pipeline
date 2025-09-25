# logger.py - Enhanced logging with detailed levels and performance tracking

__version__ = "1.5.0"
__copyright__ = "Copyright 2025, Richard D. Wissinger"
__author__ = "Richard D. Wissinger"
__email__ = "rick.wissinger@gmail.com"
__license__ = "MIT"  # or your chosen license
__status__ = "Production"

import logging
import logging.handlers
import sys
import time
import psutil
import threading
from pathlib import Path
from datetime import datetime
from functools import wraps
from typing import Dict, Any, Optional
from config import (
    LOGS_DIR, LOG_LEVELS, LOG_LEVEL, ENABLE_PERFORMANCE_LOGGING, 
    ENABLE_MEMORY_TRACKING, MAX_MEMORY_MB
)

# Add custom TRACE level
logging.addLevelName(5, "TRACE")

class EnhancedLogger:
    """Enhanced logger with performance tracking and memory monitoring"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        self._setup_logger()
        self.start_time = time.time()
        self.memory_tracker = MemoryTracker() if ENABLE_MEMORY_TRACKING else None
        
    def _setup_logger(self):
        """Setup logger with enhanced formatting and handlers"""
        if self.logger.handlers:
            return
            
        # Set level from config
        level = LOG_LEVELS.get(LOG_LEVEL, logging.INFO)
        self.logger.setLevel(level)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - '
            '%(filename)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        performance_formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d - PERF - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Main application log file (rotated)
        main_log = LOGS_DIR / f"pipeline_{datetime.now().strftime('%Y%m')}.log"
        main_handler = logging.handlers.RotatingFileHandler(
            main_log, 
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10,
            encoding='utf-8'
        )
        main_handler.setFormatter(detailed_formatter)
        main_handler.setLevel(logging.DEBUG)
        
        # Performance log file (separate)
        if ENABLE_PERFORMANCE_LOGGING:
            perf_log = LOGS_DIR / f"performance_{datetime.now().strftime('%Y%m')}.log"
            perf_handler = logging.handlers.RotatingFileHandler(
                perf_log,
                maxBytes=20*1024*1024,  # 20MB
                backupCount=5,
                encoding='utf-8'
            )
            perf_handler.setFormatter(performance_formatter)
            perf_handler.setLevel(5)  # TRACE level
            perf_handler.addFilter(PerformanceFilter())
            
        # Error log file (errors only)
        error_log = LOGS_DIR / f"errors_{datetime.now().strftime('%Y%m')}.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setFormatter(detailed_formatter)
        error_handler.setLevel(logging.ERROR)
        
        # Console handler (adaptive level)
        console_handler = logging.StreamHandler(sys.stdout)
        console_level = logging.INFO if LOG_LEVEL in ['TRACE', 'DEBUG'] else logging.WARNING
        console_handler.setLevel(console_level)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers
        self.logger.addHandler(main_handler)
        if ENABLE_PERFORMANCE_LOGGING:
            self.logger.addHandler(perf_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(console_handler)
    
    def trace(self, message: str, **kwargs):
        """Most detailed logging - API calls, timing, data flow"""
        self.logger.log(5, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Detailed logging - function entry/exit, data processing"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """General information - progress updates, milestones"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Issues that don't stop execution - retries, data anomalies"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Errors that affect specific operations"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Fatal errors that stop pipeline execution"""
        self.logger.critical(message, **kwargs)
    
    def performance(self, operation: str, duration: float, details: Dict[str, Any] = None):
        """Log performance metrics"""
        if not ENABLE_PERFORMANCE_LOGGING:
            return
            
        details = details or {}
        memory_info = self.memory_tracker.get_memory_info() if self.memory_tracker else {}
        
        perf_data = {
            'operation': operation,
            'duration_ms': round(duration * 1000, 2),
            'duration_s': round(duration, 3),
            **details,
            **memory_info
        }
        
        # Format performance message
        parts = [f"{operation}: {duration:.3f}s"]
        if details:
            parts.extend([f"{k}={v}" for k, v in details.items()])
        if memory_info:
            parts.extend([f"{k}={v}" for k, v in memory_info.items()])
            
        self.trace(f"PERF: {' | '.join(parts)}")

class PerformanceFilter(logging.Filter):
    """Filter to only show performance-related logs"""
    def filter(self, record):
        return 'PERF:' in record.getMessage()

class MemoryTracker:
    """Track memory usage and alert on high usage"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.peak_memory = 0
        self.warning_threshold = MAX_MEMORY_MB * 0.8  # 80% of max
        self.critical_threshold = MAX_MEMORY_MB * 0.95  # 95% of max
        
    def get_memory_info(self) -> Dict[str, str]:
        """Get current memory usage information"""
        try:
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # Track peak usage
            if memory_mb > self.peak_memory:
                self.peak_memory = memory_mb
            
            # Check thresholds
            if memory_mb > self.critical_threshold:
                logger = logging.getLogger('memory_tracker')
                logger.critical(f"Critical memory usage: {memory_mb:.1f}MB (>{self.critical_threshold:.1f}MB)")
            elif memory_mb > self.warning_threshold:
                logger = logging.getLogger('memory_tracker')
                logger.warning(f"High memory usage: {memory_mb:.1f}MB (>{self.warning_threshold:.1f}MB)")
            
            return {
                'memory_mb': f"{memory_mb:.1f}MB",
                'peak_mb': f"{self.peak_memory:.1f}MB",
                'memory_pct': f"{(memory_mb/MAX_MEMORY_MB)*100:.1f}%"
            }
        except Exception:
            return {'memory_mb': 'unknown'}

class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, logger: EnhancedLogger, operation: str, details: Dict[str, Any] = None):
        self.logger = logger
        self.operation = operation
        self.details = details or {}
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.logger.trace(f"Starting {self.operation}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type:
            self.details['error'] = str(exc_val)
            self.logger.error(f"Failed {self.operation} after {duration:.3f}s: {exc_val}")
        else:
            self.logger.performance(self.operation, duration, self.details)

def performance_monitor(operation_name: str = None):
    """Decorator for monitoring function performance"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = setup_logger(func.__module__)
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with PerformanceTimer(logger, op_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def log_function_entry(logger: EnhancedLogger, func_name: str, *args, **kwargs):
    """Log function entry with parameters"""
    if logger.logger.isEnabledFor(5):  # TRACE level
        args_str = ', '.join([str(arg)[:50] for arg in args[:3]])  # First 3 args, truncated
        kwargs_str = ', '.join([f"{k}={str(v)[:50]}" for k, v in list(kwargs.items())[:3]])
        params = f"({args_str}" + (f", {kwargs_str}" if kwargs_str else "") + ")"
        logger.trace(f"→ {func_name}{params}")

def log_function_exit(logger: EnhancedLogger, func_name: str, result=None, duration=None):
    """Log function exit with result and timing"""
    if logger.logger.isEnabledFor(5):  # TRACE level
        result_str = str(result)[:100] if result is not None else "None"
        duration_str = f" [{duration:.3f}s]" if duration else ""
        logger.trace(f"← {func_name} → {result_str}{duration_str}")

# Global logger cache
_logger_cache: Dict[str, EnhancedLogger] = {}
_lock = threading.Lock()

def setup_logger(name: str = "stock_pipeline") -> EnhancedLogger:
    """
    Get or create an enhanced logger instance
    Thread-safe singleton pattern for logger instances
    """
    with _lock:
        if name not in _logger_cache:
            _logger_cache[name] = EnhancedLogger(name)
        return _logger_cache[name]

def log_system_info():
    """Log system information at startup"""
    logger = setup_logger("system")
    
    # System info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    
    # Memory info
    memory = psutil.virtual_memory()
    logger.info(f"System memory: {memory.total/1024/1024/1024:.1f}GB total, "
               f"{memory.available/1024/1024/1024:.1f}GB available")
    
    # CPU info
    logger.info(f"CPU cores: {psutil.cpu_count()} logical, {psutil.cpu_count(logical=False)} physical")
    
    # Disk info
    disk = psutil.disk_usage('/')
    logger.info(f"Disk space: {disk.total/1024/1024/1024:.1f}GB total, "
               f"{disk.free/1024/1024/1024:.1f}GB free")
    
    # Configuration info
    logger.info(f"Log level: {LOG_LEVEL}")
    logger.info(f"Performance logging: {ENABLE_PERFORMANCE_LOGGING}")
    logger.info(f"Memory tracking: {ENABLE_MEMORY_TRACKING}")
    logger.info(f"Max memory limit: {MAX_MEMORY_MB}MB")

def setup_exception_logging():
    """Setup global exception handler"""
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
            
        logger = setup_logger("exception_handler")
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = handle_exception

# Setup exception handling on import
setup_exception_logging()