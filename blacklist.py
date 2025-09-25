#!/usr/bin/env python3
"""
Symbol Blacklist Manager
Maintains a persistent list of invalid symbols to avoid repeated failed lookups
"""

__version__ = "1.5.0"
__copyright__ = "Copyright 2025, Richard D. Wissinger"
__author__ = "Richard D. Wissinger"
__email__ = "rick.wissinger@gmail.com"
__license__ = "MIT"  # or your chosen license
__status__ = "Production"

import json
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Set, Dict, Any, Optional, List  # Added List import
import logging

logger = logging.getLogger(__name__)

class SymbolBlacklist:
    """Manages blacklisted symbols that are known to be invalid"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path(__file__).parent / "data"
        self.blacklist_file = self.data_dir / "symbol_blacklist.json"
        self.lock = threading.RLock()
        self._blacklist_data: Dict[str, Any] = {}
        self._blacklist_set: Set[str] = set()
        self._load_blacklist()
        self._dirty = False  # Track if changes need saving
        
    def _load_blacklist(self):
        """Load blacklist from file"""
        try:
            if self.blacklist_file.exists():
                with open(self.blacklist_file, 'r') as f:
                    self._blacklist_data = json.load(f)
                    # Extract symbols that are still blacklisted
                    self._blacklist_set = set(self._blacklist_data.get('symbols', {}).keys())
                    logger.info(f"Loaded {len(self._blacklist_set)} blacklisted symbols")
            else:
                self._initialize_blacklist()
        except Exception as e:
            logger.warning(f"Failed to load blacklist: {e}, initializing new one")
            self._initialize_blacklist()
    
    def _initialize_blacklist(self):
        """Initialize empty blacklist structure"""
        self._blacklist_data = {
            'version': '1.0',
            'last_updated': datetime.now().isoformat(),
            'total_symbols': 0,
            'symbols': {},
            'statistics': {
                'total_added': 0,
                'total_removed': 0,
                'last_cleanup': None
            }
        }
        
        # Add known invalid symbols
        known_invalid = {
            'IGZ': 'Known delisted symbol',
            'TEST001': 'Test symbol',
            'TEST002': 'Test symbol',
            'TEST003': 'Test symbol',
            'DUMMY': 'Invalid symbol',
            'FAKE': 'Invalid symbol',
            'NULL': 'Invalid symbol'
        }
        
        for symbol, reason in known_invalid.items():
            self._blacklist_data['symbols'][symbol] = {
                'added_date': datetime.now().isoformat(),
                'reason': reason,
                'error_code': 'KNOWN_INVALID',
                'retry_count': 0,
                'last_retry': None
            }
        
        self._blacklist_set = set(known_invalid.keys())
        self._save_blacklist()
    
    def _save_blacklist(self):
        """Save blacklist to file"""
        with self.lock:
            try:
                self._blacklist_data['last_updated'] = datetime.now().isoformat()
                self._blacklist_data['total_symbols'] = len(self._blacklist_set)
                
                # Ensure directory exists
                self.data_dir.mkdir(parents=True, exist_ok=True)
                
                # Write to temp file first for safety
                temp_file = self.blacklist_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(self._blacklist_data, f, indent=2)
                
                # Atomic rename
                temp_file.replace(self.blacklist_file)
                self._dirty = False
                
            except Exception as e:
                logger.error(f"Failed to save blacklist: {e}")
    
    def is_blacklisted(self, symbol: str) -> bool:
        """Check if a symbol is blacklisted"""
        with self.lock:
            return symbol.upper() in self._blacklist_set
    
    def add_symbol(self, symbol: str, reason: str = "Not found", error_code: str = "404"):
        """Add a symbol to the blacklist"""
        symbol = symbol.upper()
        
        with self.lock:
            if symbol not in self._blacklist_set:
                self._blacklist_set.add(symbol)
                self._blacklist_data['symbols'][symbol] = {
                    'added_date': datetime.now().isoformat(),
                    'reason': reason,
                    'error_code': error_code,
                    'retry_count': 0,
                    'last_retry': None
                }
                
                # Update statistics
                stats = self._blacklist_data.get('statistics', {})
                stats['total_added'] = stats.get('total_added', 0) + 1
                self._blacklist_data['statistics'] = stats
                
                self._dirty = True
                logger.debug(f"Added {symbol} to blacklist: {reason}")
                
                # Auto-save every 100 additions
                if len(self._blacklist_set) % 100 == 0:
                    self._save_blacklist()
    
    def add_symbols_batch(self, symbols: Dict[str, Dict[str, Any]]):
        """Add multiple symbols to blacklist at once"""
        with self.lock:
            added_count = 0
            for symbol, info in symbols.items():
                symbol = symbol.upper()
                if symbol not in self._blacklist_set:
                    self._blacklist_set.add(symbol)
                    self._blacklist_data['symbols'][symbol] = {
                        'added_date': datetime.now().isoformat(),
                        'reason': info.get('reason', 'Not found'),
                        'error_code': info.get('error_code', '404'),
                        'retry_count': 0,
                        'last_retry': None
                    }
                    added_count += 1
            
            if added_count > 0:
                stats = self._blacklist_data.get('statistics', {})
                stats['total_added'] = stats.get('total_added', 0) + added_count
                self._blacklist_data['statistics'] = stats
                self._dirty = True
                self._save_blacklist()
                logger.info(f"Added {added_count} symbols to blacklist")
    
    def remove_symbol(self, symbol: str):
        """Remove a symbol from blacklist (e.g., if it becomes valid again)"""
        symbol = symbol.upper()
        
        with self.lock:
            if symbol in self._blacklist_set:
                self._blacklist_set.remove(symbol)
                if symbol in self._blacklist_data['symbols']:
                    del self._blacklist_data['symbols'][symbol]
                
                stats = self._blacklist_data.get('statistics', {})
                stats['total_removed'] = stats.get('total_removed', 0) + 1
                self._blacklist_data['statistics'] = stats
                
                self._dirty = True
                logger.debug(f"Removed {symbol} from blacklist")
    
    def should_retry(self, symbol: str, retry_after_days: int = 30) -> bool:
        """Check if a blacklisted symbol should be retried"""
        symbol = symbol.upper()
        
        with self.lock:
            if symbol not in self._blacklist_data.get('symbols', {}):
                return True  # Not blacklisted
            
            symbol_info = self._blacklist_data['symbols'][symbol]
            last_retry = symbol_info.get('last_retry')
            
            if not last_retry:
                # Never retried
                added_date = symbol_info.get('added_date')
                if added_date:
                    added_dt = datetime.fromisoformat(added_date)
                    if datetime.now() - added_dt > timedelta(days=retry_after_days):
                        return True
            else:
                # Check last retry
                last_retry_dt = datetime.fromisoformat(last_retry)
                if datetime.now() - last_retry_dt > timedelta(days=retry_after_days):
                    return True
            
            return False
    
    def mark_retry(self, symbol: str, success: bool = False):
        """Mark that a symbol was retried"""
        symbol = symbol.upper()
        
        with self.lock:
            if symbol in self._blacklist_data.get('symbols', {}):
                symbol_info = self._blacklist_data['symbols'][symbol]
                symbol_info['last_retry'] = datetime.now().isoformat()
                symbol_info['retry_count'] = symbol_info.get('retry_count', 0) + 1
                
                if success:
                    # Remove from blacklist if successful
                    self.remove_symbol(symbol)
                else:
                    self._dirty = True
    
    def cleanup_old_entries(self, older_than_days: int = 180):
        """Remove blacklist entries older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        removed_count = 0
        
        with self.lock:
            symbols_to_remove = []
            
            for symbol, info in self._blacklist_data.get('symbols', {}).items():
                added_date = info.get('added_date')
                if added_date:
                    added_dt = datetime.fromisoformat(added_date)
                    if added_dt < cutoff_date:
                        # Check if it has been retried recently
                        last_retry = info.get('last_retry')
                        if last_retry:
                            last_retry_dt = datetime.fromisoformat(last_retry)
                            if last_retry_dt < cutoff_date:
                                symbols_to_remove.append(symbol)
                        else:
                            symbols_to_remove.append(symbol)
            
            for symbol in symbols_to_remove:
                self._blacklist_set.remove(symbol)
                del self._blacklist_data['symbols'][symbol]
                removed_count += 1
            
            if removed_count > 0:
                self._blacklist_data['statistics']['last_cleanup'] = datetime.now().isoformat()
                self._dirty = True
                self._save_blacklist()
                logger.info(f"Cleaned up {removed_count} old blacklist entries")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get blacklist statistics"""
        with self.lock:
            return {
                'total_blacklisted': len(self._blacklist_set),
                'statistics': self._blacklist_data.get('statistics', {}),
                'last_updated': self._blacklist_data.get('last_updated'),
                'sample_symbols': list(self._blacklist_set)[:10] if self._blacklist_set else []
            }
    
    def filter_valid_symbols(self, symbols: list) -> list:
        """Filter out blacklisted symbols from a list"""
        with self.lock:
            valid_symbols = [s for s in symbols if s.upper() not in self._blacklist_set]
            blacklisted_count = len(symbols) - len(valid_symbols)
            
            if blacklisted_count > 0:
                logger.debug(f"Filtered out {blacklisted_count} blacklisted symbols")
            
            return valid_symbols
    
    def export_blacklist(self, output_file: Path = None) -> Path:
        """Export blacklist to file"""
        if not output_file:
            output_file = self.data_dir / f"blacklist_export_{datetime.now().strftime('%Y%m%d')}.json"
        
        with self.lock:
            with open(output_file, 'w') as f:
                json.dump(self._blacklist_data, f, indent=2)
        
        logger.info(f"Exported blacklist to {output_file}")
        return output_file
    
    def import_blacklist(self, import_file: Path, merge: bool = True):
        """Import blacklist from file"""
        try:
            with open(import_file, 'r') as f:
                import_data = json.load(f)
            
            with self.lock:
                if merge:
                    # Merge with existing data
                    for symbol, info in import_data.get('symbols', {}).items():
                        if symbol not in self._blacklist_set:
                            self._blacklist_set.add(symbol)
                            self._blacklist_data['symbols'][symbol] = info
                else:
                    # Replace existing data
                    self._blacklist_data = import_data
                    self._blacklist_set = set(import_data.get('symbols', {}).keys())
                
                self._save_blacklist()
                logger.info(f"Imported blacklist from {import_file}")
                
        except Exception as e:
            logger.error(f"Failed to import blacklist: {e}")
    
    def save_if_dirty(self):
        """Save blacklist if there are unsaved changes"""
        with self.lock:
            if self._dirty:
                self._save_blacklist()


    def get_retry_candidates(self, retry_after_days: int = 30, max_retries: int = 3) -> List[str]:
        """Get list of symbols that should be retried"""
        candidates = []
        
        with self.lock:
            for symbol, info in self._blacklist_data.get('symbols', {}).items():
                retry_count = info.get('retry_count', 0)
                if retry_count < max_retries and self.should_retry(symbol, retry_after_days):
                    candidates.append(symbol)
        
        return candidates
    
    def get_blacklist_summary(self) -> Dict[str, Any]:
        """Get detailed summary of blacklist contents"""
        with self.lock:
            # Group by error code
            by_error_code = {}
            for symbol, info in self._blacklist_data.get('symbols', {}).items():
                error_code = info.get('error_code', 'UNKNOWN')
                if error_code not in by_error_code:
                    by_error_code[error_code] = []
                by_error_code[error_code].append(symbol)
            
            # Calculate age distribution
            age_distribution = {'<7d': 0, '7-30d': 0, '30-90d': 0, '90-180d': 0, '>180d': 0}
            now = datetime.now()
            
            for symbol, info in self._blacklist_data.get('symbols', {}).items():
                added_date = info.get('added_date')
                if added_date:
                    added_dt = datetime.fromisoformat(added_date)
                    age_days = (now - added_dt).days
                    
                    if age_days < 7:
                        age_distribution['<7d'] += 1
                    elif age_days < 30:
                        age_distribution['7-30d'] += 1
                    elif age_days < 90:
                        age_distribution['30-90d'] += 1
                    elif age_days < 180:
                        age_distribution['90-180d'] += 1
                    else:
                        age_distribution['>180d'] += 1
            
            return {
                'total_symbols': len(self._blacklist_set),
                'by_error_code': {k: len(v) for k, v in by_error_code.items()},
                'age_distribution': age_distribution,
                'statistics': self._blacklist_data.get('statistics', {}),
                'last_updated': self._blacklist_data.get('last_updated')
            }
    
    def __del__(self):
        """Ensure blacklist is saved on cleanup"""
        try:
            self.save_if_dirty()
        except:
            pass

# Global instance
_blacklist_instance = None
_blacklist_lock = threading.Lock()

def get_blacklist() -> SymbolBlacklist:
    """Get singleton blacklist instance"""
    global _blacklist_instance
    if _blacklist_instance is None:
        with _blacklist_lock:
            if _blacklist_instance is None:
                from config import DATA_DIR
                _blacklist_instance = SymbolBlacklist(DATA_DIR)
    return _blacklist_instance

# Utility functions for blacklist management
def manage_blacklist(command: str, *args, **kwargs):
    """Utility function to manage the blacklist"""
    blacklist = get_blacklist()
    
    if command == 'stats':
        return blacklist.get_statistics()
    elif command == 'summary':
        return blacklist.get_blacklist_summary()
    elif command == 'cleanup':
        days = kwargs.get('days', 180)
        blacklist.cleanup_old_entries(days)
    elif command == 'export':
        return blacklist.export_blacklist(kwargs.get('output_file'))
    elif command == 'import':
        blacklist.import_blacklist(kwargs.get('input_file'), kwargs.get('merge', True))
    elif command == 'retry':
        # Get symbols that should be retried
        return blacklist.get_retry_candidates(
            kwargs.get('retry_after_days', 30),
            kwargs.get('max_retries', 3)
        )
    elif command == 'remove':
        symbol = args[0] if args else None
        if symbol:
            blacklist.remove_symbol(symbol)
    elif command == 'add':
        symbol = args[0] if args else None
        reason = args[1] if len(args) > 1 else "Manual addition"
        error_code = args[2] if len(args) > 2 else "MANUAL"
        if symbol:
            blacklist.add_symbol(symbol, reason, error_code)
    elif command == 'check':
        symbol = args[0] if args else None
        if symbol:
            return blacklist.is_blacklisted(symbol)
    else:
        raise ValueError(f"Unknown blacklist command: {command}")

# Pre-populated known invalid symbols (common delisted/invalid tickers)
KNOWN_INVALID_SYMBOLS = {
    # Test symbols
    'TEST001', 'TEST002', 'TEST003', 'TEST004', 'TEST005',
    'DUMMY', 'FAKE', 'NULL', 'INVALID', 'ERROR',
    
    # Known delisted symbols (add more as discovered)
    'IGZ',  # The symbol from your error message
    
    # Add more known invalid symbols here
}