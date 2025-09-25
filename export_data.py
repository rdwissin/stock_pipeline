#!/usr/bin/env python3
"""
Enhanced Stock Data Export Utility
Export pipeline data in various formats for analysis and integration
Fixed version with SQL injection prevention and memory optimization
"""

__version__ = "1.5.0"
__copyright__ = "Copyright 2025, Richard D. Wissinger"
__author__ = "Richard D. Wissinger"
__email__ = "rick.wissinger@gmail.com"
__license__ = "MIT"
__status__ = "Production"

import sys
import csv
import json
import sqlite3
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Set
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Add the project directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from config import DB_PATH
from logger import setup_logger
from storage import get_database
from blacklist import get_blacklist

logger = setup_logger(__name__)

# ===============================================================================
# SECURITY: FIELD WHITELISTING
# ===============================================================================

# Define allowed fields for SQL queries (security whitelist)
ALLOWED_STOCK_FIELDS = {
    'symbol', 'name', 'sector', 'industry', 'exchange', 'market_cap', 
    'current_price', 'pe_ratio', 'dividend_yield', 'beta', 'volume', 
    'week_52_high', 'week_52_low', 'shares_outstanding', 'ipo_year', 
    'tradeable', 'is_blacklisted', 'blacklist_reason', 'snapshot_date',
    'enterprise_value', 'float_shares', 'previous_close', 'open_price',
    'day_high', 'day_low', 'avg_volume_3m', 'avg_volume_10d', 'peg_ratio',
    'pb_ratio', 'ps_ratio', 'price_to_book', 'dividend_rate', 'ex_dividend_date',
    'debt_to_equity', 'current_ratio', 'quick_ratio', 'return_on_equity',
    'return_on_assets', 'profit_margin', 'revenue_growth', 'earnings_growth',
    'revenue_per_share', 'book_value_per_share', 'employees', 'description',
    'website', 'country', 'shortable', 'short_ratio', 'short_percent_outstanding',
    'last_updated', 'data_source', 'created_at', 'updated_at'
}

# ===============================================================================
# DATA EXPORTER CLASS
# ===============================================================================

class DataExporter:
    """Comprehensive data export utility with security fixes"""
    
    def __init__(self):
        self.db_path = DB_PATH
        self.db = get_database()
        self.blacklist = get_blacklist()
        
    def get_available_snapshots(self) -> List[str]:
        """Get list of available snapshot dates"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT snapshot_date 
                FROM stocks 
                ORDER BY snapshot_date DESC
            """)
            return [row[0] for row in cursor.fetchall()]
    
    def get_latest_snapshot_date(self) -> Optional[str]:
        """Get the most recent snapshot date"""
        snapshots = self.get_available_snapshots()
        return snapshots[0] if snapshots else None
    
    def _validate_fields(self, fields: List[str], allowed: Set[str]) -> List[str]:
        """Validate field names against whitelist for SQL injection prevention"""
        validated = []
        for field in fields:
            if field.lower() in allowed or field.lower() == '*':
                validated.append(field)
            else:
                logger.warning(f"Ignoring invalid field: {field}")
        return validated if validated else list(allowed)[:10]  # Default to first 10 allowed fields
    
    def _validate_sql_query(self, query: str) -> bool:
        """Basic SQL injection prevention check"""
        dangerous_keywords = [
            'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 
            'EXEC', 'EXECUTE', 'REPLACE', 'TRUNCATE', 'MERGE'
        ]
        query_upper = query.upper()
        
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                logger.error(f"Dangerous SQL keyword detected: {keyword}")
                return False
        
        # Ensure query is read-only
        if not query_upper.strip().startswith('SELECT'):
            logger.error("Only SELECT queries are allowed")
            return False
            
        return True
    
    def export_stocks_csv(self, output_file: str, snapshot_date: str = None,
                         exchanges: List[str] = None, sectors: List[str] = None,
                         min_market_cap: float = 0, include_all_fields: bool = False,
                         chunk_size: int = 1000) -> int:
        """Export stocks data to CSV format with chunked processing"""
        logger.info(f"Exporting stocks to CSV: {output_file}")
        
        if not snapshot_date:
            snapshot_date = self.get_latest_snapshot_date()
        
        # Prepare fields with validation
        if include_all_fields:
            # Get actual column names from database
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(stocks)")
                db_columns = [row[1] for row in cursor.fetchall()]
                fields = [col for col in db_columns if col in ALLOWED_STOCK_FIELDS]
        else:
            fields = [
                'symbol', 'name', 'sector', 'industry', 'exchange', 'market_cap', 
                'current_price', 'pe_ratio', 'dividend_yield', 'beta', 'volume', 
                'week_52_high', 'week_52_low', 'shares_outstanding', 'ipo_year', 'tradeable'
            ]
        
        # Build query with parameterized WHERE clause
        where_conditions = ["snapshot_date = ?"]
        params = [snapshot_date]
        
        if exchanges:
            placeholders = ','.join(['?' for _ in exchanges])
            where_conditions.append(f"exchange IN ({placeholders})")
            params.extend(exchanges)
        
        if sectors:
            placeholders = ','.join(['?' for _ in sectors])
            where_conditions.append(f"sector IN ({placeholders})")
            params.extend(sectors)
        
        if min_market_cap > 0:
            where_conditions.append("market_cap >= ?")
            params.append(min_market_cap)
        
        # Build safe query with quoted field names
        field_list = ', '.join(f'"{field}"' for field in fields)
        query = f"""
            SELECT {field_list}
            FROM stocks 
            WHERE {' AND '.join(where_conditions)}
            ORDER BY market_cap DESC
        """
        
        row_count = 0
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            # Write CSV with chunked processing
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(fields)  # Header
                
                # Process in chunks to manage memory
                while True:
                    rows = cursor.fetchmany(chunk_size)
                    if not rows:
                        break
                    
                    for row in rows:
                        writer.writerow(row)
                        row_count += 1
                    
                    if row_count % 10000 == 0:
                        logger.debug(f"Processed {row_count} rows...")
        
        logger.info(f"Exported {row_count} stocks to {output_file}")
        return row_count
    
    def export_stocks_json(self, output_file: str, snapshot_date: str = None,
                          exchanges: List[str] = None, sectors: List[str] = None,
                          min_market_cap: float = 0, format_style: str = 'array',
                          chunk_size: int = 1000) -> int:
        """Export stocks data to JSON format with streaming for large datasets"""
        logger.info(f"Exporting stocks to JSON: {output_file}")
        
        if not snapshot_date:
            snapshot_date = self.get_latest_snapshot_date()
        
        # Build query with filters
        where_conditions = ["snapshot_date = ?"]
        params = [snapshot_date]
        
        if exchanges:
            placeholders = ','.join(['?' for _ in exchanges])
            where_conditions.append(f"exchange IN ({placeholders})")
            params.extend(exchanges)
        
        if sectors:
            placeholders = ','.join(['?' for _ in sectors])
            where_conditions.append(f"sector IN ({placeholders})")
            params.extend(sectors)
        
        if min_market_cap > 0:
            where_conditions.append("market_cap >= ?")
            params.append(min_market_cap)
        
        # Get only validated fields
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(stocks)")
            db_columns = [row[1] for row in cursor.fetchall()]
            valid_fields = [col for col in db_columns if col in ALLOWED_STOCK_FIELDS]
            field_list = ', '.join(f'"{field}"' for field in valid_fields)
        
        query = f"""
            SELECT {field_list}
            FROM stocks 
            WHERE {' AND '.join(where_conditions)}
            ORDER BY market_cap DESC
        """
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            columns = [description[0] for description in cursor.description]
            
            if format_style == 'array':
                # Stream to file for memory efficiency
                return self._export_json_array(cursor, columns, output_file, chunk_size)
                
            elif format_style == 'object':
                # Object format requires all data but can still be chunked
                return self._export_json_object(cursor, columns, output_file, chunk_size)
                
            elif format_style == 'metadata':
                # Metadata format with limited data
                return self._export_json_metadata(cursor, columns, output_file, snapshot_date,
                                                 exchanges, sectors, min_market_cap, chunk_size)
    
    def _export_json_array(self, cursor: sqlite3.Cursor, columns: List[str], 
                          output_file: str, chunk_size: int) -> int:
        """Export JSON in array format with streaming"""
        row_count = 0
        
        with open(output_file, 'w', encoding='utf-8') as jsonfile:
            jsonfile.write('[\n')
            first = True
            
            while True:
                rows = cursor.fetchmany(chunk_size)
                if not rows:
                    break
                
                for row in rows:
                    if not first:
                        jsonfile.write(',\n')
                    first = False
                    
                    stock_dict = dict(zip(columns, row))
                    stock_dict = {k: v if v is not None else None 
                                for k, v in stock_dict.items()}
                    json.dump(stock_dict, jsonfile, default=str)
                    row_count += 1
                
                if row_count % 10000 == 0:
                    logger.debug(f"Processed {row_count} rows...")
            
            jsonfile.write('\n]')
        
        return row_count
    
    def _export_json_object(self, cursor: sqlite3.Cursor, columns: List[str], 
                           output_file: str, chunk_size: int) -> int:
        """Export JSON in object format with chunked processing"""
        stocks_dict = {}
        row_count = 0
        
        while True:
            rows = cursor.fetchmany(chunk_size)
            if not rows:
                break
            
            for row in rows:
                stock_dict = dict(zip(columns, row))
                stock_dict = {k: v if v is not None else None 
                            for k, v in stock_dict.items()}
                symbol = stock_dict.get('symbol', f'unknown_{row_count}')
                stocks_dict[symbol] = stock_dict
                row_count += 1
            
            if row_count % 10000 == 0:
                logger.debug(f"Processed {row_count} rows...")
        
        with open(output_file, 'w', encoding='utf-8') as jsonfile:
            json.dump(stocks_dict, jsonfile, indent=2, default=str)
        
        return row_count
    
    def _export_json_metadata(self, cursor: sqlite3.Cursor, columns: List[str], 
                             output_file: str, snapshot_date: str,
                             exchanges: Optional[List[str]], sectors: Optional[List[str]],
                             min_market_cap: float, chunk_size: int,
                             max_records: int = 10000) -> int:
        """Export JSON with metadata and limited records"""
        metadata = {
            'export_date': datetime.now().isoformat(),
            'snapshot_date': snapshot_date,
            'version': __version__,
            'copyright': __copyright__,
            'filters': {
                'exchanges': exchanges,
                'sectors': sectors,
                'min_market_cap': min_market_cap
            }
        }
        
        stocks = []
        row_count = 0
        
        while row_count < max_records:
            rows = cursor.fetchmany(min(chunk_size, max_records - row_count))
            if not rows:
                break
            
            for row in rows:
                stock_dict = dict(zip(columns, row))
                stock_dict = {k: v if v is not None else None 
                            for k, v in stock_dict.items()}
                stocks.append(stock_dict)
                row_count += 1
        
        metadata['total_stocks'] = row_count
        metadata['limited_to'] = max_records if row_count >= max_records else None
        
        output_data = {'metadata': metadata, 'stocks': stocks}
        
        with open(output_file, 'w', encoding='utf-8') as jsonfile:
            json.dump(output_data, jsonfile, indent=2, default=str)
        
        return row_count
    
    def export_stocks_xml(self, output_file: str, snapshot_date: str = None,
                         exchanges: List[str] = None, sectors: List[str] = None,
                         min_market_cap: float = 0, chunk_size: int = 1000) -> int:
        """Export stocks data to XML format with memory-efficient processing"""
        logger.info(f"Exporting stocks to XML: {output_file}")
        
        if not snapshot_date:
            snapshot_date = self.get_latest_snapshot_date()
        
        # Build query with filters
        where_conditions = ["snapshot_date = ?"]
        params = [snapshot_date]
        
        if exchanges:
            placeholders = ','.join(['?' for _ in exchanges])
            where_conditions.append(f"exchange IN ({placeholders})")
            params.extend(exchanges)
        
        if sectors:
            placeholders = ','.join(['?' for _ in sectors])
            where_conditions.append(f"sector IN ({placeholders})")
            params.extend(sectors)
        
        if min_market_cap > 0:
            where_conditions.append("market_cap >= ?")
            params.append(min_market_cap)
        
        # Get validated fields
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(stocks)")
            db_columns = [row[1] for row in cursor.fetchall()]
            valid_fields = [col for col in db_columns if col in ALLOWED_STOCK_FIELDS]
            field_list = ', '.join(f'"{field}"' for field in valid_fields)
        
        query = f"""
            SELECT {field_list}
            FROM stocks 
            WHERE {' AND '.join(where_conditions)}
            ORDER BY market_cap DESC
        """
        
        # Create XML structure
        root = ET.Element("stocks")
        root.set("snapshot_date", snapshot_date or "")
        root.set("export_date", datetime.now().isoformat())
        root.set("version", __version__)
        root.set("copyright", __copyright__)
        
        stock_count = 0
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            columns = [description[0] for description in cursor.description]
            
            # Process in chunks for memory efficiency
            while True:
                rows = cursor.fetchmany(chunk_size)
                if not rows:
                    break
                
                for row in rows:
                    stock_elem = ET.SubElement(root, "stock")
                    
                    for col, value in zip(columns, row):
                        if value is not None:
                            elem = ET.SubElement(stock_elem, col)
                            elem.text = str(value)
                    
                    stock_count += 1
                
                if stock_count % 10000 == 0:
                    logger.debug(f"Processed {stock_count} stocks...")
        
        # Pretty print XML
        rough_string = ET.tostring(root, 'unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        
        # Write XML file
        with open(output_file, 'w', encoding='utf-8') as xmlfile:
            xmlfile.write(pretty_xml)
        
        logger.info(f"Exported {stock_count} stocks to {output_file}")
        return stock_count
    
    def export_market_summary(self, output_file: str, format_type: str = 'json',
                             snapshot_date: str = None) -> None:
        """Export market summary statistics"""
        logger.info(f"Exporting market summary to {format_type.upper()}: {output_file}")
        
        if not snapshot_date:
            snapshot_date = self.get_latest_snapshot_date()
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Overall market statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_stocks,
                    COUNT(CASE WHEN market_cap > 0 THEN 1 END) as stocks_with_cap,
                    AVG(CASE WHEN market_cap > 0 THEN market_cap END) as avg_market_cap,
                    SUM(CASE WHEN market_cap > 0 THEN market_cap END) as total_market_cap,
                    MAX(market_cap) as largest_market_cap,
                    AVG(CASE WHEN pe_ratio > 0 AND pe_ratio < 1000 THEN pe_ratio END) as avg_pe,
                    AVG(CASE WHEN dividend_yield > 0 THEN dividend_yield END) as avg_dividend_yield,
                    COUNT(CASE WHEN dividend_yield > 0 THEN 1 END) as dividend_paying_stocks
                FROM stocks WHERE snapshot_date = ?
            """, (snapshot_date,))
            
            overall_stats = cursor.fetchone()
            
            # Exchange breakdown
            cursor.execute("""
                SELECT exchange, COUNT(*) as count, 
                       AVG(CASE WHEN market_cap > 0 THEN market_cap END) as avg_cap,
                       SUM(CASE WHEN market_cap > 0 THEN market_cap END) as total_cap
                FROM stocks WHERE snapshot_date = ?
                GROUP BY exchange ORDER BY count DESC
            """, (snapshot_date,))
            
            exchanges = []
            for row in cursor.fetchall():
                exchanges.append({
                    'exchange': row[0],
                    'stock_count': row[1],
                    'avg_market_cap': row[2] or 0,
                    'total_market_cap': row[3] or 0
                })
            
            # Sector breakdown
            cursor.execute("""
                SELECT sector, COUNT(*) as count,
                       AVG(CASE WHEN market_cap > 0 THEN market_cap END) as avg_cap,
                       SUM(CASE WHEN market_cap > 0 THEN market_cap END) as total_cap
                FROM stocks WHERE snapshot_date = ? AND sector != 'Unknown'
                GROUP BY sector ORDER BY total_cap DESC LIMIT 20
            """, (snapshot_date,))
            
            sectors = []
            for row in cursor.fetchall():
                sectors.append({
                    'sector': row[0],
                    'stock_count': row[1],
                    'avg_market_cap': row[2] or 0,
                    'total_market_cap': row[3] or 0
                })
            
            # Market cap distribution
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN market_cap >= 200e9 THEN 'Mega Cap'
                        WHEN market_cap >= 10e9 THEN 'Large Cap'
                        WHEN market_cap >= 2e9 THEN 'Mid Cap'
                        WHEN market_cap >= 300e6 THEN 'Small Cap'
                        WHEN market_cap > 0 THEN 'Micro Cap'
                        ELSE 'No Market Cap'
                    END as category,
                    COUNT(*) as count,
                    SUM(market_cap) as total_cap
                FROM stocks WHERE snapshot_date = ?
                GROUP BY category
            """, (snapshot_date,))
            
            market_cap_dist = []
            for row in cursor.fetchall():
                market_cap_dist.append({
                    'category': row[0],
                    'stock_count': row[1],
                    'total_market_cap': row[2] or 0
                })
        
        # Add blacklist statistics
        blacklist_stats = self.blacklist.get_blacklist_summary()
        
        # Create summary data
        summary = {
            'metadata': {
                'snapshot_date': snapshot_date,
                'export_date': datetime.now().isoformat(),
                'data_source': 'Enhanced Stock Pipeline',
                'version': __version__,
                'copyright': __copyright__
            },
            'overall_statistics': {
                'total_stocks': overall_stats[0],
                'stocks_with_market_cap': overall_stats[1],
                'avg_market_cap': overall_stats[2] or 0,
                'total_market_cap': overall_stats[3] or 0,
                'largest_market_cap': overall_stats[4] or 0,
                'avg_pe_ratio': overall_stats[5] or 0,
                'avg_dividend_yield': overall_stats[6] or 0,
                'dividend_paying_stocks': overall_stats[7]
            },
            'exchange_breakdown': exchanges,
            'sector_breakdown': sectors,
            'market_cap_distribution': market_cap_dist,
            'blacklist_statistics': blacklist_stats
        }
        
        # Write output based on format
        if format_type.lower() == 'json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str)
        elif format_type.lower() == 'csv':
            # Export overall stats to CSV
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Overall statistics
                writer.writerow(['Metric', 'Value'])
                for key, value in summary['overall_statistics'].items():
                    writer.writerow([key.replace('_', ' ').title(), value])
                
                writer.writerow([])  # Empty row
                
                # Exchange breakdown
                writer.writerow(['Exchange', 'Stock Count', 'Avg Market Cap', 'Total Market Cap'])
                for exchange in summary['exchange_breakdown']:
                    writer.writerow([
                        exchange['exchange'],
                        exchange['stock_count'],
                        exchange['avg_market_cap'],
                        exchange['total_market_cap']
                    ])
        else:
            # Default to JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Market summary exported to {output_file}")
    
    def export_historical_data(self, output_file: str, symbol: str, 
                              days: int = 30, format_type: str = 'csv') -> int:
        """Export historical data for a specific symbol"""
        logger.info(f"Exporting historical data for {symbol}: {output_file}")
        
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get historical stock data (parameterized query)
            cursor.execute("""
                SELECT snapshot_date, symbol, name, current_price, market_cap,
                       pe_ratio, dividend_yield, volume, beta
                FROM stocks 
                WHERE symbol = ? AND snapshot_date >= ?
                ORDER BY snapshot_date DESC
            """, (symbol.upper(), cutoff_date))
            
            rows = cursor.fetchall()
            
            if not rows:
                logger.warning(f"No historical data found for {symbol}")
                return 0
        
        if format_type.lower() == 'csv':
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Date', 'Symbol', 'Name', 'Price', 'Market Cap',
                    'P/E Ratio', 'Dividend Yield', 'Volume', 'Beta'
                ])
                writer.writerows(rows)
        
        elif format_type.lower() == 'json':
            columns = ['date', 'symbol', 'name', 'price', 'market_cap',
                      'pe_ratio', 'dividend_yield', 'volume', 'beta']
            
            historical_data = []
            for row in rows:
                historical_data.append(dict(zip(columns, row)))
            
            output_data = {
                'symbol': symbol.upper(),
                'period_days': days,
                'data_points': len(historical_data),
                'version': __version__,
                'historical_data': historical_data
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"Exported {len(rows)} historical records for {symbol}")
        return len(rows)
    
    def export_blacklist(self, output_file: str = None) -> str:
        """Export blacklist data"""
        if not output_file:
            output_file = f"blacklist_export_{datetime.now().strftime('%Y%m%d')}.json"
        
        blacklist_path = self.blacklist.export_blacklist(Path(output_file))
        logger.info(f"Blacklist exported to {blacklist_path}")
        return str(blacklist_path)
    
    def export_custom_query(self, output_file: str, query: str,
                           format_type: str = 'csv', params: List = None) -> int:
        """Export results of a custom SQL query with validation"""
        logger.info(f"Exporting custom query results: {output_file}")
        
        if params is None:
            params = []
        
        # Validate query for SQL injection prevention
        if not self._validate_sql_query(query):
            raise ValueError("Query validation failed - potentially dangerous SQL detected")
        
        row_count = 0
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                # Get column names and data
                columns = [description[0] for description in cursor.description]
                
                if format_type.lower() == 'csv':
                    with open(output_file, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(columns)
                        
                        # Process in chunks
                        while True:
                            rows = cursor.fetchmany(1000)
                            if not rows:
                                break
                            writer.writerows(rows)
                            row_count += len(rows)
                
                elif format_type.lower() == 'json':
                    # For JSON, limit to reasonable size
                    max_records = 100000
                    data = []
                    
                    while row_count < max_records:
                        rows = cursor.fetchmany(1000)
                        if not rows:
                            break
                        
                        for row in rows:
                            data.append(dict(zip(columns, row)))
                            row_count += 1
                    
                    output_data = {
                        'query': query[:500],  # Truncate for safety
                        'export_date': datetime.now().isoformat(),
                        'row_count': row_count,
                        'version': __version__,
                        'copyright': __copyright__,
                        'data': data
                    }
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(output_data, f, indent=2, default=str)
        
        except sqlite3.Error as e:
            logger.error(f"Query execution failed: {e}")
            raise
        
        logger.info(f"Exported {row_count} records from custom query")
        return row_count

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description=f"Enhanced Stock Data Export Utility v{__version__}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{__copyright__}
Author: {__author__} ({__email__})

Examples:
  python export_data.py stocks --format csv --output stocks.csv
  python export_data.py stocks --format json --exchanges NYSE NASDAQ
  python export_data.py summary --output market_summary.json
  python export_data.py historical --symbol AAPL --days 30
  python export_data.py blacklist --output blacklist.json
  python export_data.py custom --query "SELECT * FROM stocks WHERE sector='Technology'"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Export commands')
    
    # Stocks export
    stocks_parser = subparsers.add_parser('stocks', help='Export stocks data')
    stocks_parser.add_argument('--format', choices=['csv', 'json', 'xml'], default='csv',
                              help='Output format (default: csv)')
    stocks_parser.add_argument('--output', required=True, help='Output file path')
    stocks_parser.add_argument('--date', help='Snapshot date (YYYY-MM-DD, default: latest)')
    stocks_parser.add_argument('--exchanges', nargs='+', help='Filter by exchanges')
    stocks_parser.add_argument('--sectors', nargs='+', help='Filter by sectors')
    stocks_parser.add_argument('--min-market-cap', type=float, default=0,
                              help='Minimum market cap filter')
    stocks_parser.add_argument('--all-fields', action='store_true',
                              help='Include all database fields')
    stocks_parser.add_argument('--json-format', choices=['array', 'object', 'metadata'],
                              default='array', help='JSON output format')
    
    # Market summary export
    summary_parser = subparsers.add_parser('summary', help='Export market summary')
    summary_parser.add_argument('--format', choices=['json', 'csv'], default='json',
                               help='Output format (default: json)')
    summary_parser.add_argument('--output', required=True, help='Output file path')
    summary_parser.add_argument('--date', help='Snapshot date (YYYY-MM-DD, default: latest)')
    
    # Historical data export
    hist_parser = subparsers.add_parser('historical', help='Export historical data for symbol')
    hist_parser.add_argument('--symbol', required=True, help='Stock symbol')
    hist_parser.add_argument('--format', choices=['csv', 'json'], default='csv',
                            help='Output format (default: csv)')
    hist_parser.add_argument('--output', help='Output file path (auto-generated if not specified)')
    hist_parser.add_argument('--days', type=int, default=30,
                            help='Number of days of history (default: 30)')
    
    # Blacklist export
    blacklist_parser = subparsers.add_parser('blacklist', help='Export blacklist data')
    blacklist_parser.add_argument('--output', help='Output file path')
    
    # Custom query export
    custom_parser = subparsers.add_parser('custom', help='Export custom query results')
    custom_parser.add_argument('--query', required=True, help='SQL query to execute')
    custom_parser.add_argument('--format', choices=['csv', 'json'], default='csv',
                              help='Output format (default: csv)')
    custom_parser.add_argument('--output', required=True, help='Output file path')
    custom_parser.add_argument('--params', nargs='+', help='Query parameters')
    
    # List available snapshots
    list_parser = subparsers.add_parser('list', help='List available snapshot dates')
    
    # Version
    parser.add_argument('--version', action='store_true',
                       help='Show version information')
    
    args = parser.parse_args()
    
    if args.version:
        print(f"Enhanced Stock Data Export Utility v{__version__}")
        print(f"{__copyright__}")
        print(f"Author: {__author__} ({__email__})")
        return
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize exporter
    try:
        exporter = DataExporter()
        logger.info("Data exporter initialized")
    except Exception as e:
        print(f"Error initializing exporter: {e}")
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'stocks':
            if args.format == 'json':
                count = exporter.export_stocks_json(
                    output_file=args.output,
                    snapshot_date=args.date,
                    exchanges=args.exchanges,
                    sectors=args.sectors,
                    min_market_cap=args.min_market_cap,
                    format_style=args.json_format
                )
            elif args.format == 'xml':
                count = exporter.export_stocks_xml(
                    output_file=args.output,
                    snapshot_date=args.date,
                    exchanges=args.exchanges,
                    sectors=args.sectors,
                    min_market_cap=args.min_market_cap
                )
            else:  # csv
                count = exporter.export_stocks_csv(
                    output_file=args.output,
                    snapshot_date=args.date,
                    exchanges=args.exchanges,
                    sectors=args.sectors,
                    min_market_cap=args.min_market_cap,
                    include_all_fields=args.all_fields
                )
            
            print(f"Successfully exported {count} stocks to {args.output}")
        
        elif args.command == 'summary':
            exporter.export_market_summary(
                output_file=args.output,
                format_type=args.format,
                snapshot_date=args.date
            )
            print(f"Successfully exported market summary to {args.output}")
        
        elif args.command == 'historical':
            output_file = args.output or f"{args.symbol.lower()}_historical_{args.days}d.{args.format}"
            count = exporter.export_historical_data(
                output_file=output_file,
                symbol=args.symbol,
                days=args.days,
                format_type=args.format
            )
            print(f"Successfully exported {count} historical records to {output_file}")
        
        elif args.command == 'blacklist':
            output_file = exporter.export_blacklist(args.output)
            print(f"Successfully exported blacklist to {output_file}")
        
        elif args.command == 'custom':
            params = args.params or []
            count = exporter.export_custom_query(
                output_file=args.output,
                query=args.query,
                format_type=args.format,
                params=params
            )
            print(f"Successfully exported {count} records to {args.output}")
        
        elif args.command == 'list':
            snapshots = exporter.get_available_snapshots()
            print(f"Available snapshot dates ({len(snapshots)} total):")
            for i, snapshot in enumerate(snapshots[:20]):  # Show last 20
                print(f"  {snapshot}")
            if len(snapshots) > 20:
                print(f"  ... and {len(snapshots) - 20} more")
    
    except Exception as e:
        logger.error(f"Export failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()