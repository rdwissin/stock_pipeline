#!/usr/bin/env python3
"""
Enhanced Stock Data Analysis Tool
Comprehensive analysis and reporting for the stock pipeline database
"""

__version__ = "1.5.0"
__copyright__ = "Copyright 2025, Richard D. Wissinger"
__author__ = "Richard D. Wissinger"
__email__ = "rick.wissinger@gmail.com"
__license__ = "MIT"  # or your chosen license
__status__ = "Production"

import sys
import sqlite3
import argparse
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd

# Add the project directory to the path to import our modules
sys.path.insert(0, str(Path(__file__).parent))

from config import DB_PATH
from logger import setup_logger
from storage import get_database, get_market_overview
from blacklist import get_blacklist

logger = setup_logger(__name__)

class StockAnalyzer:
    """Comprehensive stock data analyzer"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        self.db = get_database()
        self.blacklist = get_blacklist()
        
    def get_latest_snapshot_date(self) -> Optional[str]:
        """Get the most recent snapshot date"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(snapshot_date) FROM stocks")
            result = cursor.fetchone()
            return result[0] if result[0] else None
    
    def market_overview(self, snapshot_date: str = None) -> Dict[str, Any]:
        """Generate comprehensive market overview"""
        if not snapshot_date:
            snapshot_date = self.get_latest_snapshot_date()
        
        logger.info(f"Generating market overview for {snapshot_date}")
        
        overview = {}
        
        with self.db.get_connection() as conn:
            # Overall statistics
            overview['summary'] = self._get_market_summary(conn, snapshot_date)
            
            # Exchange breakdown
            overview['exchanges'] = self._get_exchange_breakdown(conn, snapshot_date)
            
            # Sector analysis
            overview['sectors'] = self._get_sector_analysis(conn, snapshot_date)
            
            # Market cap distribution
            overview['market_cap_distribution'] = self._get_market_cap_distribution(conn, snapshot_date)
            
            # Valuation metrics
            overview['valuation_metrics'] = self._get_valuation_metrics(conn, snapshot_date)
            
            # Top performers
            overview['top_performers'] = self._get_top_performers(conn, snapshot_date)
            
            # Quality metrics
            overview['data_quality'] = self._get_data_quality_metrics(conn, snapshot_date)
            
            # Blacklist statistics
            blacklist_stats = self.blacklist.get_blacklist_summary()
            overview['blacklist'] = blacklist_stats
        
        return overview
    
    def _get_market_summary(self, conn: sqlite3.Connection, snapshot_date: str) -> Dict[str, Any]:
        """Get overall market summary statistics"""
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_stocks,
                COUNT(CASE WHEN market_cap > 0 THEN 1 END) as stocks_with_market_cap,
                AVG(CASE WHEN market_cap > 0 THEN market_cap END) as avg_market_cap,
                SUM(CASE WHEN market_cap > 0 THEN market_cap END) as total_market_cap,
                MAX(market_cap) as largest_market_cap,
                AVG(CASE WHEN pe_ratio > 0 AND pe_ratio < 1000 THEN pe_ratio END) as avg_pe_ratio,
                AVG(CASE WHEN dividend_yield > 0 THEN dividend_yield END) as avg_dividend_yield,
                COUNT(CASE WHEN dividend_yield > 0 THEN 1 END) as dividend_paying_stocks,
                AVG(CASE WHEN beta IS NOT NULL THEN beta END) as avg_beta
            FROM stocks 
            WHERE snapshot_date = ?
        """, (snapshot_date,))
        
        result = cursor.fetchone()
        
        return {
            'snapshot_date': snapshot_date,
            'total_stocks': result[0] or 0,
            'stocks_with_market_cap': result[1] or 0,
            'avg_market_cap': result[2] or 0,
            'total_market_cap': result[3] or 0,
            'largest_market_cap': result[4] or 0,
            'avg_pe_ratio': result[5] or 0,
            'avg_dividend_yield': result[6] or 0,
            'dividend_paying_stocks': result[7] or 0,
            'avg_beta': result[8] or 0
        }
    
    def _get_exchange_breakdown(self, conn: sqlite3.Connection, snapshot_date: str) -> List[Dict[str, Any]]:
        """Get detailed breakdown by exchange"""
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                exchange,
                COUNT(*) as stock_count,
                COUNT(CASE WHEN market_cap > 0 THEN 1 END) as stocks_with_cap,
                AVG(CASE WHEN market_cap > 0 THEN market_cap END) as avg_market_cap,
                SUM(CASE WHEN market_cap > 0 THEN market_cap END) as total_market_cap,
                MAX(market_cap) as largest_company,
                AVG(CASE WHEN pe_ratio > 0 AND pe_ratio < 1000 THEN pe_ratio END) as avg_pe_ratio,
                AVG(CASE WHEN dividend_yield > 0 THEN dividend_yield END) as avg_dividend_yield,
                COUNT(DISTINCT sector) as unique_sectors
            FROM stocks 
            WHERE snapshot_date = ?
            GROUP BY exchange
            ORDER BY stock_count DESC
        """, (snapshot_date,))
        
        exchanges = []
        for row in cursor.fetchall():
            exchanges.append({
                'exchange': row[0],
                'stock_count': row[1],
                'stocks_with_cap': row[2],
                'avg_market_cap': row[3] or 0,
                'total_market_cap': row[4] or 0,
                'largest_company': row[5] or 0,
                'avg_pe_ratio': row[6] or 0,
                'avg_dividend_yield': row[7] or 0,
                'unique_sectors': row[8] or 0
            })
        
        return exchanges
    
    def _get_sector_analysis(self, conn: sqlite3.Connection, snapshot_date: str) -> List[Dict[str, Any]]:
        """Get detailed sector analysis"""
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                sector,
                COUNT(*) as stock_count,
                AVG(CASE WHEN market_cap > 0 THEN market_cap END) as avg_market_cap,
                SUM(CASE WHEN market_cap > 0 THEN market_cap END) as total_market_cap,
                AVG(CASE WHEN pe_ratio > 0 AND pe_ratio < 1000 THEN pe_ratio END) as avg_pe_ratio,
                AVG(CASE WHEN dividend_yield > 0 THEN dividend_yield END) as avg_dividend_yield,
                AVG(CASE WHEN beta IS NOT NULL THEN beta END) as avg_beta,
                COUNT(CASE WHEN dividend_yield > 0 THEN 1 END) as dividend_stocks,
                MAX(market_cap) as largest_company_cap
            FROM stocks 
            WHERE snapshot_date = ? AND sector != 'Unknown'
            GROUP BY sector
            ORDER BY total_market_cap DESC
        """, (snapshot_date,))
        
        sectors = []
        for row in cursor.fetchall():
            sectors.append({
                'sector': row[0],
                'stock_count': row[1],
                'avg_market_cap': row[2] or 0,
                'total_market_cap': row[3] or 0,
                'avg_pe_ratio': row[4] or 0,
                'avg_dividend_yield': row[5] or 0,
                'avg_beta': row[6] or 0,
                'dividend_stocks': row[7] or 0,
                'largest_company_cap': row[8] or 0
            })
        
        return sectors
    
    def _get_market_cap_distribution(self, conn: sqlite3.Connection, snapshot_date: str) -> List[Dict[str, Any]]:
        """Get market cap distribution"""
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN market_cap >= 200e9 THEN 'Mega Cap (>$200B)'
                    WHEN market_cap >= 10e9 THEN 'Large Cap ($10B-$200B)'
                    WHEN market_cap >= 2e9 THEN 'Mid Cap ($2B-$10B)'
                    WHEN market_cap >= 300e6 THEN 'Small Cap ($300M-$2B)'
                    WHEN market_cap > 0 THEN 'Micro Cap (<$300M)'
                    ELSE 'No Market Cap'
                END as cap_category,
                COUNT(*) as count,
                AVG(market_cap) as avg_cap,
                SUM(market_cap) as total_cap,
                AVG(CASE WHEN pe_ratio > 0 AND pe_ratio < 1000 THEN pe_ratio END) as avg_pe
            FROM stocks 
            WHERE snapshot_date = ?
            GROUP BY cap_category
            ORDER BY 
                CASE 
                    WHEN cap_category = 'Mega Cap (>$200B)' THEN 1
                    WHEN cap_category = 'Large Cap ($10B-$200B)' THEN 2
                    WHEN cap_category = 'Mid Cap ($2B-$10B)' THEN 3
                    WHEN cap_category = 'Small Cap ($300M-$2B)' THEN 4
                    WHEN cap_category = 'Micro Cap (<$300M)' THEN 5
                    ELSE 6
                END
        """, (snapshot_date,))
        
        distribution = []
        for row in cursor.fetchall():
            distribution.append({
                'category': row[0],
                'count': row[1],
                'avg_cap': row[2] or 0,
                'total_cap': row[3] or 0,
                'avg_pe': row[4] or 0
            })
        
        return distribution
    
    def _get_valuation_metrics(self, conn: sqlite3.Connection, snapshot_date: str) -> Dict[str, Any]:
        """Get overall valuation metrics"""
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                AVG(CASE WHEN pe_ratio > 0 AND pe_ratio < 1000 THEN pe_ratio END) as avg_pe,
                AVG(CASE WHEN peg_ratio > 0 AND peg_ratio < 100 THEN peg_ratio END) as avg_peg,
                AVG(CASE WHEN price_to_book > 0 AND price_to_book < 100 THEN price_to_book END) as avg_pb,
                AVG(CASE WHEN dividend_yield > 0 THEN dividend_yield END) as avg_dividend_yield,
                COUNT(CASE WHEN pe_ratio > 0 AND pe_ratio < 15 THEN 1 END) as low_pe_count,
                COUNT(CASE WHEN dividend_yield > 4 THEN 1 END) as high_dividend_count,
                COUNT(CASE WHEN peg_ratio > 0 AND peg_ratio < 1 THEN 1 END) as low_peg_count
            FROM stocks 
            WHERE snapshot_date = ? AND market_cap > 1e9
        """, (snapshot_date,))
        
        result = cursor.fetchone()
        
        return {
            'avg_pe_ratio': result[0] or 0,
            'avg_peg_ratio': result[1] or 0,
            'avg_price_to_book': result[2] or 0,
            'avg_dividend_yield': result[3] or 0,
            'low_pe_stocks': result[4] or 0,
            'high_dividend_stocks': result[5] or 0,
            'low_peg_stocks': result[6] or 0
        }
    
    def _get_top_performers(self, conn: sqlite3.Connection, snapshot_date: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get top performing stocks by various metrics"""
        cursor = conn.cursor()
        
        performers = {}
        
        # Largest by market cap
        cursor.execute("""
            SELECT symbol, name, sector, exchange, market_cap
            FROM stocks 
            WHERE snapshot_date = ? AND market_cap > 0
            ORDER BY market_cap DESC
            LIMIT 20
        """, (snapshot_date,))
        
        performers['largest_by_market_cap'] = [
            {
                'symbol': row[0], 'name': row[1], 'sector': row[2],
                'exchange': row[3], 'market_cap': row[4]
            }
            for row in cursor.fetchall()
        ]
        
        # Highest dividend yields
        cursor.execute("""
            SELECT symbol, name, sector, dividend_yield, market_cap
            FROM stocks 
            WHERE snapshot_date = ? AND dividend_yield > 0 AND market_cap > 1e9
            ORDER BY dividend_yield DESC
            LIMIT 20
        """, (snapshot_date,))
        
        performers['highest_dividend_yield'] = [
            {
                'symbol': row[0], 'name': row[1], 'sector': row[2],
                'dividend_yield': row[3], 'market_cap': row[4]
            }
            for row in cursor.fetchall()
        ]
        
        # Lowest P/E ratios (value stocks)
        cursor.execute("""
            SELECT symbol, name, sector, pe_ratio, market_cap
            FROM stocks 
            WHERE snapshot_date = ? AND pe_ratio > 0 AND pe_ratio < 50 AND market_cap > 1e9
            ORDER BY pe_ratio ASC
            LIMIT 20
        """, (snapshot_date,))
        
        performers['lowest_pe_ratio'] = [
            {
                'symbol': row[0], 'name': row[1], 'sector': row[2],
                'pe_ratio': row[3], 'market_cap': row[4]
            }
            for row in cursor.fetchall()
        ]
        
        return performers
    
    def _get_data_quality_metrics(self, conn: sqlite3.Connection, snapshot_date: str) -> Dict[str, Any]:
        """Get data quality metrics"""
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(CASE WHEN name IS NOT NULL AND name != '' THEN 1 END) as records_with_name,
                COUNT(CASE WHEN sector IS NOT NULL AND sector != 'Unknown' THEN 1 END) as records_with_sector,
                COUNT(CASE WHEN industry IS NOT NULL AND industry != 'Unknown' THEN 1 END) as records_with_industry,
                COUNT(CASE WHEN market_cap IS NOT NULL AND market_cap > 0 THEN 1 END) as records_with_market_cap,
                COUNT(CASE WHEN current_price IS NOT NULL AND current_price > 0 THEN 1 END) as records_with_price,
                COUNT(CASE WHEN pe_ratio IS NOT NULL AND pe_ratio > 0 THEN 1 END) as records_with_pe,
                COUNT(CASE WHEN dividend_yield IS NOT NULL THEN 1 END) as records_with_dividend_data,
                COUNT(CASE WHEN beta IS NOT NULL THEN 1 END) as records_with_beta
            FROM stocks 
            WHERE snapshot_date = ?
        """, (snapshot_date,))
        
        result = cursor.fetchone()
        total = result[0] or 1  # Avoid division by zero
        
        return {
            'total_records': result[0] or 0,
            'completeness': {
                'name': (result[1] or 0) / total * 100,
                'sector': (result[2] or 0) / total * 100,
                'industry': (result[3] or 0) / total * 100,
                'market_cap': (result[4] or 0) / total * 100,
                'current_price': (result[5] or 0) / total * 100,
                'pe_ratio': (result[6] or 0) / total * 100,
                'dividend_data': (result[7] or 0) / total * 100,
                'beta': (result[8] or 0) / total * 100
            }
        }
    
    def pipeline_performance_analysis(self, days: int = 30) -> Dict[str, Any]:
        """Analyze pipeline performance over time"""
        logger.info(f"Analyzing pipeline performance over last {days} days")
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get pipeline runs
            cursor.execute("""
                SELECT 
                    run_date, status, total_stocks, processing_time_seconds,
                    memory_peak_mb, exchanges_processed
                FROM pipeline_runs 
                WHERE run_date >= date('now', '-{} days')
                ORDER BY run_date DESC
            """.format(days))
            
            runs = []
            for row in cursor.fetchall():
                runs.append({
                    'run_date': row[0],
                    'status': row[1],
                    'total_stocks': row[2] or 0,
                    'processing_time': row[3] or 0,
                    'memory_peak_mb': row[4] or 0,
                    'exchanges_processed': row[5] or ''
                })
            
            # Calculate statistics
            if runs:
                successful_runs = [r for r in runs if r['status'] == 'SUCCESS']
                
                performance = {
                    'total_runs': len(runs),
                    'successful_runs': len(successful_runs),
                    'success_rate': len(successful_runs) / len(runs) * 100,
                    'recent_runs': runs[:10],  # Last 10 runs
                }
                
                if successful_runs:
                    performance.update({
                        'avg_processing_time': sum(r['processing_time'] for r in successful_runs) / len(successful_runs),
                        'avg_stocks_collected': sum(r['total_stocks'] for r in successful_runs) / len(successful_runs),
                        'avg_memory_usage': sum(r['memory_peak_mb'] for r in successful_runs) / len(successful_runs),
                        'fastest_run': min(successful_runs, key=lambda x: x['processing_time']),
                        'slowest_run': max(successful_runs, key=lambda x: x['processing_time'])
                    })
            else:
                performance = {
                    'total_runs': 0,
                    'successful_runs': 0,
                    'success_rate': 0,
                    'recent_runs': []
                }
        
        return performance
    
    def historical_comparison(self, days: int = 7) -> Dict[str, Any]:
        """Compare current data with historical snapshots"""
        logger.info(f"Performing historical comparison over {days} days")
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get available snapshot dates
            cursor.execute("""
                SELECT DISTINCT snapshot_date 
                FROM stocks 
                WHERE snapshot_date >= date('now', '-{} days')
                ORDER BY snapshot_date DESC
                LIMIT 10
            """.format(days))
            
            dates = [row[0] for row in cursor.fetchall()]
            
            if len(dates) < 2:
                return {'error': 'Insufficient historical data for comparison'}
            
            # Compare latest vs previous
            latest_date = dates[0]
            previous_date = dates[1]
            
            # Stock count comparison
            cursor.execute("""
                SELECT 
                    (SELECT COUNT(*) FROM stocks WHERE snapshot_date = ?) as latest_count,
                    (SELECT COUNT(*) FROM stocks WHERE snapshot_date = ?) as previous_count
            """, (latest_date, previous_date))
            
            latest_count, previous_count = cursor.fetchone()
            
            # Market cap comparison
            cursor.execute("""
                SELECT 
                    (SELECT SUM(market_cap) FROM stocks WHERE snapshot_date = ? AND market_cap > 0) as latest_market_cap,
                    (SELECT SUM(market_cap) FROM stocks WHERE snapshot_date = ? AND market_cap > 0) as previous_market_cap
            """, (latest_date, previous_date))
            
            latest_market_cap, previous_market_cap = cursor.fetchone()
            
            comparison = {
                'latest_date': latest_date,
                'previous_date': previous_date,
                'stock_count_change': latest_count - previous_count if previous_count else 0,
                'stock_count_change_pct': ((latest_count - previous_count) / previous_count * 100) if previous_count else 0,
                'market_cap_change': (latest_market_cap or 0) - (previous_market_cap or 0),
                'market_cap_change_pct': (((latest_market_cap or 0) - (previous_market_cap or 0)) / (previous_market_cap or 1) * 100) if previous_market_cap else 0,
                'available_dates': dates
            }
        
        return comparison
    
    def generate_report(self, output_format: str = 'json', output_file: str = None) -> str:
        """Generate comprehensive analysis report"""
        logger.info(f"Generating comprehensive analysis report in {output_format} format")
        
        # Gather all analysis data
        latest_date = self.get_latest_snapshot_date()
        if not latest_date:
            return json.dumps({'error': 'No data available'})
        
        report = {
            'report_generated': datetime.now().isoformat(),
            'version': __version__,
            'copyright': __copyright__,
            'data_as_of': latest_date,
            'market_overview': self.market_overview(latest_date),
            'pipeline_performance': self.pipeline_performance_analysis(),
            'historical_comparison': self.historical_comparison()
        }
        
        # Format output
        if output_format.lower() == 'json':
            output = json.dumps(report, indent=2, default=str)
        elif output_format.lower() == 'summary':
            output = self._format_summary_report(report)
        else:
            output = json.dumps(report, indent=2, default=str)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(output)
            logger.info(f"Report saved to {output_file}")
        
        return output
    
    def _format_summary_report(self, report: Dict[str, Any]) -> str:
        """Format report as human-readable summary"""
        summary = []
        
        # Header
        summary.append(f"ENHANCED STOCK PIPELINE v{__version__} - ANALYSIS REPORT")
        summary.append("=" * 50)
        summary.append(f"Generated: {report['report_generated']}")
        summary.append(f"Data as of: {report['data_as_of']}")
        summary.append(f"{__copyright__}")
        summary.append("")
        
        # Market Overview
        market = report['market_overview']['summary']
        summary.append("MARKET OVERVIEW")
        summary.append("-" * 20)
        summary.append(f"Total Stocks: {market['total_stocks']:,}")
        summary.append(f"Total Market Cap: ${market['total_market_cap']/1e12:.2f}T")
        summary.append(f"Average P/E Ratio: {market['avg_pe_ratio']:.1f}")
        summary.append(f"Average Dividend Yield: {market['avg_dividend_yield']:.2f}%")
        summary.append(f"Dividend Paying Stocks: {market['dividend_paying_stocks']:,}")
        summary.append("")
        
        # Blacklist Info
        if 'blacklist' in report['market_overview']:
            blacklist = report['market_overview']['blacklist']
            summary.append("BLACKLIST STATISTICS")
            summary.append("-" * 20)
            summary.append(f"Total Blacklisted Symbols: {blacklist['total_symbols']}")
            if 'by_error_code' in blacklist:
                summary.append("By Error Code:")
                for code, count in blacklist['by_error_code'].items():
                    summary.append(f"  {code}: {count}")
            summary.append("")
        
        # Exchange Breakdown
        summary.append("EXCHANGE BREAKDOWN")
        summary.append("-" * 20)
        for exchange in report['market_overview']['exchanges']:
            summary.append(f"{exchange['exchange']:8}: {exchange['stock_count']:5,} stocks, "
                         f"${exchange['total_market_cap']/1e12:5.2f}T market cap")
        summary.append("")
        
        # Top Sectors
        summary.append("TOP SECTORS BY MARKET CAP")
        summary.append("-" * 30)
        for i, sector in enumerate(report['market_overview']['sectors'][:10]):
            summary.append(f"{i+1:2}. {sector['sector']:25}: "
                         f"${sector['total_market_cap']/1e12:5.2f}T "
                         f"({sector['stock_count']:,} stocks)")
        summary.append("")
        
        # Pipeline Performance
        if 'pipeline_performance' in report:
            perf = report['pipeline_performance']
            summary.append("PIPELINE PERFORMANCE")
            summary.append("-" * 25)
            summary.append(f"Success Rate: {perf['success_rate']:.1f}%")
            summary.append(f"Total Runs: {perf['total_runs']}")
            if 'avg_processing_time' in perf:
                summary.append(f"Avg Processing Time: {perf['avg_processing_time']:.1f}s")
                summary.append(f"Avg Stocks Collected: {perf['avg_stocks_collected']:,.0f}")
                summary.append(f"Avg Memory Usage: {perf['avg_memory_usage']:.1f}MB")
        
        return "\n".join(summary)

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description=f"Enhanced Stock Data Analysis Tool v{__version__}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{__copyright__}
Author: {__author__} ({__email__})

Examples:
  python analyze_data.py overview                    # Market overview
  python analyze_data.py report --format summary     # Summary report
  python analyze_data.py report --output report.json # Save to file
  python analyze_data.py performance                 # Pipeline performance
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Analysis commands')
    
    # Market overview command
    overview_parser = subparsers.add_parser('overview', help='Generate market overview')
    overview_parser.add_argument('--date', help='Specific snapshot date (YYYY-MM-DD)')
    overview_parser.add_argument('--format', choices=['json', 'summary'], default='summary',
                               help='Output format')
    
    # Full report command
    report_parser = subparsers.add_parser('report', help='Generate comprehensive report')
    report_parser.add_argument('--format', choices=['json', 'summary'], default='summary',
                             help='Output format')
    report_parser.add_argument('--output', help='Output file path')
    
    # Performance analysis command
    perf_parser = subparsers.add_parser('performance', help='Analyze pipeline performance')
    perf_parser.add_argument('--days', type=int, default=30,
                           help='Number of days to analyze (default: 30)')
    
    # Historical comparison command
    hist_parser = subparsers.add_parser('historical', help='Historical comparison')
    hist_parser.add_argument('--days', type=int, default=7,
                           help='Number of days to compare (default: 7)')
    
    # Version command
    parser.add_argument('--version', action='store_true',
                       help='Show version information')
    
    args = parser.parse_args()
    
    if args.version:
        print(f"Enhanced Stock Data Analysis Tool v{__version__}")
        print(f"{__copyright__}")
        print(f"Author: {__author__} ({__email__})")
        return
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize analyzer
    try:
        analyzer = StockAnalyzer()
    except Exception as e:
        print(f"Error initializing analyzer: {e}")
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'overview':
            overview = analyzer.market_overview(args.date)
            if args.format == 'json':
                print(json.dumps(overview, indent=2, default=str))
            else:
                print(analyzer._format_summary_report({'market_overview': overview}))
        
        elif args.command == 'report':
            report = analyzer.generate_report(args.format, args.output)
            if not args.output:
                print(report)
        
        elif args.command == 'performance':
            performance = analyzer.pipeline_performance_analysis(args.days)
            print(json.dumps(performance, indent=2, default=str))
        
        elif args.command == 'historical':
            comparison = analyzer.historical_comparison(args.days)
            print(json.dumps(comparison, indent=2, default=str))
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()