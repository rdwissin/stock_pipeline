# verify_data.py - Fixed version

__version__ = "1.5.0"
__copyright__ = "Copyright 2025, Richard D. Wissinger"
__author__ = "Richard D. Wissinger"
__email__ = "rick.wissinger@gmail.com"
__license__ = "MIT"  # or your chosen license
__status__ = "Production"


import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

def verify_database():
    """Comprehensive database verification"""
    db_path = Path('data/stocks_enhanced.db')
    
    if not db_path.exists():
        print("❌ Database does not exist!")
        return False
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print("=== Database Verification Report ===\n")
    
    # 1. Check stocks table
    cursor.execute("""
        SELECT 
            COUNT(DISTINCT symbol) as symbols,
            COUNT(DISTINCT exchange) as exchanges,
            COUNT(*) as total_records,
            MIN(snapshot_date) as earliest,
            MAX(snapshot_date) as latest
        FROM stocks
    """)
    stocks = cursor.fetchone()
    
    print("📊 Stocks Table:")
    print(f"   Unique symbols: {stocks['symbols']:,}")
    print(f"   Exchanges: {stocks['exchanges']}")
    print(f"   Total records: {stocks['total_records']:,}")
    print(f"   Date range: {stocks['earliest']} to {stocks['latest']}")
    
    # 2. Check by exchange
    cursor.execute("""
        SELECT exchange, COUNT(DISTINCT symbol) as count
        FROM stocks
        GROUP BY exchange
        ORDER BY count DESC
    """)
    
    print("\n📈 Stocks by Exchange:")
    for row in cursor.fetchall():
        exchange = row['exchange'] if row['exchange'] else 'NULL'
        print(f"   {exchange}: {row['count']:,} symbols")
    
    # 3. Check historical prices
    cursor.execute("""
        SELECT 
            COUNT(DISTINCT symbol) as symbols,
            COUNT(*) as total_records,
            MIN(date) as earliest,
            MAX(date) as latest
        FROM historical_prices
    """)
    historical = cursor.fetchone()
    
    print("\n📅 Historical Prices:")
    print(f"   Symbols with history: {historical['symbols']:,}")
    print(f"   Total price records: {historical['total_records']:,}")
    if historical['earliest']:
        print(f"   Date range: {historical['earliest']} to {historical['latest']}")
        
        # Calculate years of history
        start = datetime.strptime(historical['earliest'], '%Y-%m-%d')
        end = datetime.strptime(historical['latest'], '%Y-%m-%d')
        years = (end - start).days / 365.25
        print(f"   Years of history: {years:.1f}")
    
    # 4. Sample data quality check (with fixed formatting)
    cursor.execute("""
        SELECT symbol, current_price, market_cap, pe_ratio
        FROM stocks
        WHERE current_price IS NOT NULL
        ORDER BY market_cap DESC NULLS LAST
        LIMIT 5
    """)
    
    rows = cursor.fetchall()
    if rows:
        print("\n🏆 Top 5 Stocks by Market Cap:")
        for row in rows:
            pe_str = f"{row['pe_ratio']:.2f}" if row['pe_ratio'] else "N/A"
            mc_str = f"${row['market_cap']:,.0f}" if row['market_cap'] else "N/A"
            print(f"   {row['symbol']}: ${row['current_price']:.2f}, Market Cap: {mc_str}, PE: {pe_str}")
    
    # 5. Data completeness check
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(current_price) as has_price,
            COUNT(market_cap) as has_cap,
            COUNT(pe_ratio) as has_pe,
            COUNT(sector) as has_sector
        FROM stocks
        WHERE current_price IS NOT NULL
    """)
    complete = cursor.fetchone()
    
    if complete['total'] > 0:
        print("\n✅ Data Completeness (non-null prices):")
        print(f"   Has price: {complete['has_price']}/{complete['total']} ({100*complete['has_price']/complete['total']:.1f}%)")
        print(f"   Has market cap: {complete['has_cap']}/{complete['total']} ({100*complete['has_cap']/complete['total']:.1f}%)")
        print(f"   Has PE ratio: {complete['has_pe']}/{complete['total']} ({100*complete['has_pe']/complete['total']:.1f}%)")
        print(f"   Has sector: {complete['has_sector']}/{complete['total']} ({100*complete['has_sector']/complete['total']:.1f}%)")
    
    conn.close()
    
    # Overall assessment
    print("\n=== Assessment ===")
    if stocks['symbols'] > 5000 and historical['symbols'] > 1000:
        print("✅ Database is well populated")
        return True
    elif stocks['symbols'] > 0:
        print("⚠️ Database has some data but needs more")
        print("   Run: ./run_pipeline.sh --full-history --force")
        return True
    else:
        print("❌ Database is empty")
        print("   Run: ./run_pipeline.sh --full-history --force")
        return False

if __name__ == "__main__":
    verify_database()
