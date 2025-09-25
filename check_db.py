# Create a verification script (save as check_db.py)

__version__ = "1.5.0"
__copyright__ = "Copyright 2025, Richard D. Wissinger"
__author__ = "Richard D. Wissinger"
__email__ = "rick.wissinger@gmail.com"
__license__ = "MIT"  # or your chosen license
__status__ = "Production"

import sqlite3
from pathlib import Path

# Configuration for historical data limits
from config import DB_PATH

db_path = Path(DB_PATH)
if not db_path.exists():
    print("Database not found!")
    exit(1)

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

print("\n=== Database Status ===")
print(f"Size: {db_path.stat().st_size / 1024 / 1024:.2f} MB")

# Check tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print(f"\nTables found: {[t[0] for t in tables]}")

# Check record counts
for table in ['stocks', 'historical_prices', 'pipeline_runs']:
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"{table}: {count:,} records")
    except:
        print(f"{table}: Not found or empty")

# Check date ranges
try:
    cursor.execute("SELECT MIN(date), MAX(date) FROM historical_prices")
    min_date, max_date = cursor.fetchone()
    print(f"\nHistorical data range: {min_date} to {max_date}")
except:
    print("No historical data yet")

conn.close()
