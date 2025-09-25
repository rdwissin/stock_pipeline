#!/bin/bash

# Stock Pipeline Maintenance Script v1.5.0
# Comprehensive maintenance tools for database and system management

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
DATA_DIR="$SCRIPT_DIR/data"
CACHE_DIR="$SCRIPT_DIR/cache"
BACKUP_DIR="$SCRIPT_DIR/backups"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

show_help() {
    echo -e "${BLUE}Stock Pipeline Maintenance Tool v1.2.0${NC}"
    echo
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  clean-logs [days]    Clean log files older than N days (default: 30)"
    echo "  clean-cache          Clear all cached data"
    echo "  backup-db            Create database backup"
    echo "  optimize-db          Optimize database performance"
    echo "  check-health         Run comprehensive health check"
    echo "  reset-permissions    Fix file permissions"
    echo "  update-deps          Update Python dependencies"
    echo "  show-stats           Show database statistics"
    echo "  vacuum-db            Run VACUUM to reclaim space"
    echo "  analyze-db           Update database statistics"
    echo "  check-integrity      Check database integrity"
    echo "  export-backup        Export database to SQL format"
    echo "  clean-historical     Remove old historical data"
    echo "  help                 Show this help message"
}

clean_logs() {
    local days=${1:-30}
    echo -e "${BLUE}🧹 Cleaning logs older than $days days...${NC}"
    
    local deleted=0
    
    # Create backup of recent logs first
    local backup_file="$LOG_DIR/logs_backup_$(date +%Y%m%d).tar.gz"
    echo "  Creating log backup: $backup_file"
    tar -czf "$backup_file" -C "$LOG_DIR" --exclude="*.tar.gz" . 2>/dev/null
    
    # Clean old logs
    find "$LOG_DIR" -name "*.log" -mtime +$days -print0 | while IFS= read -r -d '' file; do
        echo "  Deleting: $(basename "$file")"
        rm "$file"
        deleted=$((deleted + 1))
    done
    
    # Clean old backups
    find "$LOG_DIR" -name "*.tar.gz" -mtime +90 -delete
    
    echo -e "${GREEN}✅ Log cleanup completed${NC}"
}

clean_cache() {
    echo -e "${BLUE}🧹 Cleaning cache directory...${NC}"
    
    if [ -d "$CACHE_DIR" ]; then
        local cache_size=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)
        echo "  Current cache size: $cache_size"
        
        rm -rf "$CACHE_DIR"/*
        echo -e "${GREEN}✅ Cache cleared${NC}"
    else
        echo -e "${YELLOW}⚠️  Cache directory not found${NC}"
    fi
    
    # Also clean Python cache
    find "$SCRIPT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
    echo "  Python cache cleared"
}

backup_database() {
    echo -e "${BLUE}💾 Creating database backup...${NC}"
    
    mkdir -p "$BACKUP_DIR"
    
    local db_path="$DATA_DIR/stocks_enhanced.db"
    if [ -f "$db_path" ]; then
        local timestamp=$(date +%Y%m%d_%H%M%S)
        local backup_path="$BACKUP_DIR/stocks_backup_${timestamp}.db"
        
        # Show current database size
        local db_size=$(ls -lh "$db_path" | awk '{print $5}')
        echo "  Current database size: $db_size"
        
        # Create backup
        cp "$db_path" "$backup_path"
        
        # Compress backup
        gzip "$backup_path"
        local compressed_size=$(ls -lh "${backup_path}.gz" | awk '{print $5}')
        
        echo -e "${GREEN}✅ Database backed up to: ${backup_path}.gz${NC}"
        echo "  Compressed size: $compressed_size"
        
        # Clean old backups (keep last 10)
        ls -t "$BACKUP_DIR"/stocks_backup_*.db.gz | tail -n +11 | xargs -r rm
        echo "  Old backups cleaned (keeping last 10)"
    else
        echo -e "${RED}❌ Database not found${NC}"
    fi
}

optimize_database() {
    echo -e "${BLUE}⚡ Optimizing database for 10-year data...${NC}"
    
    local db_path="$DATA_DIR/stocks_enhanced.db"
    if [ -f "$db_path" ] && command -v sqlite3 &> /dev/null; then
        # Show initial size
        local initial_size=$(ls -lh "$db_path" | awk '{print $5}')
        echo "  Initial database size: $initial_size"
        
        # Run optimization commands
        echo "  Running VACUUM..."
        sqlite3 "$db_path" "VACUUM;"
        
        echo "  Running ANALYZE..."
        sqlite3 "$db_path" "ANALYZE;"
        
        echo "  Optimizing query planner..."
        sqlite3 "$db_path" "PRAGMA optimize;"
        
        echo "  Setting optimal pragmas..."
        sqlite3 "$db_path" "
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            PRAGMA cache_size=20000;
            PRAGMA temp_store=MEMORY;
            PRAGMA mmap_size=536870912;
        "
        
        # Show final size
        local final_size=$(ls -lh "$db_path" | awk '{print $5}')
        echo "  Final database size: $final_size"
        
        echo -e "${GREEN}✅ Database optimization completed${NC}"
    else
        echo -e "${RED}❌ Database not found or SQLite not available${NC}"
    fi
}

vacuum_database() {
    echo -e "${BLUE}🗜️ Running VACUUM to reclaim space...${NC}"
    
    local db_path="$DATA_DIR/stocks_enhanced.db"
    if [ -f "$db_path" ] && command -v sqlite3 &> /dev/null; then
        local initial_size=$(ls -lh "$db_path" | awk '{print $5}')
        echo "  Initial size: $initial_size"
        
        sqlite3 "$db_path" "VACUUM;"
        
        local final_size=$(ls -lh "$db_path" | awk '{print $5}')
        echo "  Final size: $final_size"
        
        echo -e "${GREEN}✅ VACUUM completed${NC}"
    fi
}

analyze_database() {
    echo -e "${BLUE}📊 Updating database statistics...${NC}"
    
    local db_path="$DATA_DIR/stocks_enhanced.db"
    if [ -f "$db_path" ] && command -v sqlite3 &> /dev/null; then
        sqlite3 "$db_path" "ANALYZE;"
        echo -e "${GREEN}✅ Statistics updated${NC}"
    fi
}

check_integrity() {
    echo -e "${BLUE}🔍 Checking database integrity...${NC}"
    
    local db_path="$DATA_DIR/stocks_enhanced.db"
    if [ -f "$db_path" ] && command -v sqlite3 &> /dev/null; then
        local result=$(sqlite3 "$db_path" "PRAGMA integrity_check;" 2>/dev/null)
        
        if [ "$result" = "ok" ]; then
            echo -e "${GREEN}✅ Database integrity check passed${NC}"
        else
            echo -e "${RED}❌ Database integrity issues detected:${NC}"
            echo "$result"
            echo
            echo "Recommended actions:"
            echo "  1. Create a backup immediately"
            echo "  2. Export data to SQL"
            echo "  3. Recreate database from export"
        fi
        
        # Quick check
        echo
        echo "Quick check results:"
        sqlite3 "$db_path" "PRAGMA quick_check;" 2>/dev/null
    fi
}

export_backup() {
    echo -e "${BLUE}📤 Exporting database to SQL format...${NC}"
    
    local db_path="$DATA_DIR/stocks_enhanced.db"
    if [ -f "$db_path" ] && command -v sqlite3 &> /dev/null; then
        local export_path="$BACKUP_DIR/stocks_export_$(date +%Y%m%d_%H%M%S).sql"
        
        echo "  Exporting to: $export_path"
        sqlite3 "$db_path" .dump > "$export_path"
        
        # Compress the export
        gzip "$export_path"
        
        local export_size=$(ls -lh "${export_path}.gz" | awk '{print $5}')
        echo -e "${GREEN}✅ Database exported to: ${export_path}.gz${NC}"
        echo "  Export size: $export_size"
    fi
}

clean_historical() {
    echo -e "${BLUE}🧹 Cleaning old historical data...${NC}"
    
    local days=${1:-3650}  # Default: keep 10 years
    echo "  Keeping last $days days of historical data"
    
    local db_path="$DATA_DIR/stocks_enhanced.db"
    if [ -f "$db_path" ] && command -v sqlite3 &> /dev/null; then
        local cutoff_date=$(date -d "$days days ago" +%Y-%m-%d 2>/dev/null || \
                          date -v -${days}d +%Y-%m-%d 2>/dev/null)
        
        echo "  Removing data before: $cutoff_date"
        
        # Count records to be deleted
        local count=$(sqlite3 "$db_path" "
            SELECT COUNT(*) FROM historical_prices 
            WHERE date < '$cutoff_date';
        ")
        
        echo "  Records to delete: $count"
        
        if [ "$count" -gt 0 ]; then
            echo -n "  Proceed? (y/N): "
            read confirm
            if [[ "$confirm" =~ ^[Yy]$ ]]; then
                sqlite3 "$db_path" "
                    DELETE FROM historical_prices 
                    WHERE date < '$cutoff_date';
                "
                echo -e "${GREEN}✅ Deleted $count old records${NC}"
                
                # Run VACUUM to reclaim space
                echo "  Running VACUUM to reclaim space..."
                sqlite3 "$db_path" "VACUUM;"
            fi
        fi
    fi
}

check_health() {
    echo -e "${BLUE}🏥 Running comprehensive health check...${NC}"
    
    # Check required files
    echo
    echo "Essential Files:"
    local required_files=("config.py" "main.py" "fetcher.py" "storage.py" "run_pipeline.sh")
    for file in "${required_files[@]}"; do
        if [ -f "$SCRIPT_DIR/$file" ]; then
            echo -e "${GREEN}✅ $file${NC}"
        else
            echo -e "${RED}❌ $file missing${NC}"
        fi
    done
    
    # Check permissions
    echo
    echo "Permissions:"
    if [ -x "$SCRIPT_DIR/run_pipeline.sh" ]; then
        echo -e "${GREEN}✅ run_pipeline.sh is executable${NC}"
    else
        echo -e "${RED}❌ run_pipeline.sh is not executable${NC}"
    fi
    
    # Check virtual environment
    echo
    echo "Python Environment:"
    if [ -d "$SCRIPT_DIR/venv" ]; then
        echo -e "${GREEN}✅ Virtual environment exists${NC}"
        
        # Check Python version
        if [ -f "$SCRIPT_DIR/venv/bin/python" ]; then
            local py_version=$("$SCRIPT_DIR/venv/bin/python" --version 2>&1)
            echo "  Python version: $py_version"
        fi
    else
        echo -e "${YELLOW}⚠️  Virtual environment not found${NC}"
    fi
    
    # Check database
    echo
    echo "Database:"
    local db_path="$DATA_DIR/stocks_enhanced.db"
    if [ -f "$db_path" ]; then
        local db_size=$(ls -lh "$db_path" | awk '{print $5}')
        echo -e "${GREEN}✅ Database exists (size: $db_size)${NC}"
        
        if command -v sqlite3 &> /dev/null; then
            # Check integrity
            local integrity=$(sqlite3 "$db_path" "PRAGMA integrity_check;" 2>/dev/null)
            if [ "$integrity" = "ok" ]; then
                echo -e "${GREEN}✅ Database integrity OK${NC}"
            else
                echo -e "${RED}❌ Database integrity issues${NC}"
            fi
            
            # Get record counts
            local stock_count=$(sqlite3 "$db_path" "SELECT COUNT(*) FROM stocks;" 2>/dev/null)
            local hist_count=$(sqlite3 "$db_path" "SELECT COUNT(*) FROM historical_prices;" 2>/dev/null)
            echo "  Stock records: ${stock_count:-0}"
            echo "  Historical records: ${hist_count:-0}"
        fi
    else
        echo -e "${YELLOW}⚠️  Database not found (will be created on first run)${NC}"
    fi
    
    # Check disk space
    echo
    echo "Disk Space:"
    local disk_avail=$(df -h "$SCRIPT_DIR" | awk 'NR==2 {print $4}')
    echo "  Available: $disk_avail"
    
    # Check for recent errors
    echo
    echo "Recent Errors:"
    if [ -d "$LOG_DIR" ]; then
        local error_count=$(grep -l "ERROR\|CRITICAL" "$LOG_DIR"/*.log 2>/dev/null | wc -l)
        if [ "$error_count" -gt 0 ]; then
            echo -e "${YELLOW}⚠️  Found errors in $error_count log files${NC}"
            echo "  Recent errors:"
            grep "ERROR\|CRITICAL" "$LOG_DIR"/*.log 2>/dev/null | tail -3
        else
            echo -e "${GREEN}✅ No recent errors found${NC}"
        fi
    fi
}

reset_permissions() {
    echo -e "${BLUE}🔧 Resetting file permissions...${NC}"
    
    # Make scripts executable
    chmod +x "$SCRIPT_DIR"/*.sh 2>/dev/null || true
    echo "  Made shell scripts executable"
    
    # Fix directory permissions
    chmod 755 "$SCRIPT_DIR"
    chmod 755 "$LOG_DIR" "$DATA_DIR" "$CACHE_DIR" "$BACKUP_DIR" 2>/dev/null || true
    echo "  Fixed directory permissions"
    
    # Fix log permissions
    chmod 644 "$LOG_DIR"/*.log 2>/dev/null || true
    echo "  Fixed log file permissions"
    
    # Fix database permissions
    if [ -f "$DATA_DIR/stocks_enhanced.db" ]; then
        chmod 644 "$DATA_DIR/stocks_enhanced.db"
        echo "  Fixed database permissions"
    fi
    
    echo -e "${GREEN}✅ Permissions reset${NC}"
}

update_dependencies() {
    echo -e "${BLUE}📦 Updating Python dependencies...${NC}"
    
    if [ -f "$SCRIPT_DIR/requirements.txt" ] && [ -d "$SCRIPT_DIR/venv" ]; then
        source "$SCRIPT_DIR/venv/bin/activate"
        
        echo "  Upgrading pip..."
        pip install --upgrade pip --quiet
        
        echo "  Updating dependencies..."
        pip install -r "$SCRIPT_DIR/requirements.txt" --upgrade
        
        echo "  Current package versions:"
        pip list | grep -E "yfinance|pandas|requests|psutil"
        
        echo -e "${GREEN}✅ Dependencies updated${NC}"
    else
        echo -e "${RED}❌ Requirements file or virtual environment not found${NC}"
    fi
}

show_statistics() {
    echo -e "${BLUE}📊 Database Statistics (10-Year Data)${NC}"
    
    local db_path="$DATA_DIR/stocks_enhanced.db"
    if [ -f "$db_path" ] && command -v sqlite3 &> /dev/null; then
        # Database size
        local db_size=$(ls -lh "$db_path" | awk '{print $5}')
        echo
        echo "Database size: $db_size"
        
        # Table sizes
        echo
        echo "Table record counts:"
        sqlite3 "$db_path" "
            SELECT 'Stocks' as table_name, COUNT(*) as records FROM stocks
            UNION ALL
            SELECT 'Historical Prices', COUNT(*) FROM historical_prices
            UNION ALL
            SELECT 'Blacklist', COUNT(*) FROM blacklist WHERE is_active = 1
            UNION ALL
            SELECT 'Pipeline Runs', COUNT(*) FROM pipeline_runs
            UNION ALL
            SELECT 'Data Quality Issues', COUNT(*) FROM data_quality;
        " | column -t -s '|'
        
        # Latest snapshot info
        echo
        echo "Latest snapshot:"
        sqlite3 "$db_path" "
            SELECT 
                snapshot_date,
                COUNT(*) as stocks,
                COUNT(CASE WHEN is_blacklisted = 1 THEN 1 END) as blacklisted
            FROM stocks 
            WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM stocks)
            GROUP BY snapshot_date;
        " | column -t -s '|'
        
        # Historical data coverage
        echo
        echo "Historical data coverage:"
        sqlite3 "$db_path" "
            SELECT 
                printf('%.1f', (julianday('now') - julianday(MIN(date))) / 365.25) as years_of_data,
                COUNT(DISTINCT symbol) as symbols_with_history,
                COUNT(*) as total_price_records,
                printf('%.0f', AVG(records_per_symbol)) as avg_records_per_symbol
            FROM (
                SELECT symbol, COUNT(*) as records_per_symbol
                FROM historical_prices
                GROUP BY symbol
            );
        " | column -t -s '|'
        
        # Exchange breakdown
        echo
        echo "Exchange breakdown (latest):"
        sqlite3 "$db_path" "
            SELECT 
                exchange,
                COUNT(*) as stocks,
                printf('%.2f', AVG(market_cap)/1e9) as avg_market_cap_billions
            FROM stocks 
            WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM stocks)
            GROUP BY exchange
            ORDER BY COUNT(*) DESC;
        " | column -t -s '|'
        
        # Top sectors
        echo
        echo "Top 5 sectors by stock count:"
        sqlite3 "$db_path" "
            SELECT 
                sector,
                COUNT(*) as stocks
            FROM stocks 
            WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM stocks)
                AND sector != 'Unknown'
            GROUP BY sector
            ORDER BY COUNT(*) DESC
            LIMIT 5;
        " | column -t -s '|'
        
        # Pipeline performance
        echo
        echo "Recent pipeline runs:"
        sqlite3 "$db_path" "
            SELECT 
                run_date,
                status,
                total_stocks,
                printf('%.1f', processing_time_seconds/60) as minutes
            FROM pipeline_runs
            ORDER BY run_date DESC
            LIMIT 5;
        " | column -t -s '|'
        
    else
        echo -e "${RED}❌ Database not found or SQLite not available${NC}"
    fi
}

# Main execution
case "${1:-help}" in
    clean-logs)
        clean_logs "$2"
        ;;
    clean-cache)
        clean_cache
        ;;
    backup-db)
        backup_database
        ;;
    optimize-db)
        optimize_database
        ;;
    vacuum-db)
        vacuum_database
        ;;
    analyze-db)
        analyze_database
        ;;
    check-integrity)
        check_integrity
        ;;
    export-backup)
        export_backup
        ;;
    clean-historical)
        clean_historical "$2"
        ;;
    check-health)
        check_health
        ;;
    reset-permissions)
        reset_permissions
        ;;
    update-deps)
        update_dependencies
        ;;
    show-stats)
        show_statistics
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}❌ Unknown command: $1${NC}"
        echo
        show_help
        exit 1
        ;;
esac