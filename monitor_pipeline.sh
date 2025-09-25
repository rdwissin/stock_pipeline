#!/bin/bash

# Stock Pipeline Monitoring Script v1.5.0
# Real-time monitoring and status reporting for the stock data pipeline

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRON_LOG="$SCRIPT_DIR/logs/cron.log"
PIPELINE_LOG="$SCRIPT_DIR/logs/pipeline_$(date +%Y%m).log"
DB_PATH="$SCRIPT_DIR/data/stocks_enhanced.db"
BLACKLIST_FILE="$SCRIPT_DIR/data/symbol_blacklist.json"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${BLUE}📊 Stock Pipeline v1.5.0 - Status Report${NC}"
echo -e "${BLUE}$(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo "========================================"

# Function to format numbers with commas
format_number() {
    echo "$1" | sed ':a;s/\B[0-9]\{3\}\>/,&/;ta'
}

# Check last run
echo
echo -e "${CYAN}📅 Last Pipeline Execution:${NC}"
if [ -f "$CRON_LOG" ]; then
    last_run=$(tail -5 "$CRON_LOG" | grep "Pipeline completed\|Pipeline runner started" | tail -1)
    if [ -n "$last_run" ]; then
        echo "  $last_run"
    else
        echo "  No recent execution found in cron log"
    fi
    
    # Check for recent errors
    recent_errors=$(tail -100 "$CRON_LOG" 2>/dev/null | grep -i error | wc -l)
    if [ "$recent_errors" -gt 0 ]; then
        echo -e "${RED}  ⚠️ Found $recent_errors recent errors in cron log${NC}"
    else
        echo -e "${GREEN}  ✅ No recent errors in cron log${NC}"
    fi
else
    echo "  Cron log not found (pipeline may not have run yet)"
fi

# Check database
echo
echo -e "${CYAN}💾 Database Status:${NC}"
if [ -f "$DB_PATH" ]; then
    if command -v sqlite3 &> /dev/null; then
        # Get latest snapshot date
        latest_date=$(sqlite3 "$DB_PATH" "SELECT MAX(snapshot_date) FROM stocks;" 2>/dev/null || echo "unknown")
        
        # Get stock counts
        if [ "$latest_date" != "unknown" ] && [ -n "$latest_date" ]; then
            stock_count=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM stocks WHERE snapshot_date='$latest_date';" 2>/dev/null || echo "0")
            blacklisted_count=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM stocks WHERE snapshot_date='$latest_date' AND is_blacklisted=1;" 2>/dev/null || echo "0")
            
            echo "  Latest data: $latest_date"
            echo "  Active stocks: $(format_number $stock_count)"
            echo "  Blacklisted: $(format_number $blacklisted_count)"
            
            # Check data freshness
            today=$(date +%Y-%m-%d)
            yesterday=$(date -d "yesterday" +%Y-%m-%d 2>/dev/null || date -v-1d +%Y-%m-%d 2>/dev/null || echo "unknown")
            
            if [ "$latest_date" = "$today" ]; then
                echo -e "${GREEN}  ✅ Data is current (today)${NC}"
            elif [ "$latest_date" = "$yesterday" ]; then
                echo -e "${YELLOW}  ⚠️ Data is from yesterday${NC}"
            else
                days_old=$(( ($(date +%s) - $(date -d "$latest_date" +%s 2>/dev/null || echo 0)) / 86400 ))
                if [ "$days_old" -gt 1 ]; then
                    echo -e "${RED}  ❌ Data is $days_old days old${NC}"
                fi
            fi
        fi
        
        # Historical data stats (10-year data)
        echo
        echo -e "${CYAN}📈 Historical Data (10-Year Coverage):${NC}"
        hist_count=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM historical_prices;" 2>/dev/null || echo "0")
        hist_symbols=$(sqlite3 "$DB_PATH" "SELECT COUNT(DISTINCT symbol) FROM historical_prices;" 2>/dev/null || echo "0")
        
        if [ "$hist_count" -gt 0 ]; then
            avg_records=$((hist_count / hist_symbols))
            years_coverage=$(sqlite3 "$DB_PATH" "
                SELECT printf('%.1f', (julianday(MAX(date)) - julianday(MIN(date))) / 365.25) 
                FROM historical_prices;
            " 2>/dev/null || echo "0")
            
            echo "  Total records: $(format_number $hist_count)"
            echo "  Symbols with history: $(format_number $hist_symbols)"
            echo "  Average records/symbol: $(format_number $avg_records)"
            echo "  Data coverage: ${years_coverage} years"
        else
            echo -e "${YELLOW}  No historical data yet${NC}"
        fi
        
        # Database size
        if [[ "$(uname)" == "Darwin" ]]; then
            db_size=$(ls -lh "$DB_PATH" | awk '{print $5}')
            echo
            echo -e "${CYAN}💽 Storage:${NC}"
            echo "  Database size: $db_size"
            
            # Check if size is concerning (>5GB for 10-year data)
            db_size_bytes=$(stat -f%z "$DB_PATH" 2>/dev/null || stat -c%s "$DB_PATH" 2>/dev/null || echo 0)
            db_size_gb=$((db_size_bytes / 1073741824))
            if [ "$db_size_gb" -gt 5 ]; then
                echo -e "${YELLOW}  ⚠️ Database is large (>5GB). Consider maintenance.${NC}"
            fi
        fi
        
        # Market overview
        echo
        echo -e "${CYAN}📊 Market Overview:${NC}"
        if [ "$latest_date" != "unknown" ] && [ -n "$latest_date" ]; then
            # Exchange breakdown
            exchange_stats=$(sqlite3 "$DB_PATH" "
                SELECT exchange, COUNT(*) as count 
                FROM stocks 
                WHERE snapshot_date='$latest_date' AND is_blacklisted=0
                GROUP BY exchange 
                ORDER BY count DESC
                LIMIT 5;
            " 2>/dev/null)
            
            if [ -n "$exchange_stats" ]; then
                echo "$exchange_stats" | while IFS='|' read -r exchange count; do
                    echo "  $exchange: $(format_number $count) stocks"
                done
            fi
            
            # Total market cap
            total_cap=$(sqlite3 "$DB_PATH" "
                SELECT printf('%.2f', SUM(market_cap)/1e12) 
                FROM stocks 
                WHERE snapshot_date='$latest_date' AND market_cap > 0;
            " 2>/dev/null || echo "0")
            
            if [ "$total_cap" != "0" ]; then
                echo "  Total Market Cap: \$${total_cap}T"
            fi
        fi
    else
        echo "  SQLite not available for detailed check"
        if [ -f "$DB_PATH" ]; then
            db_size=$(ls -lh "$DB_PATH" | awk '{print $5}')
            echo -e "${GREEN}  ✅ Database file exists (${db_size})${NC}"
        fi
    fi
else
    echo -e "${RED}  ❌ Database not found${NC}"
    echo "  Run './run_pipeline.sh --force' to initialize"
fi

# Check blacklist
echo
echo -e "${CYAN}🚫 Blacklist Status:${NC}"
if [ -f "$BLACKLIST_FILE" ]; then
    if command -v python3 &> /dev/null; then
        blacklist_count=$(python3 -c "
import json
with open('$BLACKLIST_FILE') as f:
    data = json.load(f)
    print(len(data.get('symbols', {})))
" 2>/dev/null || echo "0")
        echo "  Blacklisted symbols: $(format_number $blacklist_count)"
    else
        echo "  Blacklist file exists"
    fi
else
    echo "  No blacklist file (will be created automatically)"
fi

# Check pipeline logs
echo
echo -e "${CYAN}📝 Recent Pipeline Activity:${NC}"
if [ -f "$PIPELINE_LOG" ]; then
    # Check for errors
    pipeline_errors=$(tail -500 "$PIPELINE_LOG" 2>/dev/null | grep -E "(ERROR|CRITICAL)" | wc -l)
    pipeline_warnings=$(tail -500 "$PIPELINE_LOG" 2>/dev/null | grep "WARNING" | wc -l)
    
    if [ "$pipeline_errors" -gt 0 ]; then
        echo -e "${RED}  ❌ Found $pipeline_errors errors in recent logs${NC}"
        echo "  Recent errors:"
        tail -500 "$PIPELINE_LOG" | grep -E "(ERROR|CRITICAL)" | tail -3 | while read line; do
            echo "    ${line:0:100}..."
        done
    elif [ "$pipeline_warnings" -gt 0 ]; then
        echo -e "${YELLOW}  ⚠️ Found $pipeline_warnings warnings (no errors)${NC}"
    else
        echo -e "${GREEN}  ✅ No recent errors or warnings${NC}"
    fi
    
    # Show last completion
    last_complete=$(grep "PIPELINE EXECUTION SUMMARY\|Pipeline completed" "$PIPELINE_LOG" 2>/dev/null | tail -1)
    if [ -n "$last_complete" ]; then
        echo "  Last completion: ${last_complete:0:80}..."
    fi
else
    echo "  No pipeline log for current month"
fi

# Check system resources
echo
echo -e "${CYAN}⚙️ System Resources:${NC}"
if [[ "$(uname)" == "Darwin" ]]; then
    # Memory
    vm_stat_output=$(vm_stat)
    free_pages=$(echo "$vm_stat_output" | grep "Pages free" | awk '{print $3}' | sed 's/\.//')
    inactive_pages=$(echo "$vm_stat_output" | grep "Pages inactive" | awk '{print $3}' | sed 's/\.//')
    page_size=$(vm_stat | grep "page size" | awk '{print $8}')
    available_mb=$(((free_pages + inactive_pages) * page_size / 1024 / 1024))
    echo "  Available memory: $(format_number ${available_mb})MB"
    
    # Check if memory is low
    if [ "$available_mb" -lt 1000 ]; then
        echo -e "${YELLOW}  ⚠️ Low memory available (<1GB)${NC}"
    fi
    
    # CPU Load
    load_avg=$(uptime | awk -F'load averages:' '{print $2}' | awk '{print $1, $2, $3}')
    echo "  System load: $load_avg"
    
    # Disk space
    disk_usage=$(df -h "$SCRIPT_DIR" | awk 'NR==2 {print $5}' | sed 's/%//')
    disk_avail=$(df -h "$SCRIPT_DIR" | awk 'NR==2 {print $4}')
    echo "  Disk usage: ${disk_usage}% (${disk_avail} available)"
    
    # Check if disk is getting full
    if [ "$disk_usage" -gt 90 ]; then
        echo -e "${RED}  ❌ Disk space critical (>90% used)${NC}"
    elif [ "$disk_usage" -gt 80 ]; then
        echo -e "${YELLOW}  ⚠️ Disk space warning (>80% used)${NC}"
    fi
fi

# Check if pipeline is currently running
echo
echo -e "${CYAN}🏃 Process Status:${NC}"
if pgrep -f "python.*main.py" > /dev/null; then
    echo -e "${GREEN}  ✅ Pipeline is currently running${NC}"
    pid=$(pgrep -f "python.*main.py" | head -1)
    echo "  PID: $pid"
    
    # Show process info if available
    if command -v ps &> /dev/null; then
        ps_info=$(ps -p $pid -o %cpu,%mem,etime 2>/dev/null | tail -1)
        if [ -n "$ps_info" ]; then
            echo "  CPU/MEM/Time: $ps_info"
        fi
    fi
elif pgrep -f "historical_fetcher.py" > /dev/null; then
    echo -e "${GREEN}  ✅ Historical fetcher is running${NC}"
else
    echo "  No pipeline processes running"
fi

# Next scheduled run (if cron is set up)
echo
echo -e "${CYAN}⏰ Scheduled Runs:${NC}"
if command -v crontab &> /dev/null; then
    cron_job=$(crontab -l 2>/dev/null | grep "run_pipeline.sh" | head -1)
    if [ -n "$cron_job" ]; then
        schedule=$(echo "$cron_job" | awk '{print $1, $2, $3, $4, $5}')
        echo "  Schedule: $schedule"
        echo "  Command: run_pipeline.sh"
    else
        echo "  No scheduled runs configured"
        echo "  Run './install_cron.sh' to set up automatic updates"
    fi
fi

# Recommendations
echo
echo -e "${CYAN}💡 Recommendations:${NC}"

# Check if data is stale
if [ "$latest_date" != "unknown" ] && [ -n "$latest_date" ]; then
    days_old=$(( ($(date +%s) - $(date -d "$latest_date" +%s 2>/dev/null || echo 0)) / 86400 ))
    if [ "$days_old" -gt 1 ]; then
        echo "  • Run './run_pipeline.sh' to update stock data"
    fi
fi

# Check if historical data needs updating
if [ "$hist_count" = "0" ]; then
    echo "  • Run './run_pipeline.sh --full-history' to fetch 10-year historical data"
elif [ "$hist_symbols" -lt 1000 ]; then
    echo "  • Historical data incomplete. Run with --full-history"
fi

# Check if database needs optimization
if [ -f "$DB_PATH" ]; then
    db_size_bytes=$(stat -f%z "$DB_PATH" 2>/dev/null || stat -c%s "$DB_PATH" 2>/dev/null || echo 0)
    db_size_gb=$((db_size_bytes / 1073741824))
    if [ "$db_size_gb" -gt 3 ]; then
        echo "  • Run './maintain_pipeline.sh optimize-db' to optimize database"
    fi
fi

# Check if logs need cleaning
if [ -d "$SCRIPT_DIR/logs" ]; then
    log_count=$(find "$SCRIPT_DIR/logs" -name "*.log" -mtime +30 | wc -l)
    if [ "$log_count" -gt 10 ]; then
        echo "  • Run './maintain_pipeline.sh clean-logs 30' to clean old logs"
    fi
fi

echo
echo "========================================"
echo "For detailed logs, check:"
echo "  • Cron log: $CRON_LOG"
echo "  • Pipeline log: $PIPELINE_LOG"
echo "  • Error logs: $SCRIPT_DIR/logs/errors_*.log"
echo
echo "For maintenance tasks, run:"
echo "  ./maintain_pipeline.sh help"
echo