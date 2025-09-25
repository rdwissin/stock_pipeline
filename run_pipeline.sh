#!/bin/bash

# Enhanced Stock Data Pipeline Runner v1.5.1
# Complete signal handling for clean shutdown with Ctrl-C
# Version: 1.5.1
# Copyright 2025, Richard D. Wissinger
# Author: Richard D. Wissinger (rick.wissinger@gmail.com)

set -e

# ===============================================================================
# CONFIGURATION
# ===============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_CMD="python3"
VENV_DIR="$SCRIPT_DIR/venv"
LOG_FILE="$SCRIPT_DIR/logs/runner.log"
PID_FILE="$SCRIPT_DIR/pipeline.pid"
PERFORMANCE_LOG="$SCRIPT_DIR/logs/performance_runner.log"

# Store the PID of the Python process
PYTHON_PID=""
CLEANUP_DONE=false

# System detection
SYSTEM=$(uname -s)
ARCH=$(uname -m)
CPU_CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo "4")

# ===============================================================================
# COLORS AND FORMATTING
# ===============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Unicode symbols for better visual feedback
SUCCESS="✅"
ERROR="❌"
WARNING="⚠️"
INFO="ℹ️"
ROCKET="🚀"
GEAR="⚙️"
CHART="📊"
CLOCK="⏱️"

# ===============================================================================
# SIGNAL HANDLING
# ===============================================================================

cleanup_and_exit() {
    # Prevent multiple cleanup calls
    if [ "$CLEANUP_DONE" = true ]; then
        return
    fi
    CLEANUP_DONE=true
    
    echo
    echo -e "${YELLOW}${WARNING} Interrupt received, initiating clean shutdown...${NC}" | tee -a "$LOG_FILE"
    
    # Kill the Python process if it's running
    if [ -n "$PYTHON_PID" ] && kill -0 "$PYTHON_PID" 2>/dev/null; then
        echo "Sending SIGTERM to Python process (PID: $PYTHON_PID)..." | tee -a "$LOG_FILE"
        kill -TERM "$PYTHON_PID" 2>/dev/null || true
        
        # Wait up to 15 seconds for graceful shutdown
        echo "Waiting for Python process to terminate gracefully..." | tee -a "$LOG_FILE"
        for i in {1..15}; do
            if ! kill -0 "$PYTHON_PID" 2>/dev/null; then
                echo -e "${GREEN}Python process terminated gracefully${NC}" | tee -a "$LOG_FILE"
                break
            fi
            sleep 1
            echo -n "."
        done
        echo
        
        # Force kill if still running
        if kill -0 "$PYTHON_PID" 2>/dev/null; then
            echo -e "${RED}Force killing Python process...${NC}" | tee -a "$LOG_FILE"
            kill -KILL "$PYTHON_PID" 2>/dev/null || true
            sleep 1
        fi
    fi
    
    # Remove PID file
    if [ -f "$PID_FILE" ]; then
        rm -f "$PID_FILE"
        echo "Removed PID file" | tee -a "$LOG_FILE"
    fi
    
    echo -e "${GREEN}${SUCCESS} Clean shutdown completed${NC}" | tee -a "$LOG_FILE"
    exit 130
}

# Trap signals for clean shutdown
trap cleanup_and_exit SIGINT SIGTERM EXIT

# ===============================================================================
# LOGGING FUNCTIONS
# ===============================================================================

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_performance() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - PERF - $1" | tee -a "$PERFORMANCE_LOG"
}

success() {
    echo -e "${GREEN}${SUCCESS} $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}${ERROR} $1${NC}" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}${WARNING} $1${NC}" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}${INFO} $1${NC}" | tee -a "$LOG_FILE"
}

header() {
    echo -e "${WHITE}${1}${NC}"
    echo -e "${WHITE}$(printf '=%.0s' $(seq 1 ${#1}))${NC}"
}

# ===============================================================================
# SYSTEM INFORMATION
# ===============================================================================

get_system_info() {
    echo
    header "${ROCKET} ENHANCED STOCK PIPELINE v1.5.1 - SYSTEM INFO"
    
    info "System: $SYSTEM $ARCH"
    info "CPU Cores: $CPU_CORES"
    
    # Memory information
    if [[ "$SYSTEM" == "Darwin" ]]; then
        TOTAL_MEM=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
        info "Total Memory: ${TOTAL_MEM}GB"
        
        # Get available memory
        VM_STAT=$(vm_stat)
        FREE_PAGES=$(echo "$VM_STAT" | grep "Pages free" | awk '{print $3}' | sed 's/\.//')
        INACTIVE_PAGES=$(echo "$VM_STAT" | grep "Pages inactive" | awk '{print $3}' | sed 's/\.//')
        PAGE_SIZE=$(vm_stat | grep "page size" | awk '{print $8}')
        AVAILABLE_MB=$(((FREE_PAGES + INACTIVE_PAGES) * PAGE_SIZE / 1024 / 1024))
        info "Available Memory: ${AVAILABLE_MB}MB"
    fi
    
    # Disk space
    DISK_AVAIL=$(df -h "$SCRIPT_DIR" | awk 'NR==2 {print $4}')
    info "Available Disk Space: $DISK_AVAIL"
    
    # Current load
    LOAD_AVG=$(uptime | awk -F'load averages:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    info "System Load: $LOAD_AVG"
    
    echo
}

# ===============================================================================
# PYTHON ENVIRONMENT MANAGEMENT
# ===============================================================================

check_python() {
    header "${GEAR} PYTHON ENVIRONMENT CHECK"
    
    # Check if Python is available
    if ! command -v $PYTHON_CMD &> /dev/null; then
        warning "Python 3 not found. Checking for alternatives..."
        
        # Try alternatives
        for py_cmd in python3.13 python3.12 python3.11 python3; do
            if command -v $py_cmd &> /dev/null; then
                version=$($py_cmd --version 2>&1 | cut -d' ' -f2)
                major_minor=$(echo $version | cut -d'.' -f1-2)
                major=$(echo $major_minor | cut -d'.' -f1)
                minor=$(echo $major_minor | cut -d'.' -f2)
                
                if [[ $major -eq 3 ]] && [[ $minor -ge 11 ]]; then
                    PYTHON_CMD=$py_cmd
                    success "Found $py_cmd (version $version)"
                    break
                fi
            fi
        done
        
        if ! command -v $PYTHON_CMD &> /dev/null; then
            error "No suitable Python version found. Please install Python 3.11+"
            echo
            echo "To install Python 3.13 on macOS:"
            echo "  brew install python@3.13"
            echo
            exit 1
        fi
    fi
    
    local version=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    success "Using Python $version"
    
    # Check Python version compatibility
    local major=$(echo $version | cut -d'.' -f1)
    local minor=$(echo $version | cut -d'.' -f2)
    
    if [[ $major -lt 3 ]] || [[ $major -eq 3 && $minor -lt 11 ]]; then
        error "Python version $version is too old. Minimum required: 3.11"
        exit 1
    fi
    
    log_performance "Python check completed: $PYTHON_CMD version $version"
}

setup_virtual_environment() {
    header "${GEAR} VIRTUAL ENVIRONMENT SETUP"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        info "Creating virtual environment..."
        $PYTHON_CMD -m venv "$VENV_DIR"
        success "Virtual environment created"
    else
        info "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    success "Virtual environment activated"
    
    # Upgrade pip to latest version
    info "Checking pip..."
    pip install --upgrade pip --quiet 2>&1
    success "Pip is up to date ($(pip --version | cut -d' ' -f2))"
    
    # Install/upgrade requirements
    if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
        info "Installing/updating requirements..."
        pip install -r "$SCRIPT_DIR/requirements.txt" --quiet 2>&1
        success "Requirements installed"
        
        # Log installed packages for debugging
        pip list > "$SCRIPT_DIR/logs/installed_packages.txt" 2>/dev/null || true
    else
        warning "requirements.txt not found, skipping package installation"
    fi
    
    log_performance "Virtual environment setup completed"
}

# ===============================================================================
# PRE-FLIGHT CHECKS
# ===============================================================================

check_dependencies() {
    header "${GEAR} DEPENDENCY CHECKS"
    
    # Check for required files
    local required_files=("config.py" "main.py" "fetcher.py" "storage.py" "logger.py" "blacklist.py" "historical_fetcher.py")
    local missing_files=()
    
    for file in "${required_files[@]}"; do
        if [ -f "$SCRIPT_DIR/$file" ]; then
            success "$file found"
        else
            error "$file missing"
            missing_files+=($file)
        fi
    done
    
    if [ ${#missing_files[@]} -gt 0 ]; then
        error "Missing required files: ${missing_files[*]}"
        exit 1
    fi
    
    # Check .env file
    if [ -f "$SCRIPT_DIR/.env" ]; then
        success ".env configuration file found"
    else
        if [ -f "$SCRIPT_DIR/.env.example" ]; then
            warning ".env file not found, copying from .env.example"
            cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
            info "Please edit .env file to configure your settings"
        else
            warning ".env file not found (will use defaults)"
        fi
    fi
    
    # Check network connectivity
    info "Checking network connectivity..."
    if curl -s --head --request GET https://finance.yahoo.com --max-time 5 > /dev/null; then
        success "Yahoo Finance is reachable"
    else
        warning "Cannot reach Yahoo Finance - pipeline may fail"
    fi
    
    log_performance "Dependency checks completed"
}

check_system_resources() {
    header "${GEAR} SYSTEM RESOURCE CHECK"
    
    # Check available memory
    if [[ "$SYSTEM" == "Darwin" ]]; then
        VM_STAT=$(vm_stat)
        FREE_PAGES=$(echo "$VM_STAT" | grep "Pages free" | awk '{print $3}' | sed 's/\.//')
        INACTIVE_PAGES=$(echo "$VM_STAT" | grep "Pages inactive" | awk '{print $3}' | sed 's/\.//')
        PAGE_SIZE=$(vm_stat | grep "page size" | awk '{print $8}')
        AVAILABLE_MB=$(((FREE_PAGES + INACTIVE_PAGES) * PAGE_SIZE / 1024 / 1024))
        
        if [ "$AVAILABLE_MB" -lt 512 ]; then
            error "Low memory: ${AVAILABLE_MB}MB available (recommended: 1GB+)"
            echo "Continue anyway? (y/N): "
            read -r response
            if [[ ! "$response" =~ ^[Yy]$ ]]; then
                exit 1
            fi
        elif [ "$AVAILABLE_MB" -lt 1024 ]; then
            warning "Limited memory: ${AVAILABLE_MB}MB available (recommended: 1GB+)"
        else
            success "Memory check passed: ${AVAILABLE_MB}MB available"
        fi
    fi
    
    # Check disk space
    DISK_AVAIL_MB=$(df -m "$SCRIPT_DIR" | awk 'NR==2 {print $4}')
    if [ "$DISK_AVAIL_MB" -lt 100 ]; then
        error "Low disk space: ${DISK_AVAIL_MB}MB available (minimum: 100MB)"
        exit 1
    else
        success "Disk space check passed: ${DISK_AVAIL_MB}MB available"
    fi
    
    log_performance "System resource check completed"
}

# ===============================================================================
# PIPELINE EXECUTION
# ===============================================================================

run_pipeline() {
    header "${ROCKET} PIPELINE EXECUTION"
    
    local start_time=$(date +%s)
    local pipeline_args=("$@")
    
    info "Starting enhanced stock data pipeline v1.5.1..."
    info "Arguments: ${pipeline_args[*]}"
    info "Press Ctrl-C to stop gracefully"
    echo
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # Run the pipeline in background and capture PID
    log_performance "Pipeline execution started with args: ${pipeline_args[*]}"
    
    # Start the Python process in the background
    python main.py "${pipeline_args[@]}" &
    PYTHON_PID=$!
    
    # Save PID to file
    echo $PYTHON_PID > "$PID_FILE"
    log "Python process started with PID: $PYTHON_PID"
    
    # Wait for the Python process to complete
    wait $PYTHON_PID
    exit_code=$?
    
    # Clear PID since process completed normally
    PYTHON_PID=""
    rm -f "$PID_FILE"
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ]; then
        success "Pipeline completed successfully in ${duration}s"
        log_performance "Pipeline execution completed successfully in ${duration}s"
        
        # Show quick stats if database exists
        show_quick_stats
    elif [ $exit_code -eq 130 ]; then
        warning "Pipeline interrupted by user"
        log_performance "Pipeline interrupted after ${duration}s"
    else
        error "Pipeline failed after ${duration}s (exit code: $exit_code)"
        log_performance "Pipeline execution failed after ${duration}s (exit code: $exit_code)"
        
        # Show error analysis
        show_error_analysis
    fi
    
    return $exit_code
}

show_quick_stats() {
    echo
    header "${CHART} QUICK STATISTICS"
    
    local db_path="$SCRIPT_DIR/data/stocks_enhanced.db"
    if [ -f "$db_path" ] && command -v sqlite3 &> /dev/null; then
        echo -e "${CYAN}Database Statistics:${NC}"
        
        # Latest snapshot date
        latest_date=$(sqlite3 "$db_path" "SELECT MAX(snapshot_date) FROM stocks;" 2>/dev/null || echo "unknown")
        echo "  Latest snapshot: $latest_date"
        
        # Total stocks
        if [ "$latest_date" != "unknown" ]; then
            total_stocks=$(sqlite3 "$db_path" "SELECT COUNT(*) FROM stocks WHERE snapshot_date='$latest_date';" 2>/dev/null || echo "unknown")
            echo "  Total stocks: $total_stocks"
            
            # Historical records
            hist_count=$(sqlite3 "$db_path" "SELECT COUNT(*) FROM historical_prices;" 2>/dev/null || echo "unknown")
            echo "  Historical records: $hist_count"
            
            # Blacklisted symbols
            blacklist_count=$(sqlite3 "$db_path" "SELECT COUNT(*) FROM stocks WHERE is_blacklisted=1 AND snapshot_date='$latest_date';" 2>/dev/null || echo "0")
            echo "  Blacklisted symbols: $blacklist_count"
        fi
        
        # Database size
        db_size=$(ls -lh "$db_path" | awk '{print $5}')
        echo "  Database size: $db_size"
    else
        warning "Database not found or sqlite3 not available"
    fi
    
    echo
}

show_error_analysis() {
    echo
    header "${ERROR} ERROR ANALYSIS"
    
    # Check recent error logs
    local error_log="$SCRIPT_DIR/logs/pipeline_$(date +%Y%m).log"
    if [ -f "$error_log" ]; then
        echo -e "${RED}Recent errors:${NC}"
        tail -20 "$error_log" | grep -E "(ERROR|CRITICAL)" | tail -5
        echo
    fi
    
    echo -e "${BLUE}Troubleshooting tips:${NC}"
    echo "  1. Check logs in: $SCRIPT_DIR/logs/"
    echo "  2. Try running with --debug flag for more details"
    echo "  3. Verify network connectivity to Yahoo Finance"
    echo "  4. Check system resources (memory, disk space)"
    echo "  5. Consider running with --sequential to reduce resource usage"
    echo
}

# ===============================================================================
# MAIN EXECUTION
# ===============================================================================

main() {
    # Create logs directory
    mkdir -p "$SCRIPT_DIR/logs"
    
    # Initialize
    log "Enhanced stock pipeline runner v1.5.1 started"
    log_performance "Runner initialization"
    
    # System information
    get_system_info
    
    # Check Python environment
    check_python
    
    # Setup virtual environment
    setup_virtual_environment
    
    # Pre-flight checks
    check_dependencies
    check_system_resources
    
    # Run the pipeline
    run_pipeline "$@"
    local exit_code=$?
    
    log "Pipeline runner completed with exit code: $exit_code"
    log_performance "Runner session completed"
    
    # Cleanup will be handled by trap
    CLEANUP_DONE=true
    exit $exit_code
}

# ===============================================================================
# HELP DISPLAY
# ===============================================================================

show_help() {
    echo -e "${WHITE}Enhanced Stock Data Pipeline Runner v1.5.1${NC}"
    echo
    echo -e "${CYAN}USAGE:${NC}"
    echo "  $0 [OPTIONS]"
    echo
    echo -e "${CYAN}OPTIONS:${NC}"
    echo "  --force         Force run even if data exists for today"
    echo "  --debug         Enable debug logging"
    echo "  --trace         Enable trace logging (very verbose)"
    echo "  --sequential    Disable parallel processing"
    echo "  --full-history  Fetch complete historical data (all available)"
    echo "  --no-history    Skip historical data fetch"
    echo "  --help          Show this help message"
    echo
    echo -e "${CYAN}EXAMPLES:${NC}"
    echo "  $0                           # Normal run"
    echo "  $0 --force                   # Force run"
    echo "  $0 --force --full-history    # Force run with complete history"
    echo "  $0 --debug                   # Run with debug logging"
    echo "  $0 --sequential              # Run without parallel processing"
    echo
    echo -e "${CYAN}FEATURES:${NC}"
    echo "  • Yahoo Finance primary data source (no API key required)"
    echo "  • Fetches MAXIMUM available historical data for all stocks"
    echo "  • Clean shutdown with Ctrl-C (saves progress)"
    echo "  • Automatic blacklist management for invalid symbols"
    echo "  • Comprehensive logging and monitoring"
    echo
    echo -e "${CYAN}SIGNAL HANDLING:${NC}"
    echo "  Press Ctrl-C to initiate clean shutdown"
    echo "  The pipeline will save progress and exit gracefully"
    echo
}

# Check for help flag
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    show_help
    exit 0
fi

# Run main function with all arguments
main "$@"