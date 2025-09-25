#!/bin/bash

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Define paths
PIPELINE_SCRIPT="$SCRIPT_DIR/run_pipeline.sh"
LOG_DIR="$SCRIPT_DIR/logs"
CRON_LOG="$LOG_DIR/cron.log"
LAUNCHD_LOG="$LOG_DIR/launchd.log"
MONITOR_LOG="$LOG_DIR/cron_monitor.log"

# LaunchAgent configuration
LAUNCHD_LABEL="com.stockpipeline.updater"
LAUNCHD_PLIST="$HOME/Library/LaunchAgents/${LAUNCHD_LABEL}.plist"

# Default schedule (6 PM weekdays)
DEFAULT_SCHEDULE="0 18 * * 1-5"

# Global variables for selections
SELECTED_SCHEDULER=""
SELECTED_SCHEDULE=""

# Detect OS
OS_TYPE=$(uname)
IS_MACOS=false
if [[ "$OS_TYPE" == "Darwin" ]]; then
    IS_MACOS=true
fi

# Color codes for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Emoji indicators
SUCCESS="✅"
ERROR="❌"
WARNING="⚠️"
INFO="ℹ️"
GEAR="⚙️"
CLOCK="⏰"
CALENDAR="📅"
ROBOT="🤖"
APPLE="🍎"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_DIR/install.log"
}

# Output functions
header() {
    echo
    echo -e "${WHITE}$1${NC}"
    echo -e "${WHITE}$(printf '=%.0s' $(seq 1 ${#1}))${NC}"
}

success() {
    echo -e "${GREEN}${SUCCESS} $1${NC}"
    log "SUCCESS: $1"
}

error() {
    echo -e "${RED}${ERROR} $1${NC}"
    log "ERROR: $1"
}

warning() {
    echo -e "${YELLOW}${WARNING} $1${NC}"
    log "WARNING: $1"
}

info() {
    echo -e "${BLUE}${INFO} $1${NC}"
    log "INFO: $1"
}

# Check if script exists
check_pipeline_script() {
    if [[ ! -f "$PIPELINE_SCRIPT" ]]; then
        error "Pipeline script not found: $PIPELINE_SCRIPT"
        echo "Please ensure run_pipeline.sh exists in the same directory as this installer."
        exit 1
    fi
    
    if [[ ! -x "$PIPELINE_SCRIPT" ]]; then
        info "Making pipeline script executable..."
        chmod +x "$PIPELINE_SCRIPT"
        success "Pipeline script is now executable"
    fi
}

# Select scheduler type (macOS only)
select_scheduler() {
    if [[ "$IS_MACOS" == true ]]; then
        header "${APPLE} MACOS SCHEDULER SELECTION"
        echo "macOS supports two scheduling methods:"
        echo
        echo -e "${CYAN}1) launchd (Recommended)${NC}"
        echo "   • Native macOS scheduler"
        echo "   • Better reliability"
        echo "   • Survives system updates"
        echo "   • Proper logging integration"
        echo
        echo -e "${CYAN}2) crontab (Traditional)${NC}"
        echo "   • Unix standard scheduler"
        echo "   • Cross-platform compatible"
        echo "   • May require additional permissions"
        echo "   • Less reliable on modern macOS"
        echo
        
        while true; do
            printf "Select scheduler (1 for launchd, 2 for crontab) [1]: "
            read scheduler_choice
            scheduler_choice=${scheduler_choice:-1}
            
            case $scheduler_choice in
                1)
                    SELECTED_SCHEDULER="launchd"
                    return 0
                    ;;
                2)
                    SELECTED_SCHEDULER="crontab"
                    return 0
                    ;;
                *)
                    error "Invalid choice. Please enter 1 or 2."
                    ;;
            esac
        done
    else
        SELECTED_SCHEDULER="crontab"
    fi
}

# Show schedule options
show_schedule_options() {
    header "${CALENDAR} SCHEDULE OPTIONS"
    echo -e "${CYAN}Select a schedule by entering its number (1-8):${NC}"
    echo
    echo "  1) Default (6 PM weekdays)      → Runs at 6:00 PM Mon-Fri"
    echo "  2) Market close (4 PM weekdays)  → Runs at 4:00 PM Mon-Fri"
    echo "  3) After hours (8 PM weekdays)   → Runs at 8:00 PM Mon-Fri"
    echo "  4) Early morning (6 AM weekdays) → Runs at 6:00 AM Mon-Fri"
    echo "  5) Lunch time (12 PM weekdays)   → Runs at 12:00 PM Mon-Fri"
    echo "  6) Daily (6 PM every day)        → Runs at 6:00 PM every day"
    echo "  7) Twice daily (9 AM, 6 PM)      → Runs at 9:00 AM and 6:00 PM Mon-Fri"
    echo "  8) Custom schedule               → Enter your own schedule"
    echo
    
    if [[ "$1" == "launchd" ]]; then
        echo -e "${CYAN}Note:${NC} launchd will convert these to proper plist format"
        echo
    fi
}

# Get custom schedule
get_custom_schedule() {
    local scheduler_type="$1"
    
    echo
    echo -e "${CYAN}Enter custom schedule:${NC}"
    
    if [[ "$scheduler_type" == "launchd" ]]; then
        echo "Format: HOUR MINUTE (24-hour format)"
        echo "Example: 14 30 (for 2:30 PM)"
        echo
        printf "Hour (0-23): "
        read hour
        printf "Minute (0-59): "
        read minute
        printf "Days (1=Mon-Fri, 2=Daily, 3=Weekends): "
        read days_choice
        
        # Validate inputs
        if ! [[ "$hour" =~ ^[0-9]+$ ]] || [ "$hour" -lt 0 ] || [ "$hour" -gt 23 ]; then
            error "Invalid hour: $hour"
            return 1
        fi
        
        if ! [[ "$minute" =~ ^[0-9]+$ ]] || [ "$minute" -lt 0 ] || [ "$minute" -gt 59 ]; then
            error "Invalid minute: $minute"
            return 1
        fi
        
        # Create cron expression based on days
        case $days_choice in
            1) echo "$minute $hour * * 1-5" ;;
            2) echo "$minute $hour * * *" ;;
            3) echo "$minute $hour * * 0,6" ;;
            *) error "Invalid days choice"; return 1 ;;
        esac
    else
        echo "Format: MIN HOUR DAY MON WEEKDAY"
        echo "Example: 0 18 * * 1-5 (6 PM weekdays)"
        echo "         */30 * * * * (every 30 minutes)"
        echo
        printf "Enter cron expression: "
        read cron_expr
        echo "$cron_expr"
    fi
}

# Get schedule choice
get_schedule_choice() {
    local scheduler_type="$1"
    local choice
    local selected_schedule
    
    show_schedule_options "$scheduler_type"
    
    while true; do
        echo -n "Enter your choice (1-8) or press Enter for default [1]: "
        read choice
        choice=$(echo "$choice" | xargs)  # Trim whitespace
        
        # Default to option 1 if empty
        if [[ -z "$choice" ]]; then
            choice="1"
        fi
        
        if [[ "$choice" =~ ^[1-8]$ ]]; then
            case $choice in
                1) selected_schedule="0 18 * * 1-5" ;;
                2) selected_schedule="0 16 * * 1-5" ;;
                3) selected_schedule="0 20 * * 1-5" ;;
                4) selected_schedule="0 6 * * 1-5" ;;
                5) selected_schedule="0 12 * * 1-5" ;;
                6) selected_schedule="0 18 * * *" ;;
                7) selected_schedule="0 9,18 * * 1-5" ;;
                8) 
                    selected_schedule=$(get_custom_schedule "$scheduler_type")
                    if [[ -z "$selected_schedule" ]]; then
                        error "Failed to get custom schedule"
                        continue
                    fi
                    ;;
            esac
            
            if [ -n "$selected_schedule" ]; then
                # Set the global variable instead of echoing
                SELECTED_SCHEDULE="$selected_schedule"
                break
            fi
        else
            error "Invalid choice: $choice"
        fi
    done
}

# Parse cron schedule for launchd
parse_cron_to_launchd() {
    local cron_schedule="$1"
    local minute hour day month weekday
    
    # Parse cron expression
    read minute hour day month weekday <<< "$cron_schedule"
    
    # Create calendar interval XML
    local calendar_xml=""
    
    # Handle different schedule patterns
    if [[ "$minute" != "*" ]] && [[ "$hour" != "*" ]]; then
        # Specific time
        if [[ "$weekday" == "1-5" ]]; then
            # Weekdays
            for d in 1 2 3 4 5; do
                calendar_xml="${calendar_xml}
    <dict>
        <key>Weekday</key>
        <integer>$d</integer>
        <key>Hour</key>
        <integer>$hour</integer>
        <key>Minute</key>
        <integer>$minute</integer>
    </dict>"
            done
        elif [[ "$weekday" == "*" ]]; then
            # Every day
            calendar_xml="
    <dict>
        <key>Hour</key>
        <integer>$hour</integer>
        <key>Minute</key>
        <integer>$minute</integer>
    </dict>"
        elif [[ "$weekday" == "0,6" ]]; then
            # Weekends
            calendar_xml="
    <dict>
        <key>Weekday</key>
        <integer>0</integer>
        <key>Hour</key>
        <integer>$hour</integer>
        <key>Minute</key>
        <integer>$minute</integer>
    </dict>
    <dict>
        <key>Weekday</key>
        <integer>6</integer>
        <key>Hour</key>
        <integer>$hour</integer>
        <key>Minute</key>
        <integer>$minute</integer>
    </dict>"
        elif [[ "$hour" == *","* ]]; then
            # Multiple times per day
            IFS=',' read -ra HOURS <<< "$hour"
            if [[ "$weekday" == "1-5" ]]; then
                for h in "${HOURS[@]}"; do
                    for d in 1 2 3 4 5; do
                        calendar_xml="${calendar_xml}
    <dict>
        <key>Weekday</key>
        <integer>$d</integer>
        <key>Hour</key>
        <integer>$h</integer>
        <key>Minute</key>
        <integer>$minute</integer>
    </dict>"
                    done
                done
            else
                for h in "${HOURS[@]}"; do
                    calendar_xml="${calendar_xml}
    <dict>
        <key>Hour</key>
        <integer>$h</integer>
        <key>Minute</key>
        <integer>$minute</integer>
    </dict>"
                done
            fi
        fi
    elif [[ "$minute" == "*/"* ]]; then
        # Interval schedule (e.g., */30 for every 30 minutes)
        local interval="${minute#*/}"
        calendar_xml="
    <dict>
        <key>Minute</key>
        <integer>$interval</integer>
    </dict>"
    fi
    
    echo "$calendar_xml"
}

# Create launchd plist
create_launchd_plist() {
    local schedule="$1"
    local calendar_intervals=$(parse_cron_to_launchd "$schedule")
    
    cat > "$LAUNCHD_PLIST" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LAUNCHD_LABEL}</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>${PIPELINE_SCRIPT}</string>
    </array>
    
    <key>StartCalendarInterval</key>
    <array>${calendar_intervals}
    </array>
    
    <key>WorkingDirectory</key>
    <string>${SCRIPT_DIR}</string>
    
    <key>StandardOutPath</key>
    <string>${LAUNCHD_LOG}</string>
    
    <key>StandardErrorPath</key>
    <string>${LAUNCHD_LOG}</string>
    
    <key>RunAtLoad</key>
    <false/>
    
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin</string>
    </dict>
</dict>
</plist>
EOF
}

# Install launchd job
install_launchd() {
    local schedule="$1"
    
    info "Installing launchd job..."
    
    # Unload existing job if it exists
    if launchctl list | grep -q "$LAUNCHD_LABEL"; then
        info "Removing existing launchd job..."
        launchctl unload "$LAUNCHD_PLIST" 2>/dev/null || true
        launchctl remove "$LAUNCHD_LABEL" 2>/dev/null || true
    fi
    
    # Create plist
    create_launchd_plist "$schedule"
    
    # Load the job
    launchctl load "$LAUNCHD_PLIST"
    
    # Verify installation
    if launchctl list | grep -q "$LAUNCHD_LABEL"; then
        success "LaunchAgent installed successfully!"
        echo
        info "Job label: $LAUNCHD_LABEL"
        info "Plist location: $LAUNCHD_PLIST"
        info "Logs will be written to: $LAUNCHD_LOG"
        echo
        echo -e "${CYAN}Useful commands:${NC}"
        echo "  View status:  launchctl list | grep $LAUNCHD_LABEL"
        echo "  View logs:    tail -f $LAUNCHD_LOG"
        echo "  Run now:      launchctl start $LAUNCHD_LABEL"
        echo "  Disable:      launchctl unload $LAUNCHD_PLIST"
        echo "  Re-enable:    launchctl load $LAUNCHD_PLIST"
        echo "  Remove:       launchctl remove $LAUNCHD_LABEL"
    else
        error "Failed to install LaunchAgent"
        exit 1
    fi
}

# Install crontab job
install_crontab() {
    local schedule="$1"
    
    info "Installing crontab entry..."
    
    # Create cron entry
    local cron_entry="$schedule cd $SCRIPT_DIR && $PIPELINE_SCRIPT >> $CRON_LOG 2>&1"
    
    # Check if entry already exists
    if crontab -l 2>/dev/null | grep -q "$PIPELINE_SCRIPT"; then
        warning "Existing crontab entry found. Updating..."
        # Remove old entry
        crontab -l 2>/dev/null | grep -v "$PIPELINE_SCRIPT" | crontab -
    fi
    
    # Add new entry
    (crontab -l 2>/dev/null; echo "$cron_entry") | crontab -
    
    # Verify installation
    if crontab -l 2>/dev/null | grep -q "$PIPELINE_SCRIPT"; then
        success "Crontab entry installed successfully!"
        echo
        info "Schedule: $schedule"
        info "Logs will be written to: $CRON_LOG"
        echo
        echo -e "${CYAN}Useful commands:${NC}"
        echo "  View crontab:  crontab -l"
        echo "  Edit crontab:  crontab -e"
        echo "  View logs:     tail -f $CRON_LOG"
        echo "  Remove entry:  crontab -l | grep -v '$PIPELINE_SCRIPT' | crontab -"
    else
        error "Failed to install crontab entry"
        exit 1
    fi
}

# Display schedule in human-readable format
display_schedule() {
    local schedule="$1"
    local minute hour day month weekday
    
    read minute hour day month weekday <<< "$schedule"
    
    echo -e "${CYAN}Schedule Summary:${NC}"
    
    # Parse the schedule
    if [[ "$minute" == "*/"* ]]; then
        echo "  Frequency: Every ${minute#*/} minutes"
    elif [[ "$hour" == *","* ]]; then
        IFS=',' read -ra HOURS <<< "$hour"
        echo -n "  Time: "
        for h in "${HOURS[@]}"; do
            printf "%02d:%02d " "$h" "$minute"
        done
        echo
    else
        printf "  Time: %02d:%02d\n" "$hour" "$minute"
    fi
    
    if [[ "$weekday" == "1-5" ]]; then
        echo "  Days: Monday through Friday"
    elif [[ "$weekday" == "*" ]]; then
        echo "  Days: Every day"
    elif [[ "$weekday" == "0,6" ]]; then
        echo "  Days: Weekends only"
    fi
    
    echo "  Cron expression: $schedule"
}

# Main installation function
main() {
    # Create log directory
    mkdir -p "$LOG_DIR"
    log "Installation started"
    
    # Display header
    header "${ROBOT} STOCK PIPELINE SCHEDULER INSTALLER v1.5.1"
    
    # Check for help flag
    if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
        echo "Usage: $0 [OPTIONS]"
        echo
        echo "Options:"
        echo "  --uninstall    Remove scheduled jobs"
        echo "  --status       Check scheduler status"
        echo "  --help         Show this help message"
        exit 0
    fi
    
    # Handle uninstall
    if [[ "${1:-}" == "--uninstall" ]]; then
        header "Uninstalling scheduled jobs..."
        
        if [[ "$IS_MACOS" == true ]] && launchctl list | grep -q "$LAUNCHD_LABEL"; then
            launchctl unload "$LAUNCHD_PLIST" 2>/dev/null || true
            launchctl remove "$LAUNCHD_LABEL" 2>/dev/null || true
            rm -f "$LAUNCHD_PLIST"
            success "LaunchAgent removed"
        fi
        
        if crontab -l 2>/dev/null | grep -q "$PIPELINE_SCRIPT"; then
            crontab -l | grep -v "$PIPELINE_SCRIPT" | crontab -
            success "Crontab entry removed"
        fi
        
        exit 0
    fi
    
    # Handle status check
    if [[ "${1:-}" == "--status" ]]; then
        header "Scheduler Status"
        
        if [[ "$IS_MACOS" == true ]]; then
            echo -e "${CYAN}LaunchAgent status:${NC}"
            if launchctl list | grep -q "$LAUNCHD_LABEL"; then
                launchctl list | grep "$LAUNCHD_LABEL"
                echo
                echo "Last run log:"
                tail -n 5 "$LAUNCHD_LOG" 2>/dev/null || echo "No logs yet"
            else
                echo "No LaunchAgent found"
            fi
            echo
        fi
        
        echo -e "${CYAN}Crontab status:${NC}"
        if crontab -l 2>/dev/null | grep -q "$PIPELINE_SCRIPT"; then
            crontab -l | grep "$PIPELINE_SCRIPT"
            echo
            echo "Last run log:"
            tail -n 5 "$CRON_LOG" 2>/dev/null || echo "No logs yet"
        else
            echo "No crontab entry found"
        fi
        
        exit 0
    fi
    
    # Check pipeline script exists
    check_pipeline_script
    
    # Select scheduler type
    local scheduler_type
    if [[ "$IS_MACOS" == true ]]; then
        select_scheduler
        scheduler_type="$SELECTED_SCHEDULER"
        success "Selected scheduler: $scheduler_type"
    else
        scheduler_type="crontab"
        info "Using crontab scheduler (Linux/Unix system detected)"
    fi
    
    # Remove existing installations if present
    if [[ "$scheduler_type" == "launchd" ]]; then
        if launchctl list 2>/dev/null | grep -q "$LAUNCHD_LABEL"; then
            warning "Existing LaunchAgent found. It will be replaced."
        fi
    else
        if crontab -l 2>/dev/null | grep -q "$PIPELINE_SCRIPT"; then
            warning "Existing crontab entry found. It will be replaced."
        fi
    fi
    
    echo
    info "Configure your stock pipeline schedule"
    echo
    
    # Get schedule choice - Fixed to use global variable
    get_schedule_choice "$scheduler_type"
    local schedule="$SELECTED_SCHEDULE"
    
    if [ -z "$schedule" ]; then
        error "No schedule selected"
        exit 1
    fi
    
    success "Schedule selected!"
    echo
    display_schedule "$schedule"
    echo
    
    # Confirm installation
    echo -e "${YELLOW}Ready to install the scheduled job.${NC}"
    printf "Do you want to proceed? (y/n) [y]: "
    read confirm
    confirm=${confirm:-y}
    
    if [[ "$confirm" != "y" ]] && [[ "$confirm" != "Y" ]]; then
        warning "Installation cancelled"
        exit 0
    fi
    
    # Install based on scheduler type
    if [[ "$scheduler_type" == "launchd" ]]; then
        install_launchd "$schedule"
    else
        install_crontab "$schedule"
    fi
    
    echo
    header "${SUCCESS} Installation Complete!"
    echo
    echo "The stock pipeline will run automatically according to your schedule."
    echo "You can check the logs to monitor execution."
    echo
    
    # Test run prompt
    printf "Would you like to test run the pipeline now? (y/n) [n]: "
    read test_run
    test_run=${test_run:-n}
    
    if [[ "$test_run" == "y" ]] || [[ "$test_run" == "Y" ]]; then
        echo
        info "Running pipeline test..."
        if [[ "$scheduler_type" == "launchd" ]]; then
            launchctl start "$LAUNCHD_LABEL"
            echo "Pipeline started via LaunchAgent. Check logs at: $LAUNCHD_LOG"
        else
            "$PIPELINE_SCRIPT"
        fi
    fi
    
    echo
    success "Setup complete! Your stock pipeline is now scheduled."
    log "Installation completed successfully"
}

# Handle script arguments
case "${1:-}" in
    --uninstall|--status|--help|-h)
        main "$@"
        ;;
    *)
        main "$@"
        ;;
esac