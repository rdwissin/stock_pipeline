#!/usr/bin/env python3
"""
Enhanced Stock Pipeline - Automated Setup Script with Technical Analysis
Comprehensive setup and configuration tool for the stock data pipeline
"""

__version__ = "1.5.0"
__copyright__ = "Copyright 2025, Richard D. Wissinger"
__author__ = "Richard D. Wissinger"
__email__ = "rick.wissinger@gmail.com"
__license__ = "MIT"  # or your chosen license
__status__ = "Production"

import os
import sys
import subprocess
import platform
import shutil
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import urllib.request
import tempfile

# Check and install psutil if needed
try:
    import psutil
except ImportError:
    print("Installing psutil for system monitoring...")
    subprocess.run([sys.executable, "-m", "pip", "install", "psutil"], check=True)
    import psutil

class SetupManager:
    """Comprehensive setup manager for the stock pipeline with technical analysis"""
    
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.python_cmd = self._detect_python()
        self.system_info = self._get_system_info()
        
    def _detect_python(self) -> str:
        """Detect the best Python version to use"""
        # Preferred Python versions in order
        python_candidates = [
            'python3.13', 'python3.12', 'python3.11', 'python3', 'python'
        ]
        
        for cmd in python_candidates:
            try:
                result = subprocess.run([cmd, '--version'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    version = result.stdout.strip().split()[-1]
                    major, minor = map(int, version.split('.')[:2])
                    if major == 3 and minor >= 11:
                        print(f"✅ Found suitable Python: {cmd} (version {version})")
                        return cmd
            except FileNotFoundError:
                continue
        
        print("❌ No suitable Python version found (requires 3.11+)")
        print("Please install Python 3.11+ and try again.")
        print("\nInstallation instructions:")
        print("  macOS: brew install python@3.13")
        print("  Ubuntu: apt-get install python3.13")
        print("  Windows: Download from python.org")
        sys.exit(1)
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            'platform': platform.system(),
            'platform_version': platform.release(),
            'architecture': platform.machine(),
            'python_version': platform.python_version(),
            'cpu_cores': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 1),
            'disk_free_gb': round(psutil.disk_usage('/').free / (1024**3), 1)
        }
    
    def display_welcome(self):
        """Display welcome message and system info"""
        print("=" * 70)
        print("🚀 ENHANCED STOCK PIPELINE WITH TECHNICAL ANALYSIS - SETUP")
        print("=" * 70)
        print()
        print("This setup script will configure your stock data pipeline with:")
        print("  • Yahoo Finance data source (no API key required)")
        print("  • Technical analysis with RSI, ROE, EPS growth")
        print("  • Pattern detection (Three-day step-up, Cup & Handle)")
        print("  • BUY/SELL/HOLD recommendations")
        print("  • Parallel processing for maximum performance")
        print("  • Comprehensive logging and monitoring")
        print("  • Automated scheduling and maintenance")
        print()
        
        print("📊 SYSTEM INFORMATION:")
        print(f"  Platform: {self.system_info['platform']} {self.system_info['platform_version']}")
        print(f"  Architecture: {self.system_info['architecture']}")
        print(f"  Python: {self.system_info['python_version']}")
        print(f"  CPU Cores: {self.system_info['cpu_cores']}")
        print(f"  Memory: {self.system_info['memory_gb']}GB")
        print(f"  Free Disk: {self.system_info['disk_free_gb']}GB")
        print()
        
        # Check system requirements
        self._check_system_requirements()
    
    def _check_system_requirements(self):
        """Check if system meets minimum requirements"""
        issues = []
        
        # Check Python version
        major, minor = map(int, self.system_info['python_version'].split('.')[:2])
        if major < 3 or (major == 3 and minor < 11):
            issues.append("Python 3.11+ required")
        
        # Check memory
        if self.system_info['memory_gb'] < 1:
            issues.append("At least 1GB RAM recommended")
        
        # Check disk space
        if self.system_info['disk_free_gb'] < 0.5:
            issues.append("At least 500MB free disk space required")
        
        if issues:
            print("⚠️  SYSTEM REQUIREMENTS ISSUES:")
            for issue in issues:
                print(f"  • {issue}")
            print()
            
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("Setup cancelled.")
                sys.exit(1)
        else:
            print("✅ System requirements check passed")
            print()
    
    def setup_virtual_environment(self):
        """Setup Python virtual environment"""
        print("🔧 SETTING UP VIRTUAL ENVIRONMENT")
        print("-" * 40)
        
        venv_dir = self.project_dir / "venv"
        
        if venv_dir.exists():
            print("  Virtual environment already exists")
            response = input("  Recreate virtual environment? (y/N): ")
            if response.lower() == 'y':
                print("  Removing existing virtual environment...")
                shutil.rmtree(venv_dir)
            else:
                print("  Using existing virtual environment")
                self._install_dependencies(venv_dir)
                return str(venv_dir)
        
        print("  Creating virtual environment...")
        result = subprocess.run([
            self.python_cmd, '-m', 'venv', str(venv_dir)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Failed to create virtual environment: {result.stderr}")
            sys.exit(1)
        
        print("✅ Virtual environment created")
        
        # Install dependencies
        self._install_dependencies(venv_dir)
        
        return str(venv_dir)
    
    def _install_dependencies(self, venv_dir: Path):
        """Install Python dependencies including technical analysis requirements"""
        print("  Installing dependencies...")
        
        # Get pip path
        if platform.system() == "Windows":
            pip_path = venv_dir / "Scripts" / "pip"
        else:
            pip_path = venv_dir / "bin" / "pip"
        
        # Upgrade pip
        subprocess.run([str(pip_path), 'install', '--upgrade', 'pip'], 
                      capture_output=True)
        
        # Create enhanced requirements.txt if needed
        self._create_requirements_file()
        
        # Install requirements
        requirements_file = self.project_dir / "requirements.txt"
        if requirements_file.exists():
            result = subprocess.run([
                str(pip_path), 'install', '-r', str(requirements_file)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Failed to install dependencies: {result.stderr}")
                print("  Trying to install core dependencies individually...")
                
                # Try installing core dependencies one by one
                core_deps = [
                    'requests', 'yfinance', 'pandas', 'numpy', 'psutil', 
                    'python-dotenv', 'matplotlib', 'tabulate', 'openpyxl', 'tqdm'
                ]
                for dep in core_deps:
                    subprocess.run([str(pip_path), 'install', dep], capture_output=True)
            
            print("✅ Dependencies installed successfully")
        else:
            print("⚠️  requirements.txt not found, skipping dependency installation")
    
    def _create_requirements_file(self):
        """Create enhanced requirements.txt if it doesn't exist or update existing"""
        requirements_file = self.project_dir / "requirements.txt"
        
        content = """# Enhanced Stock Pipeline Requirements - Version 1.3.0
# Copyright 2025, Richard D. Wissinger

# Core HTTP and networking
requests>=2.31.0
aiohttp>=3.9.0
urllib3>=2.0.0

# Yahoo Finance integration - ESSENTIAL
yfinance>=0.2.40
appdirs>=1.4.4
beautifulsoup4>=4.12.0
frozendict>=2.3.0
html5lib>=1.1
lxml>=4.9.0
multitasking>=0.0.11
peewee>=3.17.0
pytz>=2024.0

# Data processing and analysis - REQUIRED
pandas>=2.1.0
numpy>=1.24.0
scipy>=1.11.0

# Technical Analysis - NEW
matplotlib>=3.8.0
tabulate>=0.9.0
tqdm>=4.66.0

# Environment and configuration
python-dotenv>=1.0.0

# Email functionality
email-validator>=2.0.0

# System monitoring and performance
psutil>=5.9.0

# Excel support
openpyxl>=3.1.0
xlsxwriter>=3.1.0

# Optional: Advanced analysis tools
scikit-learn>=1.3.0
statsmodels>=0.14.0
plotly>=5.17.0
seaborn>=0.12.0
"""
        requirements_file.write_text(content)
        print("  ✅ Updated requirements.txt with technical analysis dependencies")
    
    def configure_environment(self):
        """Configure environment variables and settings"""
        print("\n🔧 CONFIGURING ENVIRONMENT")
        print("-" * 30)
        
        env_file = self.project_dir / ".env"
        
        if env_file.exists():
            print("  .env file already exists")
            response = input("  Update with technical analysis settings? (Y/n): ")
            if response.lower() != 'n':
                self._update_env_for_technical_analysis(env_file)
                print("  ✅ Added technical analysis settings to .env")
            return
        
        print("  Creating comprehensive .env configuration...")
        self._create_comprehensive_env_config(env_file)
        print("✅ Environment configured")
    
    def _create_comprehensive_env_config(self, env_file: Path):
        """Create a comprehensive environment configuration with technical analysis"""
        # Calculate optimal settings based on system
        optimal_workers = min(8, max(2, self.system_info['cpu_cores'] - 2))
        optimal_memory = min(4096, max(512, int(self.system_info['memory_gb'] * 1024 * 0.5)))
        
        config = f"""# Enhanced Stock Pipeline Configuration with Technical Analysis
# Auto-generated by setup script v{__version__}

# ===============================================================================
# PERFORMANCE SETTINGS
# ===============================================================================
MAX_WORKERS={optimal_workers}
MAX_CONCURRENT_REQUESTS=50
BATCH_SIZE=100
MAX_MEMORY_MB={optimal_memory}
CHUNK_SIZE=500

# ===============================================================================
# REQUEST CONFIGURATION
# ===============================================================================
MAX_RETRIES=3
RETRY_DELAY=1.0
REQUEST_TIMEOUT=30
CONNECTION_TIMEOUT=10
REQUESTS_PER_MINUTE=60

# ===============================================================================
# LOGGING
# ===============================================================================
LOG_LEVEL=INFO
ENABLE_PERFORMANCE_LOGGING=true
ENABLE_MEMORY_TRACKING=true

# ===============================================================================
# DATA COLLECTION
# ===============================================================================
COLLECT_HISTORICAL_PRICES=true
COLLECT_FINANCIAL_RATIOS=true
HISTORICAL_DAYS=30

# ===============================================================================
# DATA QUALITY AND FILTERING
# ===============================================================================
MIN_MARKET_CAP=0
EXCLUDE_OTCBB=true
EXCLUDE_PINK_SHEETS=true

# ===============================================================================
# TECHNICAL ANALYSIS SETTINGS
# ===============================================================================
RUN_TECHNICAL_ANALYSIS=true

# Analysis Period
ANALYSIS_PERIOD=1y  # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
ANALYSIS_ASOF=  # Leave blank for latest, or use YYYY-MM-DD

# RSI Settings
RSI_WINDOW=14
RSI_MIN=40.0
RSI_MAX=70.0

# Fundamental Screening
ROE_MIN=10.0
EPS_YOY_MIN=0.0

# Pattern Detection
DETECT_THREE_UP=true
DETECT_CUP_HANDLE=true

# Cup & Handle Parameters
CUP_LEN_MIN=30
CUP_LEN_MAX=200
CUP_DEPTH_MIN=12.0
CUP_DEPTH_MAX=50.0
HANDLE_LEN_MIN=5
HANDLE_LEN_MAX=30
HANDLE_DEPTH_MAX=15.0
RIM_TOLERANCE=0.08
HANDLE_END_BAND=0.05

# Analysis Output
ANALYSIS_OUTPUT_FORMATS=csv,excel,json
ANALYSIS_OUTPUT_DIR=analysis_reports
ANALYSIS_CHART_MATCHES=true
ANALYSIS_CHART_DIR=analysis_charts

# Analysis Ticker Lists
ANALYSIS_WATCHLIST=data/watchlists/watchlist.txt
ANALYSIS_TOP_PERFORMERS=data/watchlists/top_performers.txt
ANALYSIS_SECTORS=data/watchlists/sectors.txt

# ===============================================================================
# SYSTEM OPTIMIZATION
# ===============================================================================
ENABLE_COMPRESSION=true
AUTO_VACUUM_DB=true
ENABLE_CACHE=true
CACHE_DURATION_HOURS=6
MAX_CACHE_SIZE_MB=100

# ===============================================================================
# EMAIL ALERTS (Optional)
# ===============================================================================
# SMTP_SERVER=smtp.gmail.com
# SMTP_PORT=587
# EMAIL_USER=
# EMAIL_PASS=
# ALERT_RECIPIENTS=
"""
        
        with open(env_file, 'w') as f:
            f.write(config)
    
    def _update_env_for_technical_analysis(self, env_file: Path):
        """Add technical analysis settings to existing .env file"""
        with open(env_file, 'r') as f:
            content = f.read()
        
        if 'TECHNICAL ANALYSIS SETTINGS' not in content:
            technical_config = """
# ===============================================================================
# TECHNICAL ANALYSIS SETTINGS
# ===============================================================================
RUN_TECHNICAL_ANALYSIS=true

# Analysis Period
ANALYSIS_PERIOD=1y
ANALYSIS_ASOF=

# RSI Settings
RSI_WINDOW=14
RSI_MIN=40.0
RSI_MAX=70.0

# Fundamental Screening
ROE_MIN=10.0
EPS_YOY_MIN=0.0

# Pattern Detection
DETECT_THREE_UP=true
DETECT_CUP_HANDLE=true

# Cup & Handle Parameters
CUP_LEN_MIN=30
CUP_LEN_MAX=200
CUP_DEPTH_MIN=12.0
CUP_DEPTH_MAX=50.0
HANDLE_LEN_MIN=5
HANDLE_LEN_MAX=30
HANDLE_DEPTH_MAX=15.0
RIM_TOLERANCE=0.08
HANDLE_END_BAND=0.05

# Analysis Output
ANALYSIS_OUTPUT_FORMATS=csv,excel,json
ANALYSIS_OUTPUT_DIR=analysis_reports
ANALYSIS_CHART_MATCHES=true
ANALYSIS_CHART_DIR=analysis_charts
"""
            with open(env_file, 'a') as f:
                f.write(technical_config)
    
    def _get_config_value(self, config_lines: List[str], key: str, default: str) -> str:
        """Get configuration value from config lines"""
        for line in config_lines:
            if line.strip().startswith(f"{key}="):
                return line.split('=', 1)[1].strip()
        return default
    
    def _update_env_config(self, env_file: Path, updates: Dict[str, str], additional_config: str = ""):
        """Update environment configuration file"""
        config_lines = []
        
        if env_file.exists():
            with open(env_file, 'r') as f:
                config_lines = f.readlines()
        
        # Update existing values
        for i, line in enumerate(config_lines):
            for key, value in updates.items():
                if line.strip().startswith(f"{key}="):
                    config_lines[i] = f"{key}={value}\n"
                    break
        
        # Add additional config
        if additional_config:
            config_lines.append(additional_config)
        
        # Write updated config
        with open(env_file, 'w') as f:
            f.writelines(config_lines)
    
    def setup_directories(self):
        """Create necessary directories including technical analysis directories"""
        print("\n📁 SETTING UP DIRECTORIES")
        print("-" * 25)
        
        directories = [
            'data', 
            'logs', 
            'cache', 
            'backups',
            'data/watchlists',  # For watchlists
            'analysis_reports',  # For technical analysis reports
            'analysis_charts',  # For pattern charts
            'reports',  # For general reports
            'charts'  # For general charts
        ]
        
        for dir_name in directories:
            dir_path = self.project_dir / dir_name
            if not dir_path.exists():
                dir_path.mkdir(parents=True)
                print(f"  ✅ Created {dir_name}/ directory")
            else:
                print(f"  📂 {dir_name}/ directory already exists")
    
    def create_watchlists(self):
        """Create sample watchlist files"""
        print("\n📋 CREATING WATCHLISTS")
        print("-" * 25)
        
        lists_dir = self.project_dir / "data/watchlists"
        lists_dir.mkdir(exist_ok=True)
        
        # Main watchlist
        watchlist = lists_dir / "watchlist.txt"
        if not watchlist.exists():
            content = """# Stock Watchlist
# One ticker per line or comma-separated
# Lines starting with # are comments

# Technology Leaders
AAPL
MSFT
GOOGL
AMZN
NVDA
META
TSLA

# Financial Services
JPM
BAC
WFC
GS
MS
V
MA

# Healthcare
UNH
JNJ
PFE
ABBV
MRK
LLY

# Energy
XOM
CVX
COP

# Consumer
WMT
HD
PG
KO
PEP
"""
            watchlist.write_text(content)
            print("  ✅ Created watchlist.txt")
        else:
            print("  📄 watchlist.txt already exists")
        
        # Top performers list
        top_performers = lists_dir / "top_performers.txt"
        if not top_performers.exists():
            content = """# Top Performing Stocks
# Updated regularly based on technical analysis

# High momentum stocks
NVDA
ARM
PLTR
AVGO
COIN

# Growth leaders
COST
LLY
NOW
PANW
CRM
"""
            top_performers.write_text(content)
            print("  ✅ Created top_performers.txt")
        else:
            print("  📄 top_performers.txt already exists")
        
        # Sector-specific lists
        sectors_list = lists_dir / "sectors.txt"
        if not sectors_list.exists():
            content = """# Sector Leaders
# Top stocks by sector

# Technology
AAPL,MSFT,NVDA

# Healthcare
UNH,JNJ,LLY

# Financials
JPM,BRK-B,V

# Energy
XOM,CVX,COP

# Consumer Discretionary
AMZN,TSLA,HD

# Consumer Staples
WMT,PG,KO

# Industrials
CAT,BA,UPS

# Materials
LIN,APD,SHW

# Real Estate
PLD,AMT,EQIX

# Utilities
NEE,SO,DUK

# Communication Services
GOOGL,META,DIS
"""
            sectors_list.write_text(content)
            print("  ✅ Created sectors.txt")
        else:
            print("  📄 sectors.txt already exists")
        
        # S&P 500 list (partial)
        sp500_list = lists_dir / "sp500_sample.txt"
        if not sp500_list.exists():
            content = """# S&P 500 Sample (Top 50 by market cap)
AAPL,MSFT,NVDA,AMZN,GOOGL,META,BRK-B,LLY,AVGO,JPM,
V,TSLA,WMT,MA,UNH,XOM,JNJ,HD,PG,COST,
ORCL,MRK,ABBV,BAC,CVX,CRM,AMD,KO,PEP,ACN,
NFLX,ADBE,TMO,WFC,CSCO,MCD,ABT,DHR,LIN,INTU,
QCOM,VZ,INTC,CMCSA,TXN,AMGN,PFE,IBM,DIS,PM
"""
            sp500_list.write_text(content)
            print("  ✅ Created sp500_sample.txt")
        else:
            print("  📄 sp500_sample.txt already exists")
    
    def make_scripts_executable(self):
        """Make shell scripts executable"""
        if platform.system() != "Windows":
            print("\n🔧 MAKING SCRIPTS EXECUTABLE")
            print("-" * 30)
            
            scripts = ['run_pipeline.sh', 'install_cron.sh', 'monitor_pipeline.sh', 'maintain_pipeline.sh']
            
            for script in scripts:
                script_path = self.project_dir / script
                if script_path.exists():
                    os.chmod(script_path, 0o755)
                    print(f"  ✅ Made {script} executable")
                else:
                    print(f"  ⚠️  {script} not found")
    
    def test_installation(self):
        """Test the installation including technical analysis"""
        print("\n🧪 TESTING INSTALLATION")
        print("-" * 25)
        
        # Test Python environment
        venv_dir = self.project_dir / "venv"
        if platform.system() == "Windows":
            python_path = venv_dir / "Scripts" / "python"
        else:
            python_path = venv_dir / "bin" / "python"
        
        # Test imports
        test_script = """
try:
    import requests
    import psutil
    import pandas
    import numpy
    import yfinance
    import matplotlib
    import tabulate
    import tqdm
    import openpyxl
    from pathlib import Path
    print("✅ Core dependencies imported successfully")
    
    # Test our modules
    import sys
    sys.path.insert(0, '.')
    
    from config import EXCHANGES, MAX_WORKERS
    print(f"✅ Configuration loaded: {len(EXCHANGES)} exchanges, {MAX_WORKERS} workers")
    
    from logger import setup_logger
    logger = setup_logger("test")
    print("✅ Logging system initialized")
    
    from storage import get_database
    db = get_database()
    print("✅ Database system ready")
    
    from blacklist import get_blacklist
    blacklist = get_blacklist()
    print("✅ Blacklist system initialized")
    
    # Test technical analysis configuration
    import os
    if os.path.exists('.env'):
        from dotenv import load_dotenv
        load_dotenv()
        analysis_enabled = os.getenv('RUN_TECHNICAL_ANALYSIS', 'false')
        print(f"✅ Technical analysis: {'enabled' if analysis_enabled == 'true' else 'disabled'}")
    
    print("\\n✅ All systems operational!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
"""
        
        print("  Testing Python environment and imports...")
        result = subprocess.run([
            str(python_path), '-c', test_script
        ], capture_output=True, text=True, cwd=str(self.project_dir))
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("❌ Installation test failed:")
            print(result.stderr)
            return False
        
        return True
    
    def display_next_steps(self):
        """Display next steps and usage information"""
        print("\n" + "=" * 70)
        print("🎉 SETUP COMPLETE!")
        print("=" * 70)
        
        print("\n📋 NEXT STEPS:")
        print()
        
        print("1. Run your first data collection with technical analysis:")
        if platform.system() == "Windows":
            print("   python main.py --force --debug")
        else:
            print("   ./run_pipeline.sh --force --debug")
        
        print("\n2. View technical analysis results:")
        print("   python technical_analysis.py")
        
        print("\n3. Schedule automatic daily runs:")
        if platform.system() != "Windows":
            print("   ./install_cron.sh")
        else:
            print("   Use Windows Task Scheduler to run main.py daily")
        
        print("\n4. Analyze your data:")
        print("   python analyze_data.py overview")
        print("   python export_data.py stocks --format excel --output analysis.xlsx")
        
        print("\n5. Query the database for recommendations:")
        print("   sqlite3 data/stocks_enhanced.db")
        print('   > SELECT * FROM technical_analysis WHERE recommendation="BUY";')
        
        print("\n📊 USEFUL COMMANDS:")
        print("   python main.py --help                 # Show all options")
        print("   python technical_analysis.py --symbols AAPL MSFT  # Analyze specific stocks")
        print("   python analyze_data.py report         # Generate comprehensive report")
        print("   tail -f logs/pipeline_*.log          # Monitor logs")
        
        print("\n🔧 CONFIGURATION:")
        print(f"   Edit .env file to customize settings")
        print(f"   Edit data/watchlists/watchlist.txt to add stocks to track")
        print(f"   Logs location: {self.project_dir}/logs/")
        print(f"   Database location: {self.project_dir}/data/stocks_enhanced.db")
        print(f"   Reports location: {self.project_dir}/analysis_reports/")
        
        print("\n📖 DOCUMENTATION:")
        print("   README.md contains comprehensive documentation")
        print("   Each module has --help option for detailed usage")
        
        print("\n💡 TIPS:")
        print("   • The pipeline runs daily to collect fresh data")
        print("   • Technical analysis identifies BUY/SELL opportunities")
        print("   • Pattern detection finds breakout setups")
        print("   • Check analysis_reports/ for daily Excel reports")
        print("   • Use --debug flag for detailed logging during testing")
        
        print("\n🆘 SUPPORT:")
        print("   • Check README.md for troubleshooting")
        print("   • Review logs in logs/ directory for errors")
        print(f"   • Contact: {__email__}")
        
        print("\n" + "=" * 70)
        print("Happy stock analysis! 📈")
        print("=" * 70)

def main():
    """Main setup function"""
    try:
        setup_manager = SetupManager()
        
        # Welcome and system check
        setup_manager.display_welcome()
        
        # Setup virtual environment
        setup_manager.setup_virtual_environment()
        
        # Configure environment
        setup_manager.configure_environment()
        
        # Setup directories
        setup_manager.setup_directories()
        
        # Create watchlists
        setup_manager.create_watchlists()
        
        # Make scripts executable
        setup_manager.make_scripts_executable()
        
        # Test installation
        if setup_manager.test_installation():
            # Display next steps
            setup_manager.display_next_steps()
        else:
            print("\n❌ Setup completed with issues. Check the test output above.")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
