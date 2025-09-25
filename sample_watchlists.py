#!/usr/bin/env python3
"""
Enhanced Sample Watchlist Setup with Investment Strategies
Creates comprehensive watchlists mapped to specific investment strategies
"""

__version__ = "1.5.0"
__copyright__ = "Copyright 2025, Richard D. Wissinger"
__author__ = "Richard D. Wissinger"
__email__ = "rick.wissinger@gmail.com"
__license__ = "MIT"
__status__ = "Production"

import os
import sys
from pathlib import Path
from datetime import datetime

# =============================================================================
# STRATEGY-BASED WATCHLIST DEFINITIONS
# =============================================================================

# Growth Strategy Watchlists
AGGRESSIVE_GROWTH_WATCHLIST = """# Aggressive Growth Strategy Watchlist
# High momentum growth stocks with strong earnings acceleration
# Target: RSI 80, ROE 20%, EPS Growth 50%
# Updated: 2025

# Technology Growth Leaders
NVDA
AMD
PLTR
CRWD
NET
SNOW
DDOG
DOCN
PATH
OKTA
ZS
PANW
MDB
TEAM
VEEV
HUBS
BILL
S
U
RBLX

# Emerging Tech
UPST
SOFI
AFRM
HOOD
COIN
SQ
SHOP
MELI
SE
NU

# Biotech Growth
MRNA
BNTX
REGN
VRTX
ALNY
BMRN
SGEN
EXAS
IONS
NBIX
"""

MODERATE_GROWTH_WATCHLIST = """# Moderate Growth Strategy Watchlist
# Sustainable growth with strong profitability
# Target: RSI 65, ROE 25%, EPS Growth 25%
# Updated: 2025

# Established Tech
MSFT
AAPL
GOOGL
META
CRM
ADBE
NOW
INTU
WDAY
CDNS
SNPS
ANSS
FTNT
IDXX
PAYC

# Consumer Growth
LULU
NKE
SBUX
CMG
DECK
ULTA
RH
ETSY
W
PTON

# Healthcare Growth
UNH
TMO
DHR
ISRG
ILMN
DXCM
PODD
TDOC
ZTS
ALGN
"""

CONSERVATIVE_GROWTH_WATCHLIST = """# Conservative Growth Strategy Watchlist
# Stable companies with consistent growth and high returns
# Target: RSI 55, ROE 30%, EPS Growth 15%
# Updated: 2025

# Blue Chip Growth
MSFT
AAPL
V
MA
UNH
HD
COST
ACN
TJX
ADP

# Defensive Growth
PG
JNJ
KO
PEP
WMT
CL
CHD
CLX
KMB
SYY

# Quality Compounders
BRK.B
MCO
SPGI
MSCI
CME
ICE
FIS
FISV
BR
TRU
"""

# Value Strategy Watchlists
DEEP_VALUE_WATCHLIST = """# Deep Value Strategy Watchlist
# Potentially oversold stocks with solid fundamentals
# Target: RSI 30, ROE 20%, EPS Growth 5%
# Updated: 2025

# Financial Value
BAC
WFC
C
USB
PNC
KEY
RF
CFG
ZION
FITB

# Energy Value
XOM
CVX
COP
EOG
OXY
DVN
FANG
MPC
PSX
VLO

# Industrial Value
BA
CAT
DE
RTX
LMT
NOC
GD
HII
TXT
TDG
"""

QUALITY_VALUE_WATCHLIST = """# Quality Value Strategy Watchlist
# High-quality companies trading at reasonable valuations
# Target: RSI 45, ROE 35%, EPS Growth 10%
# Updated: 2025

# Warren Buffett Style
BRK.B
AAPL
BAC
CVX
KO
AXP
KHC
USB
MNST
DVA

# Quality Industrials
HON
MMM
ITW
ETN
EMR
ROK
AME
ROP
DOV
PH

# Consumer Staples Value
PG
KO
PEP
CL
MDLZ
GIS
K
CPB
CAG
HSY
"""

# Dividend Strategy Watchlists
DIVIDEND_GROWTH_WATCHLIST = """# Dividend Growth Strategy Watchlist
# Companies with sustainable dividends and growth
# Target: RSI 50, ROE 25%, EPS Growth 8%
# Updated: 2025

# Dividend Aristocrats
JNJ
PG
KO
PEP
MMM
CL
MDT
ABT
WMT
TGT

# Dividend Growth
MSFT
AAPL
JPM
HD
UNH
V
MA
AVGO
TXN
ABBV

# REITs
O
PSA
WELL
SPG
PLD
AMT
CCI
EQIX
DLR
VICI
"""

HIGH_YIELD_WATCHLIST = """# High Yield Strategy Watchlist
# Mature companies with high dividend yields
# Target: RSI 40, ROE 30%, EPS Growth 3%
# Updated: 2025

# Telecom & Utilities
T
VZ
SO
DUK
NEE
AEP
XEL
WEC
ES
CNP

# Energy MLPs & High Yield
EPD
ET
KMI
MPLX
ENB
TC
PAA
WES
AM
USAC

# High Yield REITs
AGNC
STWD
ABR
TWO
BXMT
KREF
LADR
GPMT
RC
ACRE
"""

# Momentum Strategy Watchlists
BREAKOUT_MOMENTUM_WATCHLIST = """# Breakout Momentum Strategy Watchlist
# Stocks breaking out to new highs with strong fundamentals
# Target: RSI 85, ROE 15%, EPS Growth 40%
# Updated: 2025

# Tech Breakouts
SMCI
NVDA
ARM
PLTR
AVGO
MRVL
MPWR
LRCX
KLAC
AMAT

# AI & Cloud Leaders
MSFT
GOOGL
AMZN
META
CRM
SNOW
NOW
DDOG
MDB
NET

# Pharma Momentum
LLY
NVO
REGN
VRTX
ARGX
SGEN
BMRN
RVMD
KRYS
LEGN
"""

SWING_TRADING_WATCHLIST = """# Swing Trading Strategy Watchlist
# Medium-term momentum plays with solid backing
# Target: RSI 70, ROE 20%, EPS Growth 30%
# Updated: 2025

# Volatile Tech
TSLA
ROKU
SNAP
PINS
TWLO
DOCU
ZM
PTON
BYND
SPCE

# Retail Momentum
GME
AMC
BBBY
WISH
CLOV
BB
NOK
TLRY
SNDL
ACB

# Options Favorites
SPY
QQQ
IWM
AAPL
TSLA
NVDA
AMD
AMZN
NFLX
META
"""

# Sector Strategy Watchlists
TECH_SECTOR_WATCHLIST = """# Technology Sector Strategy Watchlist
# Technology sector growth stocks
# Target: RSI 70, ROE 22%, EPS Growth 35%
# Updated: 2025

# Software
MSFT
CRM
ADBE
NOW
ORCL
INTU
WDAY
TEAM
SPLK
PANW

# Semiconductors
NVDA
AMD
INTC
AVGO
QCOM
TXN
ADI
MRVL
XLNX
MU

# Internet & Cloud
GOOGL
META
AMZN
NFLX
SHOP
SNAP
PINS
TWTR
UBER
LYFT
"""

HEALTHCARE_SECTOR_WATCHLIST = """# Healthcare Sector Strategy Watchlist
# Stable healthcare companies with steady growth
# Target: RSI 55, ROE 28%, EPS Growth 12%
# Updated: 2025

# Pharma Giants
JNJ
PFE
MRK
ABBV
BMY
LLY
AMGN
GILD
BIIB
REGN

# Medical Devices
MDT
ABT
ISRG
SYK
BSX
EW
ZBH
HOLX
BAX
TFX

# Healthcare Services
UNH
CVS
CI
HUM
CNC
MOH
OSCR
CLOV
ALHC
GH
"""

FINANCIAL_SECTOR_WATCHLIST = """# Financial Sector Strategy Watchlist
# Value plays in financial services sector
# Target: RSI 45, ROE 15%, EPS Growth 8%
# Updated: 2025

# Banks
JPM
BAC
WFC
C
USB
PNC
TFC
FITB
KEY
RF

# Investment Banks
GS
MS
SCHW
BLK
BX
KKR
APO
ARES
CG
EVR

# Insurance
BRK.B
UNH
MET
PRU
AIG
TRV
ALL
CB
PGR
AFL
"""

# Risk-Based Watchlists
LOW_RISK_WATCHLIST = """# Low Risk Strategy Watchlist
# Conservative picks for risk-averse investors
# Target: RSI 50, ROE 25%, EPS Growth 10%
# Updated: 2025

# Stable Blue Chips
JNJ
PG
KO
PEP
WMT
MSFT
AAPL
BRK.B
UNH
JPM

# Utilities
NEE
SO
DUK
AEP
XEL
WEC
ES
ED
AEE
CNP

# Consumer Defensive
COST
WMT
TGT
DG
DLTR
KR
ACI
SFM
GO
NGVC
"""

HIGH_RISK_WATCHLIST = """# High Risk Strategy Watchlist
# High-risk, high-reward investment strategy
# Target: RSI 75, ROE 18%, EPS Growth 45%
# Updated: 2025

# Speculative Tech
PLTR
SPCE
LCID
RIVN
NIO
XPEV
LI
FSR
GOEV
RIDE

# Biotech Speculation
MRNA
BNTX
NVAX
OCGN
INO
SRNE
VXRT
ALT
GERN
AGEN

# Meme Stocks
GME
AMC
BB
NOK
BBBY
WISH
CLOV
SNDL
TLRY
RKT
"""

# Market Condition Watchlists
BULL_MARKET_WATCHLIST = """# Bull Market Strategy Watchlist
# Optimized for bull market conditions
# Target: RSI 70, ROE 25%, EPS Growth 30%
# Updated: 2025

# Growth Leaders
NVDA
MSFT
AAPL
GOOGL
AMZN
META
TSLA
AVGO
CRM
ADBE

# Cyclicals
JPM
BAC
GS
MS
V
MA
AXP
DFS
COF
SYF

# Risk-On Assets
ARKK
ARKG
ARKQ
ARKW
ARKF
QQQ
SMH
SOXX
IGV
VGT
"""

BEAR_MARKET_WATCHLIST = """# Bear Market Strategy Watchlist
# Defensive strategy for bear market conditions
# Target: RSI 35, ROE 30%, EPS Growth 5%
# Updated: 2025

# Defensive Stocks
JNJ
PG
KO
PEP
CL
WMT
COST
DG
DLTR
CVS

# Utilities & Staples
NEE
SO
DUK
XEL
WEC
ED
AEP
ES
AEE
CNP

# Safe Havens
GLD
SLV
TLT
IEF
SHY
BND
AGG
VCSH
MINT
SGOV
"""

# Special Strategy Watchlists
ESG_SUSTAINABLE_WATCHLIST = """# ESG & Sustainable Investing Watchlist
# Environmental, Social, and Governance focused companies
# Updated: 2025

# Clean Energy
TSLA
ENPH
SEDG
RUN
NEE
BEP
AES
PLUG
FCEL
BLDP

# Sustainable Tech
MSFT
AAPL
GOOGL
CRM
ADBE
IBM
HPE
DELL
CSCO
INTC

# Social Impact
SBUX
TGT
PG
UNH
CVS
JNJ
PFE
GILD
ABBV
BMY
"""

INTERNATIONAL_ADR_WATCHLIST = """# International ADR Watchlist
# Top international companies traded as ADRs
# Updated: 2025

# Asian Giants
BABA
TSM
BIDU
JD
PDD
NIO
XPEV
LI
SE
GRAB

# European Leaders
ASML
SAP
NESN
NOVN
ROG
MC
OR
SAN
BBVA
TEF

# Emerging Markets
VALE
ITUB
PBR
ABEV
NU
MELI
GLOB
STNE
PAGS
XP
"""

CRYPTOCURRENCY_RELATED_WATCHLIST = """# Cryptocurrency & Blockchain Watchlist
# Companies with crypto/blockchain exposure
# Updated: 2025

# Crypto Exchanges
COIN
HOOD
SQ
PYPL
SOFI

# Crypto Miners
MARA
RIOT
BTBT
HUT
HIVE
BITF
CLSK
CIFR
ARBK
GREE

# Blockchain/Crypto Holdings
MSTR
TSLA
SQ
GBTC
BITI
BITO
"""

SMALL_CAP_GEMS_WATCHLIST = """# Small Cap Gems Watchlist
# Promising small cap companies with growth potential
# Updated: 2025

# Tech Small Caps
APPS
MGNI
PUBM
TTD
ROKU
FUBO
VERI
BAND
PING
JAMF

# Healthcare Small Caps
RARE
TVTX
FOLD
PCRX
HALO
ACAD
NBIX
EXEL
INCY
JAZZ

# Consumer Small Caps
OLLI
FIVE
DKS
HIBB
BOOT
FL
CHWY
BARK
WOOF
FRPT
"""

# =============================================================================
# WATCHLIST METADATA AND CONFIGURATION
# =============================================================================

WATCHLIST_CONFIGS = {
    "Aggressive Growth": {
        "file": "aggressive_growth.txt",
        "content": AGGRESSIVE_GROWTH_WATCHLIST,
        "schedule": "daily",
        "strategies": ["Aggressive Growth", "Breakout Momentum"],
        "config": {
            "rsi_min": 70,
            "rsi_max": 90,
            "roe_min": 15,
            "eps_yoy_min": 40,
            "detect_three_up": True,
            "detect_cup_handle": True,
            "period": "6mo"
        }
    },
    "Moderate Growth": {
        "file": "moderate_growth.txt",
        "content": MODERATE_GROWTH_WATCHLIST,
        "schedule": "daily",
        "strategies": ["Moderate Growth"],
        "config": {
            "rsi_min": 50,
            "rsi_max": 80,
            "roe_min": 20,
            "eps_yoy_min": 20,
            "period": "1y"
        }
    },
    "Conservative Growth": {
        "file": "conservative_growth.txt",
        "content": CONSERVATIVE_GROWTH_WATCHLIST,
        "schedule": "weekly",
        "strategies": ["Conservative Growth", "Low Risk"],
        "config": {
            "rsi_min": 40,
            "rsi_max": 70,
            "roe_min": 25,
            "eps_yoy_min": 10,
            "period": "1y"
        }
    },
    "Deep Value": {
        "file": "deep_value.txt",
        "content": DEEP_VALUE_WATCHLIST,
        "schedule": "weekly",
        "strategies": ["Deep Value"],
        "config": {
            "rsi_min": 20,
            "rsi_max": 45,
            "roe_min": 15,
            "eps_yoy_min": 0,
            "period": "2y"
        }
    },
    "Quality Value": {
        "file": "quality_value.txt",
        "content": QUALITY_VALUE_WATCHLIST,
        "schedule": "weekly",
        "strategies": ["Quality Value"],
        "config": {
            "rsi_min": 35,
            "rsi_max": 65,
            "roe_min": 30,
            "eps_yoy_min": 5,
            "period": "1y"
        }
    },
    "Dividend Growth": {
        "file": "dividend_growth.txt",
        "content": DIVIDEND_GROWTH_WATCHLIST,
        "schedule": "monthly",
        "strategies": ["Dividend Growth"],
        "config": {
            "rsi_min": 35,
            "rsi_max": 75,
            "roe_min": 20,
            "eps_yoy_min": 5,
            "focus": "dividend_yield"
        }
    },
    "High Yield": {
        "file": "high_yield.txt",
        "content": HIGH_YIELD_WATCHLIST,
        "schedule": "monthly",
        "strategies": ["High Yield"],
        "config": {
            "rsi_min": 25,
            "rsi_max": 65,
            "roe_min": 25,
            "eps_yoy_min": 0,
            "focus": "dividend_yield"
        }
    },
    "Breakout Momentum": {
        "file": "breakout_momentum.txt",
        "content": BREAKOUT_MOMENTUM_WATCHLIST,
        "schedule": "daily",
        "strategies": ["Breakout Momentum"],
        "config": {
            "rsi_min": 75,
            "rsi_max": 95,
            "roe_min": 10,
            "eps_yoy_min": 35,
            "detect_three_up": True,
            "period": "3mo"
        }
    },
    "Swing Trading": {
        "file": "swing_trading.txt",
        "content": SWING_TRADING_WATCHLIST,
        "schedule": "daily",
        "strategies": ["Swing Trading"],
        "config": {
            "rsi_min": 60,
            "rsi_max": 85,
            "roe_min": 15,
            "eps_yoy_min": 25,
            "period": "1mo"
        }
    },
    "Tech Sector": {
        "file": "tech_sector.txt",
        "content": TECH_SECTOR_WATCHLIST,
        "schedule": "daily",
        "strategies": ["Tech Growth"],
        "config": {
            "rsi_min": 55,
            "rsi_max": 85,
            "roe_min": 18,
            "eps_yoy_min": 30,
            "sector_focus": "Technology"
        }
    },
    "Healthcare Sector": {
        "file": "healthcare_sector.txt",
        "content": HEALTHCARE_SECTOR_WATCHLIST,
        "schedule": "weekly",
        "strategies": ["Healthcare Stable"],
        "config": {
            "rsi_min": 40,
            "rsi_max": 75,
            "roe_min": 23,
            "eps_yoy_min": 8,
            "sector_focus": "Healthcare"
        }
    },
    "Financial Sector": {
        "file": "financial_sector.txt",
        "content": FINANCIAL_SECTOR_WATCHLIST,
        "schedule": "weekly",
        "strategies": ["Financial Value"],
        "config": {
            "rsi_min": 35,
            "rsi_max": 70,
            "roe_min": 12,
            "eps_yoy_min": 5,
            "sector_focus": "Financial"
        }
    },
    "Low Risk": {
        "file": "low_risk.txt",
        "content": LOW_RISK_WATCHLIST,
        "schedule": "monthly",
        "strategies": ["Low Risk"],
        "config": {
            "rsi_min": 35,
            "rsi_max": 65,
            "roe_min": 20,
            "eps_yoy_min": 8,
            "max_volatility": 20
        }
    },
    "High Risk": {
        "file": "high_risk.txt",
        "content": HIGH_RISK_WATCHLIST,
        "schedule": "daily",
        "strategies": ["High Risk"],
        "config": {
            "rsi_min": 60,
            "rsi_max": 95,
            "roe_min": 10,
            "eps_yoy_min": 40,
            "min_volatility": 40
        }
    },
    "Bull Market": {
        "file": "bull_market.txt",
        "content": BULL_MARKET_WATCHLIST,
        "schedule": "daily",
        "strategies": ["Bull Market"],
        "config": {
            "rsi_min": 60,
            "rsi_max": 85,
            "roe_min": 20,
            "eps_yoy_min": 25,
            "market_condition": "bullish"
        }
    },
    "Bear Market": {
        "file": "bear_market.txt",
        "content": BEAR_MARKET_WATCHLIST,
        "schedule": "weekly",
        "strategies": ["Bear Market"],
        "config": {
            "rsi_min": 25,
            "rsi_max": 55,
            "roe_min": 25,
            "eps_yoy_min": 0,
            "market_condition": "bearish"
        }
    },
    "ESG Sustainable": {
        "file": "esg_sustainable.txt",
        "content": ESG_SUSTAINABLE_WATCHLIST,
        "schedule": "monthly",
        "strategies": ["Moderate Growth", "Low Risk"],
        "config": {
            "rsi_min": 40,
            "rsi_max": 75,
            "roe_min": 20,
            "eps_yoy_min": 10,
            "esg_focus": True
        }
    },
    "International ADR": {
        "file": "international_adr.txt",
        "content": INTERNATIONAL_ADR_WATCHLIST,
        "schedule": "weekly",
        "strategies": ["Moderate Growth"],
        "config": {
            "rsi_min": 45,
            "rsi_max": 80,
            "roe_min": 15,
            "eps_yoy_min": 15,
            "international": True
        }
    },
    "Cryptocurrency Related": {
        "file": "crypto_related.txt",
        "content": CRYPTOCURRENCY_RELATED_WATCHLIST,
        "schedule": "daily",
        "strategies": ["High Risk", "Breakout Momentum"],
        "config": {
            "rsi_min": 50,
            "rsi_max": 90,
            "roe_min": 5,
            "eps_yoy_min": 20,
            "crypto_exposure": True
        }
    },
    "Small Cap Gems": {
        "file": "small_cap_gems.txt",
        "content": SMALL_CAP_GEMS_WATCHLIST,
        "schedule": "weekly",
        "strategies": ["Aggressive Growth", "High Risk"],
        "config": {
            "rsi_min": 45,
            "rsi_max": 85,
            "roe_min": 10,
            "eps_yoy_min": 25,
            "market_cap_max": 5000000000  # $5B
        }
    }
}

# =============================================================================
# SETUP FUNCTIONS
# =============================================================================

def create_watchlist_files(output_dir: str = "data/watchlists") -> Path:
    """Create all watchlist files in the specified directory"""
    watchlist_dir = Path(output_dir)
    watchlist_dir.mkdir(parents=True, exist_ok=True)
    
    created_files = []
    
    for name, config in WATCHLIST_CONFIGS.items():
        file_path = watchlist_dir / config["file"]
        with open(file_path, 'w') as f:
            f.write(config["content"])
        created_files.append(file_path)
        print(f"✅ Created watchlist: {name} -> {file_path}")
    
    print(f"\n📁 Created {len(created_files)} strategy-based watchlists in {watchlist_dir}")
    return watchlist_dir

def setup_watchlists_in_database(email_recipients=None):
    """Import watchlists into database with strategy configurations"""
    
    sys.path.insert(0, str(Path(__file__).parent))
    
    try:
        from watchlist_manager import WatchlistManager, Watchlist
        from storage import init_db
        
        # Initialize database
        init_db()
        
        # Create watchlist manager
        manager = WatchlistManager()
        
        print("\n💾 Setting up strategy-based watchlists in database...")
        
        for name, config in WATCHLIST_CONFIGS.items():
            # Check if watchlist exists
            existing = manager.get_watchlist(name=name)
            
            # Parse symbols from content
            symbols = []
            for line in config["content"].split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    symbols.append(line)
            
            if existing:
                # Update existing watchlist
                existing.symbols = symbols
                existing.analysis_schedule = config["schedule"]
                existing.analysis_config = {
                    "strategies": config["strategies"],
                    **config["config"]
                }
                if email_recipients and not existing.email_recipients:
                    existing.email_recipients = email_recipients
                
                manager.update_watchlist(existing)
                print(f"  📝 Updated: {name} ({len(symbols)} symbols)")
            else:
                # Create new watchlist
                watchlist = Watchlist(
                    name=name,
                    description=f"Strategy-based watchlist for {name}",
                    symbols=symbols,
                    email_recipients=email_recipients or [],
                    analysis_schedule=config["schedule"],
                    analysis_config={
                        "strategies": config["strategies"],
                        **config["config"]
                    }
                )
                
                watchlist_id = manager.create_watchlist(watchlist)
                print(f"  ✨ Created: {name} ({len(symbols)} symbols, ID: {watchlist_id})")
        
        # Display summary
        print("\n" + "=" * 60)
        print("WATCHLIST SETUP COMPLETE")
        print("=" * 60)
        
        all_watchlists = manager.get_all_watchlists()
        
        # Group by schedule
        by_schedule = {"daily": [], "weekly": [], "monthly": []}
        for wl in all_watchlists:
            by_schedule[wl.analysis_schedule].append(wl)
        
        print("\n📊 Watchlists by Analysis Schedule:")
        for schedule, watchlists in by_schedule.items():
            print(f"\n{schedule.upper()} ({len(watchlists)} watchlists):")
            for wl in watchlists:
                strategies = wl.analysis_config.get('strategies', [])
                print(f"  • {wl.name}: {len(wl.symbols)} symbols, Strategies: {', '.join(strategies)}")
        
        # Show strategy distribution
        strategy_count = {}
        for wl in all_watchlists:
            for strategy in wl.analysis_config.get('strategies', []):
                strategy_count[strategy] = strategy_count.get(strategy, 0) + 1
        
        print("\n🎯 Strategy Distribution:")
        for strategy, count in sorted(strategy_count.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {strategy}: {count} watchlists")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error: Required module not found: {e}")
        return False
    except Exception as e:
        print(f"❌ Error setting up watchlists: {e}")
        return False

def generate_analysis_schedule():
    """Generate cron schedule for watchlist analysis"""
    
    schedule = """
# ============================================================================
# AUTOMATED WATCHLIST ANALYSIS SCHEDULE (crontab)
# ============================================================================

# Daily Analysis (6:30 PM ET - after market close)
30 18 * * 1-5 cd /path/to/project && python technical_analysis_with_strategies.py --schedule daily

# Weekly Analysis (Sunday 8:00 PM ET)
0 20 * * 0 cd /path/to/project && python technical_analysis_with_strategies.py --schedule weekly

# Monthly Analysis (1st of month, 8:00 PM ET)
0 20 1 * * cd /path/to/project && python technical_analysis_with_strategies.py --schedule monthly

# Market Open Alert Check (9:25 AM ET - before market open)
25 9 * * 1-5 cd /path/to/project && python market_alerts.py --check-premarket

# Market Close Summary (4:05 PM ET - after market close)
5 16 * * 1-5 cd /path/to/project && python market_summary.py --daily-report

# Weekend Portfolio Review (Saturday 10:00 AM ET)
0 10 * * 6 cd /path/to/project && python portfolio_review.py --weekly-rebalance
"""
    
    return schedule

def display_features():
    """Display additional features and capabilities"""
    
    print("\n" + "=" * 60)
    print("ADDITIONAL FEATURES & CAPABILITIES")
    print("=" * 60)
    
    features = """
📈 STRATEGY MATCHING
  • Each stock scored against 18 investment strategies
  • Automatic best strategy identification
  • Multi-strategy portfolio optimization
  
🎯 SMART RECOMMENDATIONS
  • STRONG BUY: Score ≥ 80
  • BUY: Score ≥ 65
  • HOLD: Score ≥ 50
  • WEAK HOLD: Score ≥ 35
  • AVOID: Score < 35

📊 PATTERN DETECTION
  • Three-Up Pattern (momentum indicator)
  • Cup & Handle Pattern (breakout indicator)
  • Support/Resistance levels
  • Moving average crossovers

📧 EMAIL REPORTS
  • Beautiful HTML formatting
  • Strategy-specific recommendations
  • Performance metrics & charts
  • Attached Excel/CSV/JSON files

🔄 AUTOMATED SCHEDULING
  • Daily: High-frequency trading strategies
  • Weekly: Swing trading & sector rotation
  • Monthly: Long-term investment reviews

🛡️ RISK MANAGEMENT
  • Volatility tracking
  • Maximum drawdown analysis
  • Beta correlation
  • Diversification scoring

📱 ALERT SYSTEM (Coming Soon)
  • Price breakout alerts
  • Pattern completion notifications
  • Strategy score changes
  • Earnings announcements

📈 PERFORMANCE TRACKING (Coming Soon)
  • Portfolio returns vs benchmarks
  • Strategy performance metrics
  • Win/loss ratios
  • Risk-adjusted returns

🔄 REBALANCING SUGGESTIONS (Coming Soon)
  • Optimal portfolio weights
  • Sector allocation
  • Risk parity adjustments
  • Tax-loss harvesting

🌍 MARKET ANALYSIS (Coming Soon)
  • Market regime detection
  • Sentiment analysis
  • Correlation matrices
  • Economic indicators
"""
    
    print(features)

def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Setup enhanced strategy-based watchlists for stock analysis"
    )
    parser.add_argument(
        "--recipients",
        nargs="+",
        help="Email addresses to receive analysis reports"
    )
    parser.add_argument(
        "--output-dir",
        default="data/watchlists",
        help="Directory for watchlist files (default: data/watchlists)"
    )
    parser.add_argument(
        "--skip-files",
        action="store_true",
        help="Skip creating watchlist files"
    )
    parser.add_argument(
        "--show-schedule",
        action="store_true",
        help="Show cron schedule for automation"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ENHANCED STRATEGY-BASED WATCHLIST SETUP")
    print("=" * 60)
    print(f"Version: 2.0.0")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Create watchlist files
    if not args.skip_files:
        print("\n📁 Creating strategy-based watchlist files...")
        create_watchlist_files(args.output_dir)
    
    # Set up in database
    print("\n💾 Setting up watchlists in database...")
    success = setup_watchlists_in_database(args.recipients)
    
    if success:
        # Show features
        display_features()
        
        # Show schedule if requested
        if args.show_schedule:
            print("\n📅 CRON SCHEDULE FOR AUTOMATION:")
            print(generate_analysis_schedule())
        
        print("\n✅ Setup complete! Your strategy-based watchlist system is ready.")
        
        if args.recipients:
            print(f"\n📧 Email recipients configured: {', '.join(args.recipients)}")
        else:
            print("\n⚠️  No email recipients configured.")
            print("   Add recipients with: --recipients email1@example.com email2@example.com")
        
        print("\n🚀 Next Steps:")
        print("1. Copy .env.example to .env and configure email settings")
        print("2. Run: python setup.py --initialize")
        print("3. Test analysis: python technical_analysis_with_strategies.py --watchlist 'Aggressive Growth'")
        print("4. Schedule automated analysis using cron (use --show-schedule to see examples)")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")

if __name__ == "__main__":
    main()