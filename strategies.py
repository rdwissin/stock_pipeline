"""
Predefined investment strategies with criteria and tolerances
"""

__version__ = "1.5.0"
__copyright__ = "Copyright 2025, Richard D. Wissinger"
__author__ = "Richard D. Wissinger"
__email__ = "rick.wissinger@gmail.com"
__license__ = "MIT"
__status__ = "Production"

# =============================================================================
# INVESTMENT STRATEGIES
# Format: (RSI_target, ROE_target, EPS_growth_target)
# =============================================================================

# Growth investing strategies
AGGRESSIVE_GROWTH = {
    'criteria': (80, 20, 50),      # High RSI, good ROE, strong growth
    'tolerance': (10, 5, 30),      # Moderate tolerance
    'description': 'High momentum growth stocks with strong earnings acceleration'
}

MODERATE_GROWTH = {
    'criteria': (65, 25, 25),      # Moderate RSI, high ROE, good growth  
    'tolerance': (15, 5, 20),      # Balanced tolerance
    'description': 'Sustainable growth with strong profitability'
}

CONSERVATIVE_GROWTH = {
    'criteria': (55, 30, 15),      # Lower RSI, very high ROE, steady growth
    'tolerance': (20, 5, 15),      # More lenient
    'description': 'Stable companies with consistent growth and high returns'
}

# Value investing strategies  
DEEP_VALUE = {
    'criteria': (30, 20, 5),       # Low RSI (oversold), decent ROE, stable
    'tolerance': (15, 10, 20),     # Flexible on growth
    'description': 'Potentially oversold stocks with solid fundamentals'
}

QUALITY_VALUE = {
    'criteria': (45, 35, 10),      # Moderate RSI, excellent ROE, steady growth
    'tolerance': (20, 10, 15),     # Quality focused
    'description': 'High-quality companies trading at reasonable valuations'
}

# Dividend strategies
DIVIDEND_GROWTH = {
    'criteria': (50, 25, 8),       # Moderate RSI, good ROE, steady growth
    'tolerance': (25, 5, 12),      # Stability focused  
    'description': 'Companies with sustainable dividends and growth'
}

HIGH_YIELD = {
    'criteria': (40, 30, 3),       # Lower RSI, high ROE, minimal growth needed
    'tolerance': (30, 10, 20),     # Very flexible
    'description': 'Mature companies with high dividend yields'
}

# Momentum strategies
BREAKOUT_MOMENTUM = {
    'criteria': (85, 15, 40),      # Very high RSI, decent ROE, strong growth
    'tolerance': (5, 10, 25),      # Tight on RSI, flexible on others
    'description': 'Stocks breaking out to new highs with strong fundamentals'
}

SWING_TRADING = {
    'criteria': (70, 20, 30),      # High RSI, good ROE, good growth
    'tolerance': (10, 15, 35),     # Balanced for short-term trades
    'description': 'Medium-term momentum plays with solid backing'
}

# Sector-specific strategies
TECH_GROWTH = {
    'criteria': (70, 22, 35),      # Tech-appropriate metrics
    'tolerance': (15, 8, 30),      # Tech sector volatility considered
    'description': 'Technology sector growth stocks'
}

HEALTHCARE_STABLE = {
    'criteria': (55, 28, 12),      # Healthcare sector characteristics
    'tolerance': (20, 5, 18),      # Stability focused for healthcare
    'description': 'Stable healthcare companies with steady growth'
}

FINANCIAL_VALUE = {
    'criteria': (45, 15, 8),       # Financial sector metrics (lower ROE normal)
    'tolerance': (25, 8, 20),      # Adapted for financial sector
    'description': 'Value plays in financial services sector'
}

# Risk-based strategies
LOW_RISK = {
    'criteria': (50, 25, 10),      # Balanced metrics, lower volatility
    'tolerance': (15, 5, 10),      # Tight tolerances for consistency
    'description': 'Conservative picks for risk-averse investors'
}

MODERATE_RISK = {
    'criteria': (60, 22, 20),      # Moderate risk/reward balance
    'tolerance': (15, 8, 20),      # Standard tolerances
    'description': 'Balanced risk/reward investment approach'
}

HIGH_RISK = {
    'criteria': (75, 18, 45),      # Higher volatility acceptable for higher returns
    'tolerance': (20, 12, 40),     # Wider tolerances for aggressive approach
    'description': 'High-risk, high-reward investment strategy'
}

# Market condition strategies
BULL_MARKET = {
    'criteria': (70, 25, 30),      # Momentum-friendly in bull markets
    'tolerance': (10, 5, 25),      # Tighter criteria in good times
    'description': 'Optimized for bull market conditions'
}

BEAR_MARKET = {
    'criteria': (35, 30, 5),       # Quality and value focused in bear markets
    'tolerance': (20, 5, 25),      # More flexible on growth
    'description': 'Defensive strategy for bear market conditions'
}

SIDEWAYS_MARKET = {
    'criteria': (55, 28, 15),      # Balanced approach for sideways markets
    'tolerance': (25, 8, 20),      # Moderate flexibility
    'description': 'Range-trading strategy for sideways markets'
}

# =============================================================================
# STRATEGY COLLECTIONS
# =============================================================================

GROWTH_STRATEGIES = {
    'Aggressive Growth': AGGRESSIVE_GROWTH,
    'Moderate Growth': MODERATE_GROWTH,
    'Conservative Growth': CONSERVATIVE_GROWTH
}

VALUE_STRATEGIES = {
    'Deep Value': DEEP_VALUE,
    'Quality Value': QUALITY_VALUE
}

DIVIDEND_STRATEGIES = {
    'Dividend Growth': DIVIDEND_GROWTH,
    'High Yield': HIGH_YIELD
}

MOMENTUM_STRATEGIES = {
    'Breakout Momentum': BREAKOUT_MOMENTUM,
    'Swing Trading': SWING_TRADING
}

SECTOR_STRATEGIES = {
    'Tech Growth': TECH_GROWTH,
    'Healthcare Stable': HEALTHCARE_STABLE,
    'Financial Value': FINANCIAL_VALUE
}

RISK_STRATEGIES = {
    'Low Risk': LOW_RISK,
    'Moderate Risk': MODERATE_RISK,
    'High Risk': HIGH_RISK
}

MARKET_STRATEGIES = {
    'Bull Market': BULL_MARKET,
    'Bear Market': BEAR_MARKET,
    'Sideways Market': SIDEWAYS_MARKET
}

ALL_STRATEGIES = {
    **GROWTH_STRATEGIES,
    **VALUE_STRATEGIES,
    **DIVIDEND_STRATEGIES,
    **MOMENTUM_STRATEGIES,
    **SECTOR_STRATEGIES,
    **RISK_STRATEGIES,
    **MARKET_STRATEGIES
}