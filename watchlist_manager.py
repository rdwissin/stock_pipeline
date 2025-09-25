#!/usr/bin/env python3
"""
Watchlist Management System with Email Notifications
Manages stock watchlists and associated email recipients for technical analysis reports
"""

__version__ = "1.5.0"
__copyright__ = "Copyright 2025, Richard D. Wissinger"
__author__ = "Richard D. Wissinger"
__email__ = "rick.wissinger@gmail.com"
__license__ = "MIT"
__status__ = "Production"

import json
import sqlite3
import smtplib
import ssl
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import pandas as pd
import os

from config import DB_PATH, DATA_DIR
from logger import setup_logger, performance_monitor
from storage import get_database

logger = setup_logger(__name__)

# ===============================================================================
# EMAIL CONFIGURATION
# ===============================================================================

class EmailConfig:
    """Email configuration from environment variables"""
    def __init__(self):
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_username = os.getenv('SMTP_USERNAME', '')
        self.smtp_password = os.getenv('SMTP_PASSWORD', '')
        self.from_email = os.getenv('FROM_EMAIL', self.smtp_username)
        self.from_name = os.getenv('FROM_NAME', 'Stock Analysis System')
        self.use_tls = os.getenv('SMTP_USE_TLS', 'true').lower() == 'true'
        self.enabled = os.getenv('EMAIL_NOTIFICATIONS', 'true').lower() == 'true'
        
    def is_configured(self) -> bool:
        """Check if email is properly configured"""
        return bool(self.smtp_server and self.smtp_username and self.smtp_password)

# ===============================================================================
# DATA CLASSES
# ===============================================================================

@dataclass
class Watchlist:
    """Watchlist data structure"""
    id: Optional[int] = None
    name: str = ""
    description: str = ""
    symbols: List[str] = None
    email_recipients: List[str] = None
    analysis_schedule: str = "daily"  # daily, weekly, monthly
    last_analysis: Optional[str] = None
    created_by: str = "system"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    is_active: bool = True
    analysis_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = []
        if self.email_recipients is None:
            self.email_recipients = []
        if self.analysis_config is None:
            self.analysis_config = {}
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'symbols': self.symbols,
            'email_recipients': self.email_recipients,
            'analysis_schedule': self.analysis_schedule,
            'last_analysis': self.last_analysis,
            'created_by': self.created_by,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'is_active': self.is_active,
            'analysis_config': self.analysis_config
        }

# ===============================================================================
# DATABASE SCHEMA
# ===============================================================================

def init_watchlist_schema():
    """Initialize watchlist database schema"""
    db = get_database()
    
    with db.get_connection() as conn:
        with db.transaction(conn) as cursor:
            # Watchlists table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS watchlists (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    analysis_schedule TEXT DEFAULT 'daily',
                    last_analysis TEXT,
                    created_by TEXT DEFAULT 'system',
                    is_active BOOLEAN DEFAULT 1,
                    analysis_config TEXT,  -- JSON configuration
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Watchlist symbols table (many-to-many)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS watchlist_symbols (
                    watchlist_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    added_date TEXT DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT,
                    PRIMARY KEY (watchlist_id, symbol),
                    FOREIGN KEY (watchlist_id) REFERENCES watchlists(id) ON DELETE CASCADE
                )
            """)
            
            # Watchlist recipients table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS watchlist_recipients (
                    watchlist_id INTEGER NOT NULL,
                    email TEXT NOT NULL,
                    recipient_name TEXT,
                    notification_preferences TEXT,  -- JSON preferences
                    is_active BOOLEAN DEFAULT 1,
                    added_date TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (watchlist_id, email),
                    FOREIGN KEY (watchlist_id) REFERENCES watchlists(id) ON DELETE CASCADE
                )
            """)
            
            # Analysis history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS watchlist_analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    watchlist_id INTEGER NOT NULL,
                    analysis_date TEXT NOT NULL,
                    total_symbols INTEGER,
                    buy_signals INTEGER,
                    sell_signals INTEGER,
                    hold_signals INTEGER,
                    report_file TEXT,
                    email_sent BOOLEAN DEFAULT 0,
                    email_recipients TEXT,  -- JSON list
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (watchlist_id) REFERENCES watchlists(id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_schedule ON watchlists(analysis_schedule, is_active)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_symbols ON watchlist_symbols(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_recipients ON watchlist_recipients(email)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_history ON watchlist_analysis_history(watchlist_id, analysis_date)")
            
            logger.info("Watchlist schema initialized successfully")

# ===============================================================================
# WATCHLIST MANAGER
# ===============================================================================

class WatchlistManager:
    """Manages watchlists and their associated operations"""
    
    def __init__(self):
        self.db = get_database()
        self.email_config = EmailConfig()
        init_watchlist_schema()
        
    @performance_monitor("create_watchlist")
    def create_watchlist(self, watchlist: Watchlist) -> int:
        """Create a new watchlist"""
        with self.db.get_connection() as conn:
            with self.db.transaction(conn) as cursor:
                # Insert watchlist
                cursor.execute("""
                    INSERT INTO watchlists 
                    (name, description, analysis_schedule, created_by, is_active, analysis_config)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    watchlist.name,
                    watchlist.description,
                    watchlist.analysis_schedule,
                    watchlist.created_by,
                    watchlist.is_active,
                    json.dumps(watchlist.analysis_config)
                ))
                
                watchlist_id = cursor.lastrowid
                
                # Add symbols
                for symbol in watchlist.symbols:
                    cursor.execute("""
                        INSERT OR IGNORE INTO watchlist_symbols (watchlist_id, symbol)
                        VALUES (?, ?)
                    """, (watchlist_id, symbol.upper()))
                
                # Add recipients
                for email in watchlist.email_recipients:
                    cursor.execute("""
                        INSERT OR IGNORE INTO watchlist_recipients (watchlist_id, email)
                        VALUES (?, ?)
                    """, (watchlist_id, email.lower()))
                
                logger.info(f"Created watchlist '{watchlist.name}' with ID {watchlist_id}")
                return watchlist_id
    
    @performance_monitor("get_watchlist")
    def get_watchlist(self, watchlist_id: Optional[int] = None, 
                     name: Optional[str] = None) -> Optional[Watchlist]:
        """Get a watchlist by ID or name"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            if watchlist_id:
                cursor.execute("SELECT * FROM watchlists WHERE id = ?", (watchlist_id,))
            elif name:
                cursor.execute("SELECT * FROM watchlists WHERE name = ?", (name,))
            else:
                return None
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Get symbols
            cursor.execute("""
                SELECT symbol FROM watchlist_symbols 
                WHERE watchlist_id = ?
                ORDER BY symbol
            """, (row[0],))
            symbols = [r[0] for r in cursor.fetchall()]
            
            # Get recipients
            cursor.execute("""
                SELECT email FROM watchlist_recipients 
                WHERE watchlist_id = ? AND is_active = 1
            """, (row[0],))
            recipients = [r[0] for r in cursor.fetchall()]
            
            # Parse analysis config
            analysis_config = json.loads(row[7]) if row[7] else {}
            
            return Watchlist(
                id=row[0],
                name=row[1],
                description=row[2],
                symbols=symbols,
                email_recipients=recipients,
                analysis_schedule=row[3],
                last_analysis=row[4],
                created_by=row[5],
                is_active=bool(row[6]),
                analysis_config=analysis_config,
                created_at=row[8],
                updated_at=row[9]
            )
    
    @performance_monitor("get_all_watchlists")
    def get_all_watchlists(self, active_only: bool = True) -> List[Watchlist]:
        """Get all watchlists"""
        watchlists = []
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT id FROM watchlists"
            if active_only:
                query += " WHERE is_active = 1"
            query += " ORDER BY name"
            
            cursor.execute(query)
            
            for row in cursor.fetchall():
                watchlist = self.get_watchlist(watchlist_id=row[0])
                if watchlist:
                    watchlists.append(watchlist)
        
        return watchlists
    
    @performance_monitor("update_watchlist")
    def update_watchlist(self, watchlist: Watchlist) -> bool:
        """Update an existing watchlist"""
        if not watchlist.id:
            logger.error("Cannot update watchlist without ID")
            return False
        
        with self.db.get_connection() as conn:
            with self.db.transaction(conn) as cursor:
                # Update watchlist
                cursor.execute("""
                    UPDATE watchlists
                    SET name = ?, description = ?, analysis_schedule = ?,
                        is_active = ?, analysis_config = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (
                    watchlist.name,
                    watchlist.description,
                    watchlist.analysis_schedule,
                    watchlist.is_active,
                    json.dumps(watchlist.analysis_config),
                    watchlist.id
                ))
                
                # Update symbols (delete and re-add)
                cursor.execute("DELETE FROM watchlist_symbols WHERE watchlist_id = ?", (watchlist.id,))
                for symbol in watchlist.symbols:
                    cursor.execute("""
                        INSERT INTO watchlist_symbols (watchlist_id, symbol)
                        VALUES (?, ?)
                    """, (watchlist.id, symbol.upper()))
                
                # Update recipients (delete and re-add)
                cursor.execute("DELETE FROM watchlist_recipients WHERE watchlist_id = ?", (watchlist.id,))
                for email in watchlist.email_recipients:
                    cursor.execute("""
                        INSERT INTO watchlist_recipients (watchlist_id, email)
                        VALUES (?, ?)
                    """, (watchlist.id, email.lower()))
                
                logger.info(f"Updated watchlist '{watchlist.name}'")
                return True
    
    @performance_monitor("delete_watchlist")
    def delete_watchlist(self, watchlist_id: int) -> bool:
        """Delete a watchlist"""
        with self.db.get_connection() as conn:
            with self.db.transaction(conn) as cursor:
                cursor.execute("DELETE FROM watchlists WHERE id = ?", (watchlist_id,))
                
                if cursor.rowcount > 0:
                    logger.info(f"Deleted watchlist with ID {watchlist_id}")
                    return True
                return False
    
    @performance_monitor("add_symbols_to_watchlist")
    def add_symbols_to_watchlist(self, watchlist_id: int, symbols: List[str]) -> int:
        """Add symbols to a watchlist"""
        added = 0
        
        with self.db.get_connection() as conn:
            with self.db.transaction(conn) as cursor:
                for symbol in symbols:
                    cursor.execute("""
                        INSERT OR IGNORE INTO watchlist_symbols (watchlist_id, symbol)
                        VALUES (?, ?)
                    """, (watchlist_id, symbol.upper()))
                    added += cursor.rowcount
                
                cursor.execute("""
                    UPDATE watchlists SET updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (watchlist_id,))
        
        logger.info(f"Added {added} symbols to watchlist {watchlist_id}")
        return added
    
    @performance_monitor("remove_symbols_from_watchlist")
    def remove_symbols_from_watchlist(self, watchlist_id: int, symbols: List[str]) -> int:
        """Remove symbols from a watchlist"""
        removed = 0
        
        with self.db.get_connection() as conn:
            with self.db.transaction(conn) as cursor:
                for symbol in symbols:
                    cursor.execute("""
                        DELETE FROM watchlist_symbols 
                        WHERE watchlist_id = ? AND symbol = ?
                    """, (watchlist_id, symbol.upper()))
                    removed += cursor.rowcount
                
                cursor.execute("""
                    UPDATE watchlists SET updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (watchlist_id,))
        
        logger.info(f"Removed {removed} symbols from watchlist {watchlist_id}")
        return removed
    
    @performance_monitor("add_recipients_to_watchlist")
    def add_recipients_to_watchlist(self, watchlist_id: int, emails: List[str]) -> int:
        """Add email recipients to a watchlist"""
        added = 0
        
        with self.db.get_connection() as conn:
            with self.db.transaction(conn) as cursor:
                for email in emails:
                    cursor.execute("""
                        INSERT OR IGNORE INTO watchlist_recipients (watchlist_id, email)
                        VALUES (?, ?)
                    """, (watchlist_id, email.lower()))
                    added += cursor.rowcount
                
                cursor.execute("""
                    UPDATE watchlists SET updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (watchlist_id,))
        
        logger.info(f"Added {added} recipients to watchlist {watchlist_id}")
        return added
    
    @performance_monitor("get_watchlists_for_schedule")
    def get_watchlists_for_schedule(self, schedule: str = "daily") -> List[Watchlist]:
        """Get watchlists that match a specific schedule"""
        watchlists = []
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id FROM watchlists
                WHERE analysis_schedule = ? AND is_active = 1
                ORDER BY name
            """, (schedule,))
            
            for row in cursor.fetchall():
                watchlist = self.get_watchlist(watchlist_id=row[0])
                if watchlist:
                    watchlists.append(watchlist)
        
        return watchlists
    
    @performance_monitor("should_analyze_watchlist")
    def should_analyze_watchlist(self, watchlist: Watchlist) -> bool:
        """Check if a watchlist is due for analysis based on schedule"""
        if not watchlist.is_active:
            return False
        
        if not watchlist.last_analysis:
            return True
        
        last_analysis = datetime.fromisoformat(watchlist.last_analysis)
        now = datetime.now()
        
        if watchlist.analysis_schedule == "daily":
            return (now - last_analysis).days >= 1
        elif watchlist.analysis_schedule == "weekly":
            return (now - last_analysis).days >= 7
        elif watchlist.analysis_schedule == "monthly":
            return (now - last_analysis).days >= 30
        else:
            return False
    
    @performance_monitor("record_analysis")
    def record_analysis(self, watchlist_id: int, analysis_results: Dict[str, Any],
                       report_file: str = None, email_sent: bool = False) -> int:
        """Record analysis history"""
        with self.db.get_connection() as conn:
            with self.db.transaction(conn) as cursor:
                # Update last analysis date
                cursor.execute("""
                    UPDATE watchlists 
                    SET last_analysis = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (watchlist_id,))
                
                # Get recipients for history
                cursor.execute("""
                    SELECT email FROM watchlist_recipients
                    WHERE watchlist_id = ? AND is_active = 1
                """, (watchlist_id,))
                recipients = [r[0] for r in cursor.fetchall()]
                
                # Insert history record
                cursor.execute("""
                    INSERT INTO watchlist_analysis_history
                    (watchlist_id, analysis_date, total_symbols, buy_signals, 
                     sell_signals, hold_signals, report_file, email_sent, email_recipients)
                    VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    watchlist_id,
                    analysis_results.get('total_symbols', 0),
                    analysis_results.get('buy_signals', 0),
                    analysis_results.get('sell_signals', 0),
                    analysis_results.get('hold_signals', 0),
                    report_file,
                    email_sent,
                    json.dumps(recipients)
                ))
                
                history_id = cursor.lastrowid
                logger.info(f"Recorded analysis history for watchlist {watchlist_id}")
                return history_id
    
    @performance_monitor("import_watchlist_from_file")
    def import_watchlist_from_file(self, file_path: str, name: str = None,
                                  recipients: List[str] = None) -> Optional[int]:
        """Import watchlist from a text file (one symbol per line)"""
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"Watchlist file not found: {file_path}")
            return None
        
        symbols = []
        
        # Read symbols from file
        with open(path, 'r') as f:
            for line in f:
                symbol = line.strip().upper()
                if symbol and not symbol.startswith('#'):  # Skip comments
                    symbols.append(symbol)
        
        if not symbols:
            logger.warning(f"No symbols found in file: {file_path}")
            return None
        
        # Create watchlist name from filename if not provided
        if not name:
            name = path.stem.replace('_', ' ').title()
        
        # Create watchlist
        watchlist = Watchlist(
            name=name,
            description=f"Imported from {path.name}",
            symbols=symbols,
            email_recipients=recipients or [],
            analysis_schedule="daily"
        )
        
        watchlist_id = self.create_watchlist(watchlist)
        logger.info(f"Imported watchlist '{name}' with {len(symbols)} symbols from {file_path}")
        
        return watchlist_id
    
    @performance_monitor("export_watchlist_to_file")
    def export_watchlist_to_file(self, watchlist_id: int, file_path: str) -> bool:
        """Export watchlist symbols to a text file"""
        watchlist = self.get_watchlist(watchlist_id)
        
        if not watchlist:
            logger.error(f"Watchlist {watchlist_id} not found")
            return False
        
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            f.write(f"# Watchlist: {watchlist.name}\n")
            f.write(f"# Description: {watchlist.description}\n")
            f.write(f"# Exported: {datetime.now().isoformat()}\n")
            f.write(f"# Symbols: {len(watchlist.symbols)}\n\n")
            
            for symbol in sorted(watchlist.symbols):
                f.write(f"{symbol}\n")
        
        logger.info(f"Exported watchlist '{watchlist.name}' to {file_path}")
        return True

# ===============================================================================
# EMAIL NOTIFICATION SYSTEM
# ===============================================================================

class EmailNotifier:
    """Handles email notifications for watchlist analysis reports"""
    
    def __init__(self):
        self.config = EmailConfig()
        
    def send_analysis_report(self, watchlist: Watchlist, report_files: Dict[str, str],
                            analysis_summary: Dict[str, Any]) -> bool:
        """Send analysis report via email to watchlist recipients"""
        
        if not self.config.enabled:
            logger.info("Email notifications are disabled")
            return False
        
        if not self.config.is_configured():
            logger.error("Email is not properly configured")
            return False
        
        if not watchlist.email_recipients:
            logger.info(f"No email recipients for watchlist '{watchlist.name}'")
            return False
        
        try:
            # Create email message
            msg = MIMEMultipart('mixed')
            msg['Subject'] = f"Stock Analysis Report: {watchlist.name} - {datetime.now().strftime('%Y-%m-%d')}"
            msg['From'] = f"{self.config.from_name} <{self.config.from_email}>"
            msg['To'] = ', '.join(watchlist.email_recipients)
            
            # Create HTML body
            html_body = self._create_html_report(watchlist, analysis_summary)
            msg.attach(MIMEText(html_body, 'html'))
            
            # Attach report files
            for file_type, file_path in report_files.items():
                if file_path and Path(file_path).exists():
                    self._attach_file(msg, file_path, file_type)
            
            # Send email
            success = self._send_email(msg, watchlist.email_recipients)
            
            if success:
                logger.info(f"Sent analysis report for '{watchlist.name}' to {len(watchlist.email_recipients)} recipients")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send analysis report: {e}")
            return False
    
    def _create_html_report(self, watchlist: Watchlist, summary: Dict[str, Any]) -> str:
        """Create HTML email body with analysis summary"""
        
        buy_signals = summary.get('buy_signals', [])
        top_performers = summary.get('top_performers', [])
        patterns = summary.get('pattern_matches', [])
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                .buy {{ color: green; font-weight: bold; }}
                .sell {{ color: red; font-weight: bold; }}
                .hold {{ color: orange; }}
                .summary-box {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .footer {{ color: #7f8c8d; font-size: 12px; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <h1>📊 Stock Analysis Report: {watchlist.name}</h1>
            <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            <p><strong>Description:</strong> {watchlist.description or 'N/A'}</p>
            
            <div class="summary-box">
                <h2>Summary</h2>
                <ul>
                    <li><strong>Total Stocks Analyzed:</strong> {summary.get('total_analyzed', 0)}</li>
                    <li><strong>Buy Signals:</strong> <span class="buy">{summary.get('buy_count', 0)}</span></li>
                    <li><strong>Sell Signals:</strong> <span class="sell">{summary.get('sell_count', 0)}</span></li>
                    <li><strong>Hold Signals:</strong> <span class="hold">{summary.get('hold_count', 0)}</span></li>
                    <li><strong>Pattern Matches:</strong> {summary.get('pattern_count', 0)}</li>
                </ul>
            </div>
        """
        
        # Add buy signals table
        if buy_signals:
            html += """
            <h2>🟢 Top Buy Signals</h2>
            <table>
                <tr>
                    <th>Symbol</th>
                    <th>RSI</th>
                    <th>ROE %</th>
                    <th>EPS YoY %</th>
                    <th>Return %</th>
                    <th>Patterns</th>
                </tr>
            """
            
            for signal in buy_signals[:10]:  # Top 10
                patterns_str = []
                if signal.get('three_up_pattern'):
                    patterns_str.append('Three-Up')
                if signal.get('cup_handle_valid'):
                    patterns_str.append('Cup&Handle')
                
                html += f"""
                <tr>
                    <td><strong>{signal['symbol']}</strong></td>
                    <td>{signal.get('rsi', 'N/A'):.1f if signal.get('rsi') else 'N/A'}</td>
                    <td>{signal.get('roe_pct', 'N/A'):.1f if signal.get('roe_pct') else 'N/A'}</td>
                    <td>{signal.get('eps_yoy_pct', 'N/A'):.1f if signal.get('eps_yoy_pct') else 'N/A'}</td>
                    <td>{signal.get('period_return_pct', 'N/A'):.1f if signal.get('period_return_pct') else 'N/A'}</td>
                    <td>{', '.join(patterns_str) or 'None'}</td>
                </tr>
                """
            
            html += "</table>"
        
        # Add pattern matches
        if patterns:
            html += """
            <h2>📈 Pattern Matches</h2>
            <table>
                <tr>
                    <th>Symbol</th>
                    <th>Pattern</th>
                    <th>Pivot Price</th>
                    <th>Current Price</th>
                    <th>Potential %</th>
                </tr>
            """
            
            for pattern in patterns[:10]:  # Top 10
                potential = 0
                if pattern.get('pivot_price') and pattern.get('current_price'):
                    potential = ((pattern['pivot_price'] - pattern['current_price']) / pattern['current_price']) * 100
                
                html += f"""
                <tr>
                    <td><strong>{pattern['symbol']}</strong></td>
                    <td>{pattern.get('pattern_type', 'N/A')}</td>
                    <td>${pattern.get('pivot_price', 0):.2f}</td>
                    <td>${pattern.get('current_price', 0):.2f}</td>
                    <td>{potential:.1f}%</td>
                </tr>
                """
            
            html += "</table>"
        
        # Add footer
        html += f"""
            <div class="footer">
                <hr>
                <p><em>This report was automatically generated by the Stock Analysis System.</em></p>
                <p><em>Full detailed reports are attached to this email.</em></p>
                <p><em>Disclaimer: This analysis is for informational purposes only and should not be considered as investment advice.</em></p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _attach_file(self, msg: MIMEMultipart, file_path: str, file_type: str):
        """Attach a file to the email message"""
        try:
            with open(file_path, 'rb') as f:
                attachment = MIMEApplication(f.read())
                
            filename = Path(file_path).name
            attachment.add_header('Content-Disposition', 'attachment', filename=filename)
            msg.attach(attachment)
            
            logger.debug(f"Attached {file_type} file: {filename}")
            
        except Exception as e:
            logger.warning(f"Failed to attach {file_type} file: {e}")
    
    def _send_email(self, msg: MIMEMultipart, recipients: List[str]) -> bool:
        """Send email using SMTP"""
        try:
            # Create SMTP connection
            if self.config.use_tls:
                context = ssl.create_default_context()
                server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
                server.starttls(context=context)
            else:
                server = smtplib.SMTP_SSL(self.config.smtp_server, self.config.smtp_port)
            
            # Login and send
            server.login(self.config.smtp_username, self.config.smtp_password)
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

# ===============================================================================
# WATCHLIST FILE SCANNER
# ===============================================================================

class WatchlistFileScanner:
    """Scans directories for watchlist files and imports them"""
    
    def __init__(self, watchlist_dir: str = None):
        self.watchlist_dir = Path(watchlist_dir) if watchlist_dir else DATA_DIR / "watchlists"
        self.manager = WatchlistManager()
        
    def scan_and_import(self, recipients: List[str] = None) -> Dict[str, int]:
        """Scan directory for watchlist files and import them"""
        if not self.watchlist_dir.exists():
            logger.warning(f"Watchlist directory does not exist: {self.watchlist_dir}")
            return {}
        
        imported = {}
        
        # Scan for text files
        for file_path in self.watchlist_dir.glob("*.txt"):
            name = file_path.stem.replace('_', ' ').title()
            
            # Check if watchlist already exists
            existing = self.manager.get_watchlist(name=name)
            if existing:
                logger.info(f"Watchlist '{name}' already exists, updating...")
                # Update symbols
                symbols = self._read_symbols_from_file(file_path)
                if symbols:
                    existing.symbols = symbols
                    if recipients and not existing.email_recipients:
                        existing.email_recipients = recipients
                    self.manager.update_watchlist(existing)
                    imported[name] = existing.id
            else:
                # Import new watchlist
                watchlist_id = self.manager.import_watchlist_from_file(
                    str(file_path), name, recipients
                )
                if watchlist_id:
                    imported[name] = watchlist_id
        
        logger.info(f"Imported/updated {len(imported)} watchlists from {self.watchlist_dir}")
        return imported
    
    def _read_symbols_from_file(self, file_path: Path) -> List[str]:
        """Read symbols from a file"""
        symbols = []
        
        with open(file_path, 'r') as f:
            for line in f:
                symbol = line.strip().upper()
                if symbol and not symbol.startswith('#'):
                    symbols.append(symbol)
        
        return symbols

# ===============================================================================
# PUBLIC API
# ===============================================================================

def get_watchlist_manager() -> WatchlistManager:
    """Get watchlist manager instance"""
    return WatchlistManager()

def get_email_notifier() -> EmailNotifier:
    """Get email notifier instance"""
    return EmailNotifier()

def scan_watchlist_files(directory: str = None, recipients: List[str] = None) -> Dict[str, int]:
    """Scan and import watchlist files from directory"""
    scanner = WatchlistFileScanner(directory)
    return scanner.scan_and_import(recipients)

def analyze_watchlist(watchlist_name: str, send_email: bool = True) -> bool:
    """Analyze a specific watchlist and optionally send email report"""
    from technical_analysis import analyze_watchlist as tech_analyze
    
    manager = get_watchlist_manager()
    watchlist = manager.get_watchlist(name=watchlist_name)
    
    if not watchlist:
        logger.error(f"Watchlist '{watchlist_name}' not found")
        return False
    
    # Run technical analysis
    success = tech_analyze(watchlist, send_email)
    
    return success

def analyze_all_due_watchlists(schedule: str = None) -> int:
    """Analyze all watchlists that are due based on their schedule"""
    from technical_analysis import analyze_watchlist as tech_analyze
    
    manager = get_watchlist_manager()
    
    if schedule:
        watchlists = manager.get_watchlists_for_schedule(schedule)
    else:
        watchlists = manager.get_all_watchlists(active_only=True)
    
    analyzed = 0
    
    for watchlist in watchlists:
        if manager.should_analyze_watchlist(watchlist):
            logger.info(f"Analyzing watchlist '{watchlist.name}'...")
            success = tech_analyze(watchlist, send_email=True)
            if success:
                analyzed += 1
    
    logger.info(f"Analyzed {analyzed} watchlists")
    return analyzed