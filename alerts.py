# alerts.py - Email notification system


__version__ = "1.5.0"
__copyright__ = "Copyright 2025, Richard D. Wissinger"
__author__ = "Richard D. Wissinger"
__email__ = "rick.wissinger@gmail.com"
__license__ = "MIT"  # or your chosen license
__status__ = "Production"

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional
from datetime import datetime
from config import SMTP_SERVER, SMTP_PORT, EMAIL_USER, EMAIL_PASS, ALERT_RECIPIENTS
from logger import setup_logger

logger = setup_logger(__name__)

def send_alert(subject: str, message: str, recipients: Optional[List[str]] = None) -> bool:
    """
    Send email alert for pipeline failures or important events
    """
    if not EMAIL_USER or not EMAIL_PASS:
        logger.warning("Email credentials not configured. Skipping alert.")
        return False
        
    recipients = recipients or ALERT_RECIPIENTS
    if not recipients:
        logger.warning("No alert recipients configured.")
        return False
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = f"Stock Pipeline Alert: {subject}"
        
        # Add body
        body = f"""
        Stock Data Pipeline Alert
        =========================
        
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Subject: {subject}
        
        Details:
        {message}
        
        This is an automated message from your stock data pipeline.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Connect and send
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls(context=context)
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
            
        logger.info(f"Alert sent successfully to {len(recipients)} recipients")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send alert: {e}")
        return False