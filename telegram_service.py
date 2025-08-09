"""
Simple Telegram Service for Intelligent Trading System
"""

import asyncio
import aiohttp
import structlog
from datetime import datetime
from typing import Optional, Dict, Any
from config import TradingConfig

logger = structlog.get_logger()


class TelegramService:
    """Simple Telegram service for notifications."""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.enabled = config.telegram_enabled and config.telegram_bot_token
        self.bot_token = config.telegram_bot_token
        self.chat_id = config.telegram_chat_id
        self.authorized_users = [u.strip() for u in config.telegram_authorized_users.split(',') if u.strip()]
        
        self.logger = logger.bind(service="telegram")
        
        if self.enabled:
            self.logger.info("Telegram service enabled", 
                           has_chat_id=bool(self.chat_id),
                           authorized_users_count=len(self.authorized_users))
        else:
            self.logger.info("Telegram service disabled")
    
    async def send_message(self, text: str, chat_id: Optional[str] = None, 
                          parse_mode: str = "HTML") -> bool:
        """Send message to Telegram."""
        if not self.enabled:
            self.logger.debug("Telegram disabled - message not sent")
            return False
        
        target_chat = chat_id or self.chat_id
        if not target_chat:
            self.logger.error("No chat ID configured")
            return False
        
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": target_chat,
            "text": text,
            "parse_mode": parse_mode
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("ok"):
                            self.logger.debug("Message sent successfully")
                            return True
                        else:
                            self.logger.error("Telegram API error", 
                                            description=result.get("description"))
                            return False
                    else:
                        self.logger.error("HTTP error sending message", 
                                        status=response.status)
                        return False
        
        except Exception as e:
            self.logger.error("Exception sending message", error=str(e))
            return False
    
    async def send_session_start(self, account_summary: Dict[str, Any], session_info: Dict[str, Any]):
        """Send trading session start notification."""
        if not self.enabled:
            return False
        
        message = f"""🚀 <b>INTELLIGENT TRADING SESSION STARTED</b>

📊 <b>Account Information:</b>
• Account ID: {account_summary.get('account_id', 'N/A')[:8]}...
• Tier: {account_summary.get('tier', 'N/A')}
• Symbol: {account_summary.get('symbol', 'SOL-BRL')}

💰 <b>Trading Fees (SOL-BRL):</b>
• Maker: {account_summary.get('fees', {}).get('maker', 0):.4f}%
• Taker: {account_summary.get('fees', {}).get('taker', 0):.4f}%
• Total Cycle: {account_summary.get('fees', {}).get('total_cycle', 0):.4f}%

💵 <b>Initial Balances:</b>
• SOL: {account_summary.get('balances', {}).get('SOL', 0):.6f}
• BRL: R$ {account_summary.get('balances', {}).get('BRL', 0):.2f}

🎯 <b>Session Config:</b>
• Budget: R$ {self.config.total_budget:.2f}
• Max positions: {self.config.max_positions} × R$ {self.config.position_size:.2f}
• Strategy: ML Ensemble (60% Tree + 40% MLP)
• Anti-loss protection: ON

⏰ <b>Started:</b> {session_info.get('start_time', datetime.now()).strftime('%H:%M:%S')}

#TradingSession #SOL #MLStrategy"""
        
        return await self.send_message(message)
    
    async def send_signal_alert(self, signal: Dict[str, Any], position_info: Dict[str, Any]):
        """Send trading signal notification."""
        if not self.enabled:
            return False
        
        signal_type = signal['signal_type'].upper()
        emoji = "🟢" if signal_type == "BUY" else "🔴"
        
        message = f"""{emoji} <b>{signal_type} SIGNAL</b>

💰 <b>SOL-BRL:</b> R$ {signal['price']:.2f}
📊 <b>Confidence:</b> {signal['confidence']:.1%}
🤖 <b>ML Predictions:</b>
• Tree: {signal.get('predictions', {}).get('tree', 0):.3f}
• MLP: {signal.get('predictions', {}).get('mlp', 0):.3f}
• Final: {signal.get('predictions', {}).get('final', 0):.3f}

💵 <b>Position:</b> R$ {position_info['position_size']:.2f}
📈 <b>Quantity:</b> {position_info['quantity']:.6f} SOL

⏰ {datetime.now().strftime('%H:%M:%S')}

#Signal #{signal_type} #SOL"""
        
        return await self.send_message(message)
    
    async def send_trade_execution(self, trade_result: Dict[str, Any], is_profitable: Optional[bool] = None):
        """Send trade execution notification."""
        if not self.enabled:
            return False
        
        side = trade_result['side'].upper()
        emoji = "✅" if trade_result['status'] == 'filled' else "⚠️"
        
        profit_info = ""
        if is_profitable is not None:
            profit_emoji = "📈" if is_profitable else "📉"
            profit_info = f"\n{profit_emoji} <b>Profitable:</b> {'Yes' if is_profitable else 'No (Protected)'}"
        
        message = f"""{emoji} <b>TRADE EXECUTED</b>

📊 <b>Order:</b> {side} SOL-BRL
💰 <b>Price:</b> R$ {trade_result['price']:.2f}
📈 <b>Quantity:</b> {trade_result['quantity']:.6f} SOL
💵 <b>Value:</b> R$ {trade_result['value']:.2f}
💸 <b>Fees:</b> R$ {trade_result['fees']:.2f}{profit_info}

🔖 <b>Order ID:</b> {trade_result['order_id'][:12]}...
⏰ {datetime.now().strftime('%H:%M:%S')}

#Trade #{side} #Executed"""
        
        return await self.send_message(message)
    
    async def send_protection_alert(self, position: Dict[str, Any], current_price: float, reason: str):
        """Send anti-loss protection alert."""
        if not self.enabled:
            return False
        
        message = f"""🛡️ <b>ANTI-LOSS PROTECTION</b>

Sell signal rejected to prevent loss:
• Entry: R$ {position['entry_price']:.2f}
• Current: R$ {current_price:.2f}
• Breakeven: R$ {position.get('breakeven_price', 0):.2f}

📝 <b>Reason:</b> {reason}

⏰ {datetime.now().strftime('%H:%M:%S')}

#Protection #NoLoss #SOL"""
        
        return await self.send_message(message)
    
    async def send_session_end(self, session_summary: Dict[str, Any], final_balances: Dict[str, float]):
        """Send trading session end notification."""
        if not self.enabled:
            return False
        
        profit_emoji = "📈" if session_summary.get('net_pnl', 0) >= 0 else "📉"
        
        message = f"""🏁 <b>TRADING SESSION ENDED</b>

📊 <b>Session Summary:</b>
• Duration: {session_summary.get('duration_hours', 0):.1f}h
• Total trades: {session_summary.get('total_trades', 0)}
• Win rate: {session_summary.get('win_rate', 0):.1f}%

{profit_emoji} <b>Performance:</b>
• Initial: R$ {session_summary.get('initial_balance', 0):.2f}
• Final: R$ {session_summary.get('final_balance', 0):.2f}
• P&L: R$ {session_summary.get('net_pnl', 0):+.2f} ({session_summary.get('roi_pct', 0):+.2f}%)
• Fees paid: R$ {session_summary.get('total_fees_paid', 0):.2f}

💰 <b>Final Balances:</b>
• SOL: {final_balances.get('SOL', 0):.6f}
• BRL: R$ {final_balances.get('BRL', 0):.2f}

⏰ <b>Ended:</b> {session_summary.get('end_time', datetime.now()).strftime('%H:%M:%S')}

#TradingSession #Complete #Summary"""
        
        return await self.send_message(message)