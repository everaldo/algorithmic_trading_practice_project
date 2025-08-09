#!/usr/bin/env python3
"""
Intelligent Trading System with Anti-Loss Protection
==================================================

Features:
- Smart account info collection (accountId, tier, fees)
- Anti-loss protection (don't sell at loss considering fees)
- Telegram integration for real-time notifications
- Budget management (R$100 split into max 5 positions)
- Comprehensive logging at start, during trades, and end

Author: Everaldo Gomes
Date: 2025-08-08
"""

import asyncio
import sys
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import structlog
import pandas as pd

# Import from local wheel
from mercado_bitcoin_python import MBTpqoa
from mercado_bitcoin_python.api.trading_endpoints import AccountInfo, TradingFees, Balance

# Local imports
from config import TradingConfig
from telegram_service import TelegramService
from simple_trading_exercise import SimpleMLEnsembleStrategy, TradingConfig as MLTradingConfig
from collect_sol_data import load_collected_data

# Configure structured logging
def configure_logging():
    """Configure logging: traditional format for console/file."""
    from pathlib import Path
    import logging as std_logging
    from loguru import logger as loguru_logger
    import sys
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Traditional format for console and main log file
    traditional_format = "%(asctime)s [%(levelname)-8s] %(message)s"
    
    # Configure standard logging - BOTH console and file  
    file_handler = std_logging.FileHandler(logs_dir / "intelligent_trading.log")
    file_handler.setFormatter(std_logging.Formatter(traditional_format, datefmt="%Y-%m-%d %H:%M:%S"))
    file_handler.setLevel(std_logging.DEBUG)  # Keep all levels in file
    
    console_handler = std_logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(std_logging.Formatter(traditional_format, datefmt="%Y-%m-%d %H:%M:%S"))
    console_handler.setLevel(std_logging.INFO)  # Only INFO+ in console
    
    root_logger = std_logging.getLogger()
    root_logger.setLevel(std_logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Configure loguru (used by mercado_bitcoin_python) to also log to our files
    # Keep default handler for nice colored console output - set to INFO level
    loguru_logger.configure(handlers=[
        {"sink": sys.stdout, "level": "INFO", "colorize": True},  # Console: INFO+ only
        {"sink": logs_dir / "intelligent_trading.log", "level": "DEBUG", "format": "{time:YYYY-MM-DD HH:mm:ss} [{level:<8}] {message}"}  # File: all levels
    ])
    
    # Configure structlog - colorful console renderer (no JSON)
    structlog.configure(
        processors=[
            structlog.dev.ConsoleRenderer(colors=True)  # Colorful format like loguru, no JSON
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),  # Direct printing
        cache_logger_on_first_use=True,
    )

configure_logging()
logging = structlog.get_logger()


@dataclass
class TradingPosition:
    """Track individual trading position."""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    entry_price: float
    entry_time: datetime
    fees_paid: float
    order_id: str
    
    @property
    def entry_value(self) -> float:
        """Total value invested including fees."""
        return (self.quantity * self.entry_price) + self.fees_paid
    
    def calculate_breakeven_price(self, exit_fees_pct: float) -> float:
        """Calculate minimum price to breakeven on exit."""
        # Need to recover entry value after paying exit fees
        return self.entry_value / (self.quantity * (1 - exit_fees_pct / 100))
    
    def calculate_current_pnl(self, current_price: float, exit_fees_pct: float) -> float:
        """Calculate current P&L including exit fees."""
        gross_exit_value = self.quantity * current_price
        exit_fees = gross_exit_value * (exit_fees_pct / 100)
        net_exit_value = gross_exit_value - exit_fees
        
        return net_exit_value - self.entry_value


@dataclass
class TradingSession:
    """Track complete trading session."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    initial_balance: float = 0.0
    final_balance: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_fees_paid: float = 0.0
    positions: List[TradingPosition] = None
    
    def __post_init__(self):
        if self.positions is None:
            self.positions = []
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100
    
    @property
    def net_pnl(self) -> float:
        return self.final_balance - self.initial_balance
    
    @property
    def roi_pct(self) -> float:
        if self.initial_balance == 0:
            return 0.0
        return (self.net_pnl / self.initial_balance) * 100
    
    @property
    def duration_hours(self) -> float:
        if not self.end_time:
            return 0.0
        return (self.end_time - self.start_time).total_seconds() / 3600


class SimpleDatabase:
    """Simple SQLite database for session tracking."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()
    
    def _create_tables(self):
        """Create necessary tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT,
                    end_time TEXT,
                    initial_balance REAL,
                    final_balance REAL,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    total_fees_paid REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    symbol TEXT,
                    side TEXT,
                    quantity REAL,
                    entry_price REAL,
                    entry_time TEXT,
                    fees_paid REAL,
                    order_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            """)
    
    def save_session(self, session: TradingSession):
        """Save session to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sessions 
                (session_id, start_time, end_time, initial_balance, final_balance,
                 total_trades, winning_trades, losing_trades, total_fees_paid)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id,
                session.start_time.isoformat(),
                session.end_time.isoformat() if session.end_time else None,
                session.initial_balance,
                session.final_balance,
                session.total_trades,
                session.winning_trades,
                session.losing_trades,
                session.total_fees_paid
            ))


class IntelligentTradingSystem:
    """Main intelligent trading system with anti-loss protection."""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logging.bind(system="intelligent_trading")
        
        # Create directories
        config.create_directories()
        
        # Initialize components
        ml_config = MLTradingConfig(
            asset_symbol=config.asset_symbol,
            total_budget=config.total_budget,
            max_positions=config.max_positions,
            position_size=config.position_size
        )
        self.strategy = SimpleMLEnsembleStrategy(ml_config)
        self.telegram = TelegramService(config)
        self.database = SimpleDatabase(config.database_path)
        
        # Initialize real API client
        from mercado_bitcoin_python.api.client import MercadoBitcoinClient, MBConfig
        
        self.api = None
        self.api_client = None
        
        if config.mb_api_key and config.mb_api_secret:
            try:
                # Initialize the real Mercado Bitcoin client
                mb_config = MBConfig(
                    api_key=config.mb_api_key,
                    api_secret=config.mb_api_secret,
                    api_key_type="read_write" if not config.mb_read_only else "read_only",
                    safe_mode=config.mb_read_only,
                    enable_cache=True,
                    cache_dir=f"{config.data_path}/cache"
                )
                
                self.api_client = MercadoBitcoinClient(mb_config)
                self.logger.info("Real API client initialized",
                               safe_mode=mb_config.safe_mode,
                               api_type=mb_config.api_key_type)
                
                # Also create MBTpqoa for compatibility (if needed for some legacy calls)
                try:
                    self.api = MBTpqoa(
                        api_key=config.mb_api_key,
                        api_secret=config.mb_api_secret,
                        read_only=config.mb_read_only
                    )
                except Exception as mbapi_error:
                    self.logger.warning("MBTpqoa initialization failed, using only MercadoBitcoinClient", error=str(mbapi_error))
                    self.api = None
                
            except Exception as e:
                self.logger.error("API initialization failed", error=str(e))
        
        # Trading state
        self.account_id: Optional[str] = None
        self.account_summary: Optional[Dict[str, Any]] = None
        self.active_positions: List[TradingPosition] = []
        self.session: Optional[TradingSession] = None
        self.market_data: Optional[pd.DataFrame] = None
    
    async def initialize_session(self):
        """Initialize trading session with account info collection."""
        self.logger.info("Initializing intelligent trading session")
        
        session_id = f"session_{int(datetime.now().timestamp())}"
        self.session = TradingSession(
            session_id=session_id,
            start_time=datetime.now()
        )
        
        if self.api_client:
            try:
                # Get account information using real API client
                accounts = self.api_client.get_accounts()
                if accounts:
                    self.account_id = accounts[0].account_id
                    self.logger.info("Account ID obtained", account_id=self.account_id[:8] + "...")
                
                # Get comprehensive account summary
                self.account_summary = self.api_client.get_account_summary(
                    self.account_id, 
                    self.config.asset_symbol
                )
                
                # Store initial balance
                self.session.initial_balance = self.account_summary['balances']['BRL']
                
                self.logger.info("Account summary collected",
                               tier=self.account_summary['tier'],
                               maker_fee=self.account_summary['fees']['maker'],
                               taker_fee=self.account_summary['fees']['taker'])
                
            except Exception as e:
                self.logger.error("Failed to collect account info", error=str(e))
                # Create mock summary for testing
                self.account_summary = {
                    'account_id': 'test_account',
                    'tier': 'standard',
                    'symbol': self.config.asset_symbol,
                    'fees': {'maker': 0.25, 'taker': 0.35, 'total_cycle': 1.2},
                    'balances': {'SOL': 0.0, 'BRL': 100.0},
                    'positions': 0
                }
                self.session.initial_balance = 100.0
        else:
            # Mock data for test mode
            self.account_summary = {
                'account_id': 'test_account',
                'tier': 'standard',
                'symbol': self.config.asset_symbol,
                'fees': {'maker': 0.25, 'taker': 0.35, 'total_cycle': 1.2},
                'balances': {'SOL': 0.0, 'BRL': 100.0},
                'positions': 0
            }
            self.session.initial_balance = 100.0
        
        # Load market data and train strategy
        await self._prepare_strategy()
        
        # Print session info
        self._print_session_info()
        
        # Send Telegram notification
        session_info = {
            'start_time': self.session.start_time,
            'session_id': self.session.session_id
        }
        await self.telegram.send_session_start(self.account_summary, session_info)
    
    def _print_session_info(self):
        """Print comprehensive session information."""
        print("=" * 80)
        print("🤖 INTELLIGENT TRADING SYSTEM - SESSION START")
        print("=" * 80)
        print(f"📊 Account Information:")
        print(f"   Account ID: {self.account_summary['account_id'][:12]}...")
        print(f"   Tier: {self.account_summary['tier']}")
        print(f"   Asset: {self.account_summary['symbol']}")
        
        print(f"\n💰 Trading Fees ({self.account_summary['symbol']}):")
        print(f"   Maker Fee: {self.account_summary['fees']['maker']:.4f}%")
        print(f"   Taker Fee: {self.account_summary['fees']['taker']:.4f}%")
        print(f"   Total Cycle (Buy+Sell): {self.account_summary['fees']['total_cycle']:.4f}%")
        
        print(f"\n💵 Initial Balances:")
        print(f"   SOL: {self.account_summary['balances']['SOL']:.6f}")
        print(f"   BRL: R$ {self.account_summary['balances']['BRL']:.2f}")
        
        print(f"\n🎯 Session Configuration:")
        print(f"   Total Budget: R$ {self.config.total_budget:.2f}")
        print(f"   Max Positions: {self.config.max_positions}")
        print(f"   Position Size: R$ {self.config.position_size:.2f}")
        print(f"   Anti-Loss Protection: ENABLED")
        
        print(f"\n⏰ Session Started: {self.session.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    
    async def _prepare_strategy(self):
        """Load fresh data and prepare ML strategy."""
        self.logger.info("Preparing ML strategy with fresh data")
        
        # Step 1: Update cache with latest data (last week)
        await self._update_market_data_cache()
        
        # Step 2: Load updated market data
        try:
            self.market_data = load_collected_data(f"{self.config.data_path}/exercise/sol_data.db")
            self.logger.info(f"Loaded {len(self.market_data)} bars from updated cache")
        except Exception as e:
            self.logger.warning("Could not load collected data", error=str(e))
            # Create sample data for testing
            dates = pd.date_range(start='2025-08-01', periods=2016, freq='5T')
            self.market_data = pd.DataFrame({
                'timestamp': dates,
                'open': 180.0 + (pd.Series(range(2016)) * 0.01),
                'high': 185.0 + (pd.Series(range(2016)) * 0.01),
                'low': 175.0 + (pd.Series(range(2016)) * 0.01),
                'close': 180.0 + (pd.Series(range(2016)) * 0.01),
                'volume': 1000.0
            })
        
        # Step 3: Always retrain with fresh data
        self.logger.info("Training ML model with fresh weekly data")
        result = self.strategy.train_once(self.market_data)
        if result['status'] != 'success':
            raise Exception(f"Strategy training failed: {result.get('error')}")
        
        self.logger.info("ML strategy ready", 
                        is_trained=self.strategy.is_trained,
                        features=len(self.strategy.feature_columns) if hasattr(self.strategy, 'feature_columns') else 0)
    
    async def _update_market_data_cache(self):
        """Update market data cache with latest candles."""
        try:
            from collect_sol_data import SOLDataCollector
            
            self.logger.info("Updating market data cache...")
            
            # Pass API configuration to collector
            api_config = {
                'api_key': self.config.mb_api_key,
                'api_secret': self.config.mb_api_secret
            } if self.config.mb_api_key and self.config.mb_api_secret else None
            
            collector = SOLDataCollector(api_config)
            
            # Collect fresh data for the last week
            collector.collect_and_save()
            self.logger.info("Market data cache updated successfully")
            
        except Exception as e:
            self.logger.warning("Failed to update cache, using existing data", error=str(e))
    
    def _is_profitable_to_sell(self, position: TradingPosition, current_price: float) -> bool:
        """Check if selling would be profitable considering fees."""
        if not self.account_summary:
            return True  # Allow if no fee info
        
        exit_fee_pct = self.account_summary['fees']['maker']  # Assume maker fee for limit orders
        current_pnl = position.calculate_current_pnl(current_price, exit_fee_pct)
        
        self.logger.info("Profitability check",
                        entry_price=position.entry_price,
                        current_price=current_price,
                        breakeven_price=position.calculate_breakeven_price(exit_fee_pct),
                        current_pnl=current_pnl,
                        is_profitable=current_pnl > 0)
        
        return current_pnl > 0
    
    async def process_signal(self, signal: Dict[str, Any]) -> bool:
        """Process trading signal with anti-loss protection."""
        signal_type = signal['signal_type']
        current_price = signal['price']
        
        self.logger.info("Processing signal", 
                        type=signal_type,
                        price=current_price,
                        confidence=signal['confidence'])
        
        if signal_type == 'buy':
            return await self._process_buy_signal(signal)
        elif signal_type == 'sell':
            return await self._process_sell_signal(signal)
        
        return False
    
    async def _process_buy_signal(self, signal: Dict[str, Any]) -> bool:
        """Process buy signal."""
        # Check if we can open new position
        if len(self.active_positions) >= self.config.max_positions:
            self.logger.warning("Cannot open new position - max positions reached")
            return False
        
        # Calculate position size
        position_value = self.config.position_size
        quantity = position_value / signal['price']
        
        position_info = {
            'position_size': position_value,
            'quantity': quantity
        }
        
        # Send Telegram alert
        await self.telegram.send_signal_alert(signal, position_info)
        
        # Execute trade (or simulate)
        trade_result = await self._execute_buy_order(signal['price'], quantity)
        
        if trade_result['status'] == 'filled':
            # Create position record
            position = TradingPosition(
                symbol=self.config.asset_symbol,
                side='buy',
                quantity=quantity,
                entry_price=trade_result['price'],
                entry_time=datetime.now(),
                fees_paid=trade_result['fees'],
                order_id=trade_result['order_id']
            )
            
            self.active_positions.append(position)
            self.session.total_trades += 1
            self.session.total_fees_paid += trade_result['fees']
            
            # Send execution notification
            await self.telegram.send_trade_execution(trade_result)
            
            self.logger.info("Buy position opened",
                           quantity=quantity,
                           price=trade_result['price'],
                           value=trade_result['value'])
            return True
        
        return False
    
    async def _process_sell_signal(self, signal: Dict[str, Any]) -> bool:
        """Process sell signal with anti-loss protection."""
        if not self.active_positions:
            self.logger.info("No positions to sell")
            return False
        
        # Find position to sell (FIFO)
        position = self.active_positions[0]
        current_price = signal['price']
        
        # ANTI-LOSS PROTECTION: Check profitability
        if not self._is_profitable_to_sell(position, current_price):
            self.logger.warning("ANTI-LOSS PROTECTION: Rejecting unprofitable sell signal",
                              entry_price=position.entry_price,
                              current_price=current_price)
            
            # Send notification about protection
            position_dict = {
                'entry_price': position.entry_price,
                'breakeven_price': position.calculate_breakeven_price(self.account_summary['fees']['maker'])
            }
            reason = f"Would lose R$ {position.calculate_breakeven_price(self.account_summary['fees']['maker']) - current_price:.2f} in fees"
            await self.telegram.send_protection_alert(position_dict, current_price, reason)
            
            return False
        
        # Send signal alert
        position_info = {
            'position_size': position.quantity * current_price,
            'quantity': position.quantity
        }
        await self.telegram.send_signal_alert(signal, position_info)
        
        # Execute sell
        trade_result = await self._execute_sell_order(current_price, position.quantity)
        
        if trade_result['status'] == 'filled':
            # Calculate P&L
            pnl = position.calculate_current_pnl(
                trade_result['price'], 
                self.account_summary['fees']['maker']
            )
            
            # Update session stats
            self.session.total_trades += 1
            self.session.total_fees_paid += trade_result['fees']
            
            if pnl > 0:
                self.session.winning_trades += 1
            else:
                self.session.losing_trades += 1
            
            # Remove position
            self.active_positions.remove(position)
            
            # Send execution notification
            await self.telegram.send_trade_execution(trade_result, is_profitable=pnl > 0)
            
            self.logger.info("Sell position closed",
                           quantity=position.quantity,
                           entry_price=position.entry_price,
                           exit_price=trade_result['price'],
                           pnl=pnl)
            return True
        
        return False
    
    async def _execute_buy_order(self, price: float, quantity: float) -> Dict[str, Any]:
        """Execute buy order (real or simulated)."""
        if self.api_client and self.account_id and not self.config.mode == "test":
            try:
                # Use the real MercadoBitcoinClient
                result = self.api_client.place_market_order(
                    account_id=self.account_id,
                    symbol=self.config.asset_symbol,
                    side='buy',
                    quantity=quantity,
                    dry_run=self.config.mb_read_only
                )
                
                # Convert API response to expected format
                return {
                    'order_id': result.get('orderId', result.get('order_id', f'api_{int(datetime.now().timestamp())}')),
                    'status': 'filled',
                    'side': 'buy',
                    'price': price,  # Market price
                    'quantity': quantity,
                    'value': price * quantity,
                    'fees': (price * quantity) * (self.account_summary['fees']['taker'] / 100)
                }
            except Exception as e:
                self.logger.error("Real buy order failed", error=str(e))
        
        # Simulate order for test mode
        fees = (price * quantity) * (self.account_summary['fees']['taker'] / 100)
        return {
            'order_id': f'sim_buy_{int(datetime.now().timestamp())}',
            'status': 'filled',
            'side': 'buy',
            'price': price,
            'quantity': quantity,
            'value': price * quantity,
            'fees': fees
        }
    
    async def _execute_sell_order(self, price: float, quantity: float) -> Dict[str, Any]:
        """Execute sell order (real or simulated)."""
        if self.api_client and self.account_id and not self.config.mode == "test":
            try:
                # Use the real MercadoBitcoinClient
                result = self.api_client.place_market_order(
                    account_id=self.account_id,
                    symbol=self.config.asset_symbol,
                    side='sell',
                    quantity=quantity,
                    dry_run=self.config.mb_read_only
                )
                
                # Convert API response to expected format
                gross_value = price * quantity
                fees = gross_value * (self.account_summary['fees']['maker'] / 100)
                
                return {
                    'order_id': result.get('orderId', result.get('order_id', f'api_{int(datetime.now().timestamp())}')),
                    'status': 'filled',
                    'side': 'sell',
                    'price': price,  # Market price
                    'quantity': quantity,
                    'value': gross_value,
                    'fees': fees
                }
            except Exception as e:
                self.logger.error("Real sell order failed", error=str(e))
        
        # Simulate order for test mode
        gross_value = price * quantity
        fees = gross_value * (self.account_summary['fees']['maker'] / 100)
        
        return {
            'order_id': f'sim_sell_{int(datetime.now().timestamp())}',
            'status': 'filled',
            'side': 'sell',
            'price': price,
            'quantity': quantity,
            'value': gross_value,
            'fees': fees
        }
    
    async def finalize_session(self):
        """Finalize trading session."""
        if not self.session:
            return
        
        self.session.end_time = datetime.now()
        
        # Force close all positions if at session end to calculate final P&L
        if self.active_positions:
            self.logger.info("Force closing remaining positions")
            
            # Get current price for final valuation
            current_price = self.market_data['close'].iloc[-1] if self.market_data is not None else 200.0
            
            for position in self.active_positions:
                # Check if profitable to sell
                if self._is_profitable_to_sell(position, current_price):
                    trade_result = await self._execute_sell_order(current_price, position.quantity)
                    self.session.total_trades += 1
                    self.session.total_fees_paid += trade_result['fees']
                    
                    pnl = position.calculate_current_pnl(current_price, self.account_summary['fees']['maker'])
                    if pnl > 0:
                        self.session.winning_trades += 1
                    else:
                        self.session.losing_trades += 1
                else:
                    self.logger.warning("FINAL PROTECTION: Not selling losing position",
                                      symbol=position.symbol,
                                      entry_price=position.entry_price,
                                      current_price=current_price)
            
            self.active_positions.clear()
        
        # Calculate final balances (simulated)
        self.session.final_balance = self.session.initial_balance + self.session.net_pnl
        
        final_balances = {
            'SOL': sum(p.quantity for p in self.active_positions),  # Remaining SOL
            'BRL': self.session.final_balance
        }
        
        # Save session to database
        self.database.save_session(self.session)
        
        # Print final summary
        self._print_session_summary()
        
        # Send Telegram summary
        session_summary = {
            'duration_hours': self.session.duration_hours,
            'total_trades': self.session.total_trades,
            'win_rate': self.session.win_rate,
            'initial_balance': self.session.initial_balance,
            'final_balance': self.session.final_balance,
            'net_pnl': self.session.net_pnl,
            'roi_pct': self.session.roi_pct,
            'total_fees_paid': self.session.total_fees_paid,
            'end_time': self.session.end_time
        }
        await self.telegram.send_session_end(session_summary, final_balances)
        
        self.logger.info("Trading session finalized",
                        duration_hours=self.session.duration_hours,
                        total_trades=self.session.total_trades,
                        net_pnl=self.session.net_pnl)
    
    def _print_session_summary(self):
        """Print final session summary."""
        print("\n" + "=" * 80)
        print("🏁 INTELLIGENT TRADING SYSTEM - SESSION COMPLETE")
        print("=" * 80)
        print(f"📊 Session Performance:")
        print(f"   Duration: {self.session.duration_hours:.1f} hours")
        print(f"   Total Trades: {self.session.total_trades}")
        print(f"   Win Rate: {self.session.win_rate:.1f}%")
        
        print(f"\n💰 Financial Summary:")
        print(f"   Initial Balance: R$ {self.session.initial_balance:.2f}")
        print(f"   Final Balance: R$ {self.session.final_balance:.2f}")
        print(f"   Net P&L: R$ {self.session.net_pnl:+.2f} ({self.session.roi_pct:+.2f}%)")
        print(f"   Total Fees Paid: R$ {self.session.total_fees_paid:.2f}")
        
        print(f"\n🛡️ Anti-Loss Protection:")
        print(f"   Unprofitable sells prevented: ON")
        print(f"   Final positions protected: {len(self.active_positions)}")
        
        print(f"\n⏰ Completed: {self.session.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)


async def get_current_price_mock(symbol: str = "SOL-BRL") -> float:
    """Mock function to get current price - replace with real API call."""
    import random
    # Simulate SOL price around 180-220 BRL with volatility
    base_price = 200.0
    volatility = random.uniform(-0.05, 0.05)  # ±5% volatility
    return base_price * (1 + volatility)


async def run_live_trading_session():
    """Run 6-hour live trading session with real-time signal generation."""
    print("🚀 STARTING 6-HOUR LIVE TRADING SESSION")
    print("=" * 80)
    
    config = TradingConfig.from_env(".env")
    system = IntelligentTradingSystem(config)
    
    # Initialize session
    await system.initialize_session()
    
    # Calculate session duration
    session_duration_seconds = config.session_duration_hours * 3600  # 6h = 21600s
    signal_interval_seconds = 300  # Generate signal every 5 minutes (candle period)
    
    print(f"⏰ Session will run for {config.session_duration_hours} hours")
    print(f"📊 Generating signals every {signal_interval_seconds} seconds")
    print(f"🎯 Total expected signals: {session_duration_seconds // signal_interval_seconds}")
    print("=" * 80)
    
    start_time = datetime.now()
    signal_count = 0
    
    try:
        while True:
            current_time = datetime.now()
            elapsed_seconds = (current_time - start_time).total_seconds()
            
            # Check if session duration reached
            if elapsed_seconds >= session_duration_seconds:
                print(f"\n⏰ Session duration ({config.session_duration_hours}h) completed!")
                break
            
            # Get latest 5-minute candle
            try:
                latest_candle = await fetch_latest_candle(system.api_client)
                if not latest_candle:
                    await asyncio.sleep(signal_interval_seconds)
                    continue
                    
                signal_count += 1
                current_price = latest_candle['close']
                
                # Generate ML signal using trained model and latest candle
                signal = await generate_ml_signal(system.strategy, system.market_data, latest_candle, signal_count)
                
                if signal:
                    # Process the signal
                    success = await system.process_signal(signal)
                    
                    remaining_hours = (session_duration_seconds - elapsed_seconds) / 3600
                    print(f"📊 Signal #{signal_count} | "
                          f"Price: R$ {current_price:.2f} | "
                          f"Type: {signal['signal_type']} | "
                          f"Confidence: {signal['confidence']:.1%} | "
                          f"Remaining: {remaining_hours:.1f}h")
                    
                    if success:
                        print(f"✅ Signal executed successfully")
                    else:
                        print(f"⚠️ Signal not executed (protection/limits)")
                
                # Wait before next signal
                await asyncio.sleep(signal_interval_seconds)
                
            except Exception as e:
                print(f"❌ Error in trading loop: {e}")
                await asyncio.sleep(signal_interval_seconds)
                continue
    
    except KeyboardInterrupt:
        print(f"\n🛑 Session interrupted by user after {elapsed_seconds/3600:.1f}h")
    
    finally:
        # Finalize session
        print(f"\n📊 Finalizing session after {signal_count} signals...")
        await system.finalize_session()
        
        print("🏁 LIVE TRADING SESSION COMPLETED")
        print("=" * 80)


async def fetch_latest_candle(api_client, symbol: str = "SOL-BRL") -> Optional[Dict[str, Any]]:
    """Fetch the latest 5-minute candle from the real API."""
    try:
        if api_client:
            # Get the most recent candle using countback=1
            df = api_client.get_candles(
                symbol=symbol,
                timeframe="5m", 
                countback=1,
                use_cache=False  # Always get fresh data
            )
            
            if not df.empty:
                latest = df.iloc[-1]
                return {
                    'timestamp': latest.name,  # Use index as timestamp
                    'open': float(latest['open']),
                    'high': float(latest['high']),
                    'low': float(latest['low']),
                    'close': float(latest['close']),
                    'volume': float(latest['volume'])
                }
        
        # Fallback: simulate realistic candle if no API
        import random
        current_time = datetime.now()
        base_price = 200.0
        volatility = random.uniform(-0.02, 0.02)  # ±2% volatility
        close_price = base_price * (1 + volatility)
        
        # Create realistic OHLCV
        open_price = close_price * (1 + random.uniform(-0.005, 0.005))
        high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.003))
        low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.003))
        volume = random.uniform(800, 1500)
        
        return {
            'timestamp': current_time,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        }
        
    except Exception as e:
        print(f"❌ Error fetching candle: {e}")
        return None


async def generate_ml_signal(strategy, market_data: pd.DataFrame, latest_candle: Dict[str, Any], signal_count: int) -> Optional[Dict[str, Any]]:
    """Generate ML signal using trained model and latest candle."""
    try:
        # Convert latest candle to DataFrame format
        import pandas as pd
        
        candle_df = pd.DataFrame([latest_candle])
        
        # Append to historical data for feature calculation (SMAs need history)
        extended_data = pd.concat([market_data, candle_df], ignore_index=True)
        
        # Use only the latest row for prediction (but calculate features with full history)
        signal_result = strategy.generate_signal(extended_data.tail(1))
        
        if signal_result and signal_result.get('signal_type') != 'hold':
            return {
                'signal_type': signal_result['signal_type'],
                'price': latest_candle['close'],
                'confidence': signal_result.get('confidence', 0.5),
                'predictions': signal_result.get('predictions', {}),
                'reason': f"ML prediction on candle #{signal_count}",
                'timestamp': latest_candle['timestamp']
            }
        
        return None
        
    except Exception as e:
        print(f"❌ Error generating ML signal: {e}")
        return None


# Test function (kept for debugging)
async def test_intelligent_system():
    """Test the intelligent trading system with simulated signals."""
    print("Testing Intelligent Trading System...")
    
    config = TradingConfig.from_env()
    system = IntelligentTradingSystem(config)
    
    # Initialize
    await system.initialize_session()
    
    # Simulate some signals
    test_signals = [
        {
            'signal_type': 'buy',
            'price': 180.0,
            'confidence': 0.85,
            'predictions': {'tree': 0.4, 'mlp': 0.3, 'final': 0.37}
        },
        {
            'signal_type': 'sell', 
            'price': 175.0,  # Loss scenario
            'confidence': 0.80,
            'predictions': {'tree': -0.3, 'mlp': -0.4, 'final': -0.34}
        },
        {
            'signal_type': 'sell',
            'price': 190.0,  # Profit scenario
            'confidence': 0.90,
            'predictions': {'tree': -0.5, 'mlp': -0.2, 'final': -0.38}
        }
    ]
    
    for signal in test_signals:
        await system.process_signal(signal)
        await asyncio.sleep(1)  # Small delay
    
    # Finalize
    await system.finalize_session()
    
    print("Test completed!")


def main():
    """Main entry point - choose between test mode and live trading."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("🧪 Running in TEST mode")
        asyncio.run(test_intelligent_system())
    else:
        print("🔴 Running in LIVE TRADING mode")
        asyncio.run(run_live_trading_session())


if __name__ == "__main__":
    main()
