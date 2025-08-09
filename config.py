"""
Simple Configuration for Intelligent Trading System
"""

import os
from dataclasses import dataclass
from typing import Optional
import structlog
from pathlib import Path

logger = structlog.get_logger()


@dataclass
class TradingConfig:
    """Simple trading configuration from environment variables."""
    # System
    mode: str = "test"
    debug: bool = True
    
    # Trading
    total_budget: float = 100.0
    max_positions: int = 5
    position_size: float = 20.0
    asset_symbol: str = "SOL-BRL"
    
    # API
    mb_api_key: Optional[str] = None
    mb_api_secret: Optional[str] = None
    mb_read_only: bool = True
    
    # Telegram
    telegram_enabled: bool = False
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    telegram_authorized_users: str = ""
    
    # Paths
    database_path: str = "data/trading.db"
    model_path: str = "models/"
    data_path: str = "data/"
    log_path: str = "logs/"
    
    # ML Settings
    training_test_split: float = 0.3
    
    # Session
    session_duration_hours: int = 6
    data_collection_bars: int = 2016
    
    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "TradingConfig":
        """Load configuration from environment variables."""
        # Load .env file if provided
        if env_file and Path(env_file).exists():
            from dotenv import load_dotenv
            load_dotenv(env_file)
        
        return cls(
            mode=os.getenv("MODE", "test"),
            debug=os.getenv("DEBUG", "true").lower() == "true",
            
            total_budget=float(os.getenv("TOTAL_BUDGET", "100.0")),
            max_positions=int(os.getenv("MAX_POSITIONS", "5")),
            position_size=float(os.getenv("POSITION_SIZE", "20.0")),
            asset_symbol=os.getenv("ASSET_SYMBOL", "SOL-BRL"),
            
            mb_api_key=os.getenv("MB_API_KEY"),
            mb_api_secret=os.getenv("MB_API_SECRET"),
            mb_read_only=os.getenv("MB_READ_ONLY", "true").lower() == "true",
            
            telegram_enabled=os.getenv("TELEGRAM_ENABLED", "false").lower() == "true",
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
            telegram_authorized_users=os.getenv("TELEGRAM_AUTHORIZED_USERS", ""),
            
            database_path=os.getenv("DATABASE_PATH", "data/trading.db"),
            model_path=os.getenv("MODEL_PATH", "models/"),
            data_path=os.getenv("DATA_PATH", "data/"),
            log_path=os.getenv("LOG_PATH", "logs/"),
            
            training_test_split=float(os.getenv("TRAINING_TEST_SPLIT", "0.3")),
            
            session_duration_hours=int(os.getenv("SESSION_DURATION_HOURS", "6")),
            data_collection_bars=int(os.getenv("DATA_COLLECTION_BARS", "2016"))
        )
    
    def create_directories(self):
        """Create necessary directories."""
        for path_attr in ['data_path', 'model_path', 'log_path']:
            path = Path(getattr(self, path_attr))
            path.mkdir(parents=True, exist_ok=True)
        
        # Ensure database directory exists
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.total_budget <= 0:
            raise ValueError("Total budget must be positive")
        
        if self.max_positions <= 0:
            raise ValueError("Max positions must be positive")
        
        if self.position_size <= 0:
            raise ValueError("Position size must be positive")
        
        if self.position_size * self.max_positions > self.total_budget:
            raise ValueError("Position size * max positions exceeds total budget")
        
        logger.info("Configuration loaded",
                   mode=self.mode,
                   budget=self.total_budget,
                   positions=self.max_positions,
                   telegram_enabled=self.telegram_enabled)