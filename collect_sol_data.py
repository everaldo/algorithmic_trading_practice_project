#!/usr/bin/env python3
"""
SOL-BRL Data Collection for Trading Exercise
==========================================

Collect exactly 2,016 bars (1 week of 5min data) for SOL-BRL.
Based on candle_downloader.py from practice_cpf_program.

Author: Everaldo Gomes  
Date: 2025-08-08
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import sqlite3
from typing import Optional

# Add mercado-bitcoin-python to path
sys.path.append('/home/everaldo/Code/mercado-bitcoin-python')
from mercado_bitcoin_python import MercadoBitcoinClient
from mercado_bitcoin_python.api.client import MBConfig

class SOLDataCollector:
    """Simplified data collector for SOL-BRL exercise data."""
    
    def __init__(self, api_config: dict = None):
        """
        Initialize collector.
        
        Args:
            api_config: Dict with api_key and api_secret (optional)
        """
        self.asset_symbol = "SOL-BRL"
        self.timeframe = "5m"
        self.target_bars = 2016  # 1 week = 5min * 12/h * 24h * 7d
        
        # Data storage
        self.data_path = Path("data/exercise")
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize API (if config available)
        self.api = None
        if api_config and api_config.get('api_key') and api_config.get('api_secret'):
            try:
                # Create MBConfig
                config = MBConfig(
                    api_key=api_config['api_key'],
                    api_secret=api_config['api_secret'],
                    api_key_type='read_only',
                    safe_mode=True,
                    enable_cache=True
                )
                
                self.api = MercadoBitcoinClient(config)
                print(f"✅ Connected to Mercado Bitcoin API")
            except Exception as e:
                print(f"⚠️  API connection failed: {e}")
                print("Will create synthetic data instead")
        else:
            print("ℹ️  No API config provided, will create synthetic data")
    
    def calculate_date_range(self) -> tuple[datetime, datetime]:
        """
        Calculate start/end dates for exactly 2,016 bars.
        From current time (2025-08-08) going back 1 week.
        """
        end_date = datetime(2025, 8, 8, 12, 0)  # Today at noon
        start_date = end_date - timedelta(minutes=5 * self.target_bars)
        
        print(f"📅 Target period: {start_date} → {end_date}")
        print(f"📊 Target bars: {self.target_bars} (5min intervals)")
        
        return start_date, end_date
    
    def collect_real_data(self) -> Optional[pd.DataFrame]:
        """Collect real SOL-BRL data from API."""
        if not self.api:
            return None
            
        try:
            start_date, end_date = self.calculate_date_range()
            
            print(f"🔄 Collecting {self.asset_symbol} data...")
            print(f"   Timeframe: {self.timeframe}")
            print(f"   Period: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
            
            # Use get_candles method (native API)
            data = self.api.get_candles(
                symbol=self.asset_symbol,
                timeframe=self.timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if data is None or data.empty:
                print("❌ No data returned from API")
                return None
            
            # Standardize column names
            if 'timestamp' in data.columns and 'timestamp' != data.index.name:
                data.set_index('timestamp', inplace=True)
            
            # Ensure we have OHLCV columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                print(f"❌ Missing columns: {missing_cols}")
                return None
            
            print(f"✅ Collected {len(data)} bars of real {self.asset_symbol} data")
            print(f"   Date range: {data.index[0]} → {data.index[-1]}")
            print(f"   Price range: {data['close'].min():.2f} - {data['close'].max():.2f} BRL")
            
            return data
            
        except Exception as e:
            print(f"❌ Error collecting real data: {e}")
            return None
    
    def create_synthetic_data(self) -> pd.DataFrame:
        """Create synthetic SOL-BRL data as fallback."""
        print("🎭 Creating synthetic SOL-BRL data...")
        
        start_date, end_date = self.calculate_date_range()
        
        # Create time index
        dates = pd.date_range(start_date, end_date, freq='5min')[:-1]  # Remove last to get exact count
        
        # SOL realistic price movement (around 150 BRL)
        base_price = 150.0
        np.random.seed(42)  # Reproducible
        
        # Generate more realistic price movement
        n = len(dates)
        
        # Trend component (gradual increase)
        trend = np.linspace(0, 0.3, n)  # 30% increase over week
        
        # Random walk component  
        random_walk = np.cumsum(np.random.normal(0, 0.002, n))  # 0.2% std per 5min
        
        # Cyclical component (daily patterns)
        daily_cycle = np.sin(2 * np.pi * np.arange(n) / (12 * 24)) * 0.05  # 5% daily oscillation
        
        # Combine components
        log_prices = np.log(base_price) + trend + random_walk + daily_cycle
        prices = np.exp(log_prices)
        
        # Generate OHLC with realistic spreads
        spread = 0.001  # 0.1% typical spread
        
        opens = prices * (1 + np.random.normal(0, spread/2, n))
        highs = prices * (1 + np.abs(np.random.normal(0, spread, n)))
        lows = prices * (1 - np.abs(np.random.normal(0, spread, n)))
        closes = prices
        
        # Ensure OHLC logic: low <= open,close <= high
        lows = np.minimum(lows, np.minimum(opens, closes))
        highs = np.maximum(highs, np.maximum(opens, closes))
        
        # Generate volume (correlated with volatility)
        volatility = np.abs(np.diff(log_prices, prepend=log_prices[0]))
        volumes = 1000 + volatility * 100000 + np.random.exponential(2000, n)
        
        data = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes.astype(int)
        }, index=dates)
        
        data.index.name = 'timestamp'
        
        print(f"✅ Created {len(data)} synthetic bars")
        print(f"   Date range: {data.index[0]} → {data.index[-1]}")
        print(f"   Price range: {data['close'].min():.2f} - {data['close'].max():.2f} BRL")
        print(f"   Volume range: {data['volume'].min():,} - {data['volume'].max():,}")
        
        return data
    
    def save_to_sqlite(self, data: pd.DataFrame) -> Path:
        """Save data to SQLite database."""
        db_path = self.data_path / "sol_data.db"
        
        with sqlite3.connect(db_path) as conn:
            # Save data
            data.to_sql("sol_bars", conn, if_exists='replace', index=True)
            
            # Create index for fast timestamp queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON sol_bars (timestamp)")
            
            # Add metadata table
            metadata = pd.DataFrame([{
                'asset_symbol': self.asset_symbol,
                'timeframe': self.timeframe,
                'total_bars': len(data),
                'start_date': str(data.index[0]),
                'end_date': str(data.index[-1]),
                'collection_date': str(datetime.now()),
                'data_type': 'real' if self.api else 'synthetic'
            }])
            
            metadata.to_sql("metadata", conn, if_exists='replace', index=False)
        
        print(f"💾 Data saved to: {db_path}")
        return db_path
    
    def save_to_csv(self, data: pd.DataFrame) -> Path:
        """Save data to CSV as backup."""
        csv_path = self.data_path / f"{self.asset_symbol.replace('-', '_')}_{self.timeframe}.csv"
        data.to_csv(csv_path)
        print(f"💾 CSV backup saved to: {csv_path}")
        return csv_path
    
    def collect_and_save(self) -> tuple[pd.DataFrame, Path]:
        """Main collection method."""
        print("=" * 60)
        print("SOL-BRL Data Collection for Trading Exercise")
        print("=" * 60)
        
        # Try to collect real data first
        data = self.collect_real_data()
        
        # Fallback to synthetic if real data fails
        if data is None:
            data = self.create_synthetic_data()
        
        # Validate data
        if len(data) != self.target_bars:
            print(f"⚠️  Got {len(data)} bars, expected {self.target_bars}")
            if len(data) < self.target_bars:
                print("   Using available data (should be sufficient for exercise)")
        
        # Save data
        db_path = self.save_to_sqlite(data)
        csv_path = self.save_to_csv(data)
        
        # Data quality summary
        print("\n📊 Data Quality Summary:")
        print(f"   Total bars: {len(data):,}")
        print(f"   Missing values: {data.isnull().sum().sum()}")
        print(f"   Date range: {(data.index[-1] - data.index[0]).days} days")
        print(f"   Price volatility: {(data['close'].std() / data['close'].mean() * 100):.2f}%")
        print(f"   Average volume: {data['volume'].mean():,.0f}")
        
        print("\n✅ Data collection completed!")
        print(f"Ready for ML training with {len(data)} bars")
        
        return data, db_path

def load_collected_data(db_path: str = "data/exercise/sol_data.db") -> pd.DataFrame:
    """Load previously collected data."""
    with sqlite3.connect(db_path) as conn:
        data = pd.read_sql("SELECT * FROM sol_bars", conn, index_col='timestamp', parse_dates=['timestamp'])
    
    print(f"📈 Loaded {len(data)} bars from {db_path}")
    return data

if __name__ == "__main__":
    # Example usage
    collector = SOLDataCollector()
    data, db_path = collector.collect_and_save()
    
    # Test loading
    print("\n" + "=" * 40)
    print("Testing data loading...")
    loaded_data = load_collected_data(str(db_path))
    print(f"Loaded data matches: {len(data) == len(loaded_data)}")
    
    print("\nData collection script completed successfully!")