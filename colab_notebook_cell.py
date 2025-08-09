# 🔐 Complete Environment Setup for Google Colab
# Copy this entire cell to the top of your Jupyter notebook

import os
from pathlib import Path

def setup_complete_environment():
    """
    Setup ALL environment variables for the intelligent trading system
    Handles all 19 variables from .env.example with fallback to demo mode
    """
    
    # Check if we're in Colab
    try:
        from google.colab import userdata
        print("🔐 Google Colab detected - using Colab Secrets...")
        
        try:
            # System Mode
            os.environ['MODE'] = userdata.get('MODE', 'test')
            os.environ['DEBUG'] = userdata.get('DEBUG', 'true')
            
            # Trading Configuration
            os.environ['TOTAL_BUDGET'] = userdata.get('TOTAL_BUDGET', '100.0')
            os.environ['MAX_POSITIONS'] = userdata.get('MAX_POSITIONS', '5')
            os.environ['POSITION_SIZE'] = userdata.get('POSITION_SIZE', '20.0')
            os.environ['ASSET_SYMBOL'] = userdata.get('ASSET_SYMBOL', 'SOL-BRL')
            
            # Mercado Bitcoin API
            os.environ['MB_API_KEY'] = userdata.get('MB_API_KEY')
            os.environ['MB_API_SECRET'] = userdata.get('MB_API_SECRET')
            os.environ['MB_READ_ONLY'] = userdata.get('MB_READ_ONLY', 'true')
            
            # Database
            os.environ['DATABASE_PATH'] = userdata.get('DATABASE_PATH', 'data/trading.db')
            
            # Logging
            os.environ['LOG_LEVEL'] = userdata.get('LOG_LEVEL', 'INFO')
            os.environ['LOG_PATH'] = userdata.get('LOG_PATH', 'logs/')
            
            # Telegram Configuration
            os.environ['TELEGRAM_ENABLED'] = userdata.get('TELEGRAM_ENABLED', 'false')
            os.environ['TELEGRAM_BOT_TOKEN'] = userdata.get('TELEGRAM_BOT_TOKEN', 'demo_token')
            os.environ['TELEGRAM_CHAT_ID'] = userdata.get('TELEGRAM_CHAT_ID', 'demo_chat')
            os.environ['TELEGRAM_AUTHORIZED_USERS'] = userdata.get('TELEGRAM_AUTHORIZED_USERS', 'demo_user')
            
            # ML Strategy Settings
            os.environ['MODEL_PATH'] = userdata.get('MODEL_PATH', 'models/')
            os.environ['DATA_PATH'] = userdata.get('DATA_PATH', 'data/')
            os.environ['TRAINING_TEST_SPLIT'] = userdata.get('TRAINING_TEST_SPLIT', '0.3')
            
            # Session Settings
            os.environ['SESSION_DURATION_HOURS'] = userdata.get('SESSION_DURATION_HOURS', '6')
            os.environ['DATA_COLLECTION_BARS'] = userdata.get('DATA_COLLECTION_BARS', '2016')
            
            print("✅ Successfully loaded from Colab secrets")
            return True
            
        except Exception as e:
            print(f"⚠️  Some Colab secrets missing: {e}")
            
    except ImportError:
        print("📓 Not in Colab - checking for .env...")
        try:
            from dotenv import load_dotenv
            load_dotenv('.env')
            print("✅ Loaded .env file")
            return True
        except:
            pass
    
    # DEMO MODE - All variables with safe defaults
    print("🧪 Setting up DEMO MODE...")
    
    # System Mode
    os.environ['MODE'] = 'demo'
    os.environ['DEBUG'] = 'true'
    
    # Trading Configuration
    os.environ['TOTAL_BUDGET'] = '100.0'
    os.environ['MAX_POSITIONS'] = '5'
    os.environ['POSITION_SIZE'] = '20.0'
    os.environ['ASSET_SYMBOL'] = 'SOL-BRL'
    
    # Mercado Bitcoin API (DEMO)
    os.environ['MB_API_KEY'] = 'demo_api_key_safe_for_sharing'
    os.environ['MB_API_SECRET'] = 'demo_secret_safe_for_sharing'
    os.environ['MB_READ_ONLY'] = 'true'
    
    # Database (Colab-friendly paths)
    os.environ['DATABASE_PATH'] = '/content/data/trading.db'
    
    # Logging
    os.environ['LOG_LEVEL'] = 'INFO'
    os.environ['LOG_PATH'] = '/content/logs/'
    
    # Telegram (disabled in demo)
    os.environ['TELEGRAM_ENABLED'] = 'false'
    os.environ['TELEGRAM_BOT_TOKEN'] = 'demo_bot_token_not_real'
    os.environ['TELEGRAM_CHAT_ID'] = 'demo_chat_id'
    os.environ['TELEGRAM_AUTHORIZED_USERS'] = 'demo_users'
    
    # ML Strategy Settings
    os.environ['MODEL_PATH'] = '/content/models/'
    os.environ['DATA_PATH'] = '/content/data/'
    os.environ['TRAINING_TEST_SPLIT'] = '0.3'
    
    # Session Settings
    os.environ['SESSION_DURATION_HOURS'] = '6'
    os.environ['DATA_COLLECTION_BARS'] = '2016'
    
    print("✅ DEMO environment ready - all API calls will be mocked")
    return False


def create_directories():
    """Create necessary directories for Colab"""
    dirs = [
        Path(os.environ.get('DATA_PATH', '/content/data')),
        Path(os.environ.get('LOG_PATH', '/content/logs')),
        Path(os.environ.get('MODEL_PATH', '/content/models'))
    ]
    
    for directory in dirs:
        directory.mkdir(exist_ok=True, parents=True)
        print(f"📁 Created: {directory}")


def print_config_summary():
    """Print configuration summary"""
    print("\n" + "="*50)
    print("📊 CONFIGURATION SUMMARY")
    print("="*50)
    
    # Key settings
    key_vars = [
        ('MODE', 'System Mode'),
        ('ASSET_SYMBOL', 'Trading Asset'), 
        ('TOTAL_BUDGET', 'Budget'),
        ('MB_READ_ONLY', 'Safety Mode'),
        ('TELEGRAM_ENABLED', 'Notifications'),
        ('LOG_LEVEL', 'Logging'),
        ('SESSION_DURATION_HOURS', 'Session Length')
    ]
    
    for var, desc in key_vars:
        value = os.environ.get(var, 'NOT SET')
        print(f"  {desc}: {value}")
    
    print("="*50)
    
    # Safety notice
    mode = os.environ.get('MODE', 'unknown')
    read_only = os.environ.get('MB_READ_ONLY', 'unknown')
    
    if mode == 'demo':
        print("🧪 DEMO MODE: Safe for sharing, no real trades")
    elif read_only == 'true':
        print("🔒 READ-ONLY: Safe mode, no actual trades")
    else:
        print("⚠️  LIVE MODE: Real trading enabled!")


# MAIN EXECUTION - Run this cell first!
print("🚀 INTELLIGENT TRADING SYSTEM - Environment Setup")
print("="*60)

# Setup environment
real_mode = setup_complete_environment()

# Create directories
create_directories()

# Print summary
print_config_summary()

# Configure asyncio for notebooks
import nest_asyncio
nest_asyncio.apply()

print(f"\n✅ Environment ready for {'REAL' if real_mode else 'DEMO'} trading!")
print("🔄 Asyncio configured for notebook compatibility")
print("="*60)