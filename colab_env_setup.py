# 🔐 Complete Environment Configuration for Google Colab
# Comprehensive setup for all environment variables from .env.example

import os
from pathlib import Path

def setup_complete_environment():
    """
    Setup all environment variables for the intelligent trading system
    Handles all variables from .env.example with fallback to demo mode
    """
    
    # Check if we're in Colab
    try:
        from google.colab import userdata
        print("🔐 Google Colab detected - using Colab Secrets...")
        
        try:
            # Mercado Bitcoin API Configuration
            os.environ['MB_API_KEY'] = userdata.get('MB_API_KEY')
            os.environ['MB_API_SECRET'] = userdata.get('MB_API_SECRET')
            os.environ['MB_READ_ONLY'] = userdata.get('MB_READ_ONLY', 'true')
            
            # Trading Configuration
            os.environ['ASSET_SYMBOL'] = userdata.get('ASSET_SYMBOL', 'SOL-BRL')
            os.environ['TOTAL_BUDGET'] = userdata.get('TOTAL_BUDGET', '100.0')
            os.environ['MAX_POSITIONS'] = userdata.get('MAX_POSITIONS', '5')
            os.environ['POSITION_SIZE'] = userdata.get('POSITION_SIZE', '20.0')
            os.environ['SESSION_DURATION_HOURS'] = userdata.get('SESSION_DURATION_HOURS', '6')
            
            # System Configuration
            os.environ['MODE'] = userdata.get('MODE', 'test')
            os.environ['DATA_PATH'] = userdata.get('DATA_PATH', 'data')
            os.environ['DATABASE_PATH'] = userdata.get('DATABASE_PATH', 'data/sessions.db')
            os.environ['LOG_LEVEL'] = userdata.get('LOG_LEVEL', 'INFO')
            
            # Telegram Configuration (optional)
            try:
                os.environ['TELEGRAM_BOT_TOKEN'] = userdata.get('TELEGRAM_BOT_TOKEN')
                os.environ['TELEGRAM_CHAT_ID'] = userdata.get('TELEGRAM_CHAT_ID')
                os.environ['TELEGRAM_ENABLED'] = userdata.get('TELEGRAM_ENABLED', 'false')
                telegram_configured = True
            except:
                telegram_configured = False
                os.environ['TELEGRAM_ENABLED'] = 'false'
            
            print("✅ Successfully loaded credentials from Colab secrets")
            print(f"   📊 Asset: {os.environ['ASSET_SYMBOL']}")
            print(f"   💰 Budget: R$ {os.environ['TOTAL_BUDGET']}")
            print(f"   🔒 Read-only mode: {os.environ['MB_READ_ONLY']}")
            print(f"   📱 Telegram: {'✅ Configured' if telegram_configured else '❌ Not configured'}")
            
            return True
            
        except Exception as e:
            print(f"⚠️  Some Colab secrets not configured: {e}")
            print("🔽 Falling back to demo mode...")
            
    except ImportError:
        print("📓 Not in Google Colab - checking for .env file...")
        
        # Not in Colab - try to load .env
        try:
            from dotenv import load_dotenv
            if Path('.env').exists():
                load_dotenv('.env')
                print("✅ Loaded .env file")
                return True
            else:
                print("⚠️  .env file not found")
        except ImportError:
            print("⚠️  python-dotenv not available")
        except Exception as e:
            print(f"⚠️  Error loading .env: {e}")
    
    # Fallback: Demo mode with all variables
    print("🧪 Setting up DEMO MODE with safe defaults...")
    
    # Mercado Bitcoin API Configuration (DEMO)
    os.environ['MB_API_KEY'] = 'demo_api_key_for_testing'
    os.environ['MB_API_SECRET'] = 'demo_api_secret_for_testing'
    os.environ['MB_READ_ONLY'] = 'true'  # Always read-only in demo
    
    # Trading Configuration
    os.environ['ASSET_SYMBOL'] = 'SOL-BRL'
    os.environ['TOTAL_BUDGET'] = '100.0'
    os.environ['MAX_POSITIONS'] = '5'
    os.environ['POSITION_SIZE'] = '20.0'
    os.environ['SESSION_DURATION_HOURS'] = '6'
    
    # System Configuration  
    os.environ['MODE'] = 'demo'  # Force demo mode
    os.environ['DATA_PATH'] = '/content/data'  # Colab-friendly path
    os.environ['DATABASE_PATH'] = '/content/data/sessions.db'
    os.environ['LOG_LEVEL'] = 'INFO'
    
    # Telegram Configuration (disabled in demo)
    os.environ['TELEGRAM_BOT_TOKEN'] = 'demo_bot_token'
    os.environ['TELEGRAM_CHAT_ID'] = 'demo_chat_id'
    os.environ['TELEGRAM_ENABLED'] = 'false'
    
    print("✅ DEMO environment configured")
    print("   🎭 All API calls will be mocked")
    print("   📊 Asset: SOL-BRL")
    print("   💰 Budget: R$ 100.00")
    print("   🔒 Read-only: enabled")
    print("   📱 Telegram: disabled")
    
    return False  # False = demo mode


def print_environment_summary():
    """Print a summary of all configured environment variables"""
    
    print("\n" + "=" * 60)
    print("📋 ENVIRONMENT CONFIGURATION SUMMARY")
    print("=" * 60)
    
    # Group variables by category
    categories = {
        "🏦 Mercado Bitcoin API": [
            ('MB_API_KEY', 'API Key'),
            ('MB_API_SECRET', 'API Secret'), 
            ('MB_READ_ONLY', 'Read-Only Mode')
        ],
        "💹 Trading Configuration": [
            ('ASSET_SYMBOL', 'Trading Asset'),
            ('TOTAL_BUDGET', 'Total Budget'),
            ('MAX_POSITIONS', 'Max Positions'),
            ('POSITION_SIZE', 'Position Size'),
            ('SESSION_DURATION_HOURS', 'Session Duration')
        ],
        "⚙️ System Configuration": [
            ('MODE', 'System Mode'),
            ('DATA_PATH', 'Data Path'),
            ('DATABASE_PATH', 'Database Path'),
            ('LOG_LEVEL', 'Log Level')
        ],
        "📱 Telegram Configuration": [
            ('TELEGRAM_BOT_TOKEN', 'Bot Token'),
            ('TELEGRAM_CHAT_ID', 'Chat ID'),
            ('TELEGRAM_ENABLED', 'Telegram Enabled')
        ]
    }
    
    for category, vars_list in categories.items():
        print(f"\n{category}:")
        for env_var, description in vars_list:
            value = os.environ.get(env_var, 'NOT SET')
            
            # Mask sensitive values
            if 'KEY' in env_var or 'SECRET' in env_var or 'TOKEN' in env_var:
                if value and value != 'NOT SET' and 'demo' not in value.lower():
                    display_value = value[:8] + '...' + value[-4:] if len(value) > 12 else '***masked***'
                else:
                    display_value = value
            else:
                display_value = value
                
            print(f"   {description}: {display_value}")
    
    print("=" * 60)


# Main execution function
def setup_environment_for_notebook():
    """
    Complete setup function to be called at the start of the notebook
    """
    print("🚀 INTELLIGENT TRADING SYSTEM - Environment Setup")
    print("=" * 60)
    
    # Setup environment
    is_real_mode = setup_complete_environment()
    
    # Print summary
    print_environment_summary()
    
    # Create necessary directories
    data_path = Path(os.environ.get('DATA_PATH', 'data'))
    data_path.mkdir(exist_ok=True, parents=True)
    
    logs_path = Path('logs')
    logs_path.mkdir(exist_ok=True)
    
    print(f"\n📁 Created directories:")
    print(f"   Data: {data_path}")
    print(f"   Logs: {logs_path}")
    
    # Final status
    mode = "REAL TRADING" if is_real_mode else "DEMO/SIMULATION"
    safety = "🔒 SAFE" if os.environ.get('MB_READ_ONLY') == 'true' else "⚠️  LIVE"
    
    print(f"\n🎯 System ready in {mode} mode ({safety})")
    print("=" * 60)
    
    return is_real_mode


if __name__ == "__main__":
    setup_environment_for_notebook()