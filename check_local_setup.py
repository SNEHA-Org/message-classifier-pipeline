#!/usr/bin/env python3
"""
Local Environment Setup Checker
This script verifies that your local environment is properly configured to run the audio transcription pipeline.
"""

import os
import sys
from dotenv import load_dotenv

def check_env_file():
    """Check if .env file exists"""
    if not os.path.exists('.env'):
        print("❌ .env file not found")
        print("   → Create one by copying .env.sample: cp .env.sample .env")
        return False
    print("✅ .env file exists")
    return True

def check_dependencies():
    """Check if required Python packages are installed"""
    required_packages = {
        'pandas': 'pandas',
        'openai': 'openai',
        'psycopg2': 'psycopg2-binary',
        'sqlalchemy': 'SQLAlchemy',
        'requests': 'requests',
        'dotenv': 'python-dotenv',
    }
    
    missing = []
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
            print(f"✅ {package_name} is installed")
        except ImportError:
            print(f"❌ {package_name} is NOT installed")
            missing.append(package_name)
    
    if missing:
        print(f"\n⚠️  Install missing packages with: pip install {' '.join(missing)}")
        return False
    return True

def check_env_variables():
    """Check if environment variables are properly set"""
    load_dotenv()
    
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API key',
        'DB_USER': 'Database username',
        'DB_PASSWORD': 'Database password',
        'DB_HOST': 'Database host',
        'DB_NAME': 'Database name',
    }
    
    missing = []
    empty = []
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value is None:
            print(f"❌ {var} is not set ({description})")
            missing.append(var)
        elif not value or value.strip() == '' or value == 'sk-':
            print(f"⚠️  {var} is empty or has placeholder value ({description})")
            empty.append(var)
        else:
            # Mask sensitive values in output
            if 'PASSWORD' in var or 'KEY' in var:
                masked = value[:10] + '...' if len(value) > 10 else '***'
                print(f"✅ {var} is set ({masked})")
            else:
                print(f"✅ {var} is set ({value})")
    
    if missing or empty:
        print(f"\n⚠️  Edit your .env file and fill in the required values")
        return False
    return True

def check_database_connection():
    """Check if database connection works"""
    try:
        from sqlalchemy import create_engine, text
        
        load_dotenv()
        DB_USER = os.getenv("DB_USER")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_HOST = os.getenv("DB_HOST")
        DB_PORT = os.getenv("DB_PORT", "5432")
        DB_NAME = os.getenv("DB_NAME")
        
        if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
            print("❌ Cannot test database connection - credentials not set")
            return False
        
        engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ Database connection successful")
            return True
            
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def check_table_access():
    """Check if we can access the target table"""
    try:
        from sqlalchemy import create_engine, text
        import pandas as pd
        
        load_dotenv()
        DB_USER = os.getenv("DB_USER")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_HOST = os.getenv("DB_HOST")
        DB_PORT = os.getenv("DB_PORT", "5432")
        DB_NAME = os.getenv("DB_NAME")
        
        if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
            print("❌ Cannot test table access - credentials not set")
            return False
        
        engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        
        SCHEMA_NAME = "abhishek"
        TABLE_NAME = "glific_messages_funnel_test"
        
        query = f"SELECT COUNT(*) as count FROM {SCHEMA_NAME}.{TABLE_NAME}"
        result = pd.read_sql_query(query, con=engine)
        count = result.iloc[0]['count']
        
        print(f"✅ Table access successful - {count} rows found")
        return True
        
    except Exception as e:
        print(f"❌ Table access failed: {e}")
        return False

def main():
    print("="*60)
    print("🔍 CHECKING LOCAL ENVIRONMENT SETUP")
    print("="*60)
    print()
    
    checks = [
        ("Environment File", check_env_file),
        ("Python Dependencies", check_dependencies),
        ("Environment Variables", check_env_variables),
        ("Database Connection", check_database_connection),
        ("Table Access", check_table_access),
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\n📋 Checking: {check_name}")
        print("-" * 60)
        result = check_func()
        results.append((check_name, result))
    
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    all_passed = True
    for check_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {check_name}")
        if not result:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✅ All checks passed! You can run the audio transcription script.")
        print("   Run: python3 audio_transcription.py")
        return 0
    else:
        print("\n❌ Some checks failed. Please fix the issues above before running the script.")
        print("   See README_LOCAL_SETUP.md for detailed setup instructions.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

