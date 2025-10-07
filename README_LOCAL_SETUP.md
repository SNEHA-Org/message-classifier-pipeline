# Local Development Setup Guide

This guide explains how to run the audio transcription pipeline on your local machine.

## Prerequisites

- Python 3.10 or higher
- Access to the PostgreSQL database
- OpenAI API key

## Setup Steps

### 1. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv env

# Activate it
source env/bin/activate  # On macOS/Linux
# OR
env\Scripts\activate  # On Windows
```

### 2. Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Pre-install numpy to avoid version conflicts
pip install numpy==1.26.4

# Install other dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root with your actual credentials:

```bash
# Copy the sample file
cp .env.sample .env

# Edit .env and fill in your credentials
```

Your `.env` file should look like this (with actual values):

```
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_FINE_TUNED_MODEL=ft:gpt-3.5-turbo-xxxx
DB_USER=your_database_user
DB_PASSWORD=your_database_password
DB_HOST=your_database_host
DB_PORT=5432
DB_NAME=your_database_name
```

### 4. Run the Script

```bash
python3 audio_transcription.py
```

## Differences Between Local and GitHub Actions

| Aspect | Local | GitHub Actions |
|--------|-------|----------------|
| Python Version | Your system Python (3.12) | Python 3.10 |
| Environment Variables | `.env` file | GitHub Secrets |
| Numpy Version | Auto-installed | Pre-pinned to 1.26.4 |
| Database Access | Direct connection | Uses secrets |

## Common Issues

### Issue: "No rows fetched"
**Cause**: All audio messages already have transcriptions in `body_final` column.
**Solution**: This is expected behavior. The script only processes audio messages where `body_final` IS NULL.

### Issue: "Database credentials not found"
**Cause**: `.env` file is missing or has empty values.
**Solution**: Follow Step 3 above to configure your `.env` file properly.

### Issue: "pandas/numpy compatibility error"
**Cause**: Version mismatch between pandas and numpy.
**Solution**: 
```bash
pip uninstall -y pandas numpy
pip install numpy==1.26.4
pip install -r requirements.txt
```

### Issue: "python-dotenv could not parse statement"
**Cause**: Environment variables in `.env` file have empty values.
**Solution**: Fill in actual values in your `.env` file.

## Testing Database Connection

To verify your database connection works, you can run this quick test:

```python
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

with engine.connect() as conn:
    result = conn.execute(text("SELECT 1"))
    print("✅ Database connection successful!")
```

## Security Note

⚠️ **NEVER commit your `.env` file to git!** 

The `.env` file is already in `.gitignore` to prevent accidental commits of sensitive credentials.

