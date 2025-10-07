# Troubleshooting Guide

## Quick Start

To verify your local setup is working, run:

```bash
python3 check_local_setup.py
```

This will check:
- ✅ .env file exists
- ✅ All Python dependencies are installed
- ✅ Environment variables are properly set
- ✅ Database connection works
- ✅ Table access is successful

## Common Issues and Solutions

### 1. Script runs in GitHub Actions but not locally

**Symptoms:**
- Works fine when triggered via GitHub Actions
- Fails or shows 0 rows fetched locally

**Solution:**
Your local environment needs proper configuration. The key difference is:
- **GitHub Actions**: Uses secrets stored in repository settings
- **Local**: Uses `.env` file

**Fix:**
1. Ensure `.env` file exists (copy from `.env.sample`)
2. Fill in actual credentials in `.env` file
3. Run `python3 check_local_setup.py` to verify

### 2. "No rows fetched" (0 rows)

**Symptoms:**
```
🔄 Fetching a batch...
📥 Rows fetched: 0
```

**Causes:**
1. **All audio messages already transcribed** (most common)
   - Query looks for `body_final IS NULL`
   - If all audio messages have transcriptions, 0 rows returned
   
2. **No audio messages in database**
   - Check `message_type` column values
   
3. **Audio messages missing media URLs**
   - Query requires `media_url IS NOT NULL`

**Solutions:**

**Option A**: Check what's in the database
```bash
python3 check_local_setup.py
```

**Option B**: Modify query to reprocess all audio
Edit `audio_transcription.py` line 125 and remove the `AND body_final IS NULL` condition:
```python
query = f"""
    SELECT {ID_COLUMN}, {MEDIA_URL_COLUMN}
    FROM {SCHEMA_NAME}.{TABLE_NAME}
    WHERE {MESSAGE_TYPE_COLUMN} = 'audio'
      AND {MEDIA_URL_COLUMN} IS NOT NULL
    ORDER BY {ID_COLUMN}
    LIMIT {BATCH_SIZE}
"""
```

**Option C**: Process only empty transcriptions
Edit query to check for empty strings too:
```python
WHERE {MESSAGE_TYPE_COLUMN} = 'audio'
  AND {MEDIA_URL_COLUMN} IS NOT NULL
  AND ({TEXT_COLUMN} IS NULL OR {TEXT_COLUMN} = '')
```

### 3. pandas/numpy compatibility error

**Symptoms:**
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```

**Solution:**
```bash
pip uninstall -y pandas numpy
pip install numpy==1.26.4
pip install -r requirements.txt
```

### 4. "python-dotenv could not parse statement"

**Symptoms:**
```
python-dotenv could not parse statement starting at line 1
```

**Cause:**
`.env` file has empty or malformed values

**Solution:**
Edit `.env` file and ensure all required variables have actual values (no empty values after `=`)

### 5. Database connection errors

**Symptoms:**
```
❌ Database connection failed
```

**Possible Causes:**
- Wrong credentials in `.env` file
- Database is not accessible from your network
- Firewall blocking connection
- VPN required but not connected

**Solution:**
1. Verify credentials are correct
2. Test connection: `python3 check_local_setup.py`
3. Check if VPN is required for database access
4. Verify database host is reachable: `ping your-db-host`

### 6. ModuleNotFoundError

**Symptoms:**
```
ModuleNotFoundError: No module named 'pandas'
```

**Solution:**
Install dependencies:
```bash
pip3 install -r requirements.txt
```

For better isolation, use a virtual environment:
```bash
python3 -m venv env
source env/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 7. OpenAI API errors

**Symptoms:**
```
openai.error.AuthenticationError: Incorrect API key provided
```

**Solution:**
1. Verify your API key in `.env` file
2. Ensure it starts with `sk-proj-` or `sk-`
3. Check key hasn't expired on OpenAI dashboard
4. Verify you have credits available

## Environment Comparison

| Aspect | GitHub Actions | Local |
|--------|---------------|-------|
| Python Version | 3.10 | Your system Python |
| Credentials | GitHub Secrets | `.env` file |
| Numpy Version | Pre-pinned 1.26.4 | Auto-installed |
| Virtual Env | Not used | Recommended |

## Best Practices for Local Development

1. **Always use virtual environment**
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

2. **Match GitHub Actions Python version**
   ```bash
   pyenv install 3.10
   pyenv local 3.10
   ```

3. **Pin numpy version like GitHub Actions**
   ```bash
   pip install numpy==1.26.4
   ```

4. **Keep .env file secure**
   - Never commit `.env` to git
   - Already in `.gitignore`
   - Use strong passwords

5. **Run setup checker before script**
   ```bash
   python3 check_local_setup.py && python3 audio_transcription.py
   ```

## Getting Help

If you continue to have issues:

1. Run diagnostic script:
   ```bash
   python3 check_local_setup.py
   ```

2. Check the error logs:
   ```bash
   cat audio_transcription_errors.log
   ```

3. Verify your environment:
   ```bash
   python3 --version
   pip list | grep -E "pandas|numpy|openai|sqlalchemy"
   ```

4. Test database connection separately before running full script

