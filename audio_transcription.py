import os
import pandas as pd
import openai
import logging
import psycopg2
import requests
from openai import OpenAI
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from time import sleep
from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor, as_completed

# =================== CONFIGURATION ===================

TABLE_NAME = "glific_messages_funnel_test"
SCHEMA_NAME = "abhishek"
TEXT_COLUMN = "body_final"
PHONETIC_COLUMN = "body_final_phonetic"
ID_COLUMN = "id"
MEDIA_URL_COLUMN = "media_url"
MESSAGE_TYPE_COLUMN = "message_type"
BATCH_SIZE = 50
MAX_WORKERS = 5  # Number of parallel threads
SLEEP_BETWEEN_BATCHES = 1  # seconds
LOG_FILE = "audio_transcription_errors.log"

# =================== ENV + LOGGING ===================

# Load environment variables
load_dotenv()

# Fetch the API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.WARNING
)

client = OpenAI(api_key=openai_api_key)

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# =================== TRANSCRIPTION ===================

def transcribe_audio_from_url(row_id, url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to download audio: HTTP {response.status_code}")

        with NamedTemporaryFile(delete=True, suffix=".mp3") as tmp_file:
            tmp_file.write(response.content)
            tmp_file.flush()

            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=open(tmp_file.name, "rb"),
                response_format="text",
                language="hi"  # Whisper can handle Hindi/Marathi/Urdu
            )
            return row_id, transcript
    except Exception as e:
        logging.error(f"Transcription error for ID {row_id}: {e}")
        return row_id, None

def transliterate_to_english(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a language expert. Convert the following Hindi, Urdu, or Marathi text to its phonetic transliteration in English using Latin script. Do not translate."},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Transliteration error: {e}")
        return None

def parallel_transcribe(df):
    results = []
    failures = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(transcribe_audio_from_url, row[ID_COLUMN], row[MEDIA_URL_COLUMN]): row[ID_COLUMN]
            for _, row in df.iterrows()
        }

        for future in as_completed(futures):
            row_id = futures[future]
            try:
                result_id, transcript = future.result()
                if transcript:
                    phonetic = transliterate_to_english(transcript)
                    results.append((result_id, transcript, phonetic))
                else:
                    failures.append(row_id)
            except Exception as e:
                logging.error(f"Unexpected error for ID {row_id}: {e}")
                failures.append(row_id)

    df_results = pd.DataFrame(results, columns=[ID_COLUMN, TEXT_COLUMN, PHONETIC_COLUMN])
    return df_results, failures

# =================== DB OPERATIONS ===================

def fetch_batch():
    print("🔄 Fetching a batch...")
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        query = f"""
            SELECT {ID_COLUMN}, {MEDIA_URL_COLUMN}
            FROM {SCHEMA_NAME}.{TABLE_NAME}
            WHERE {MESSAGE_TYPE_COLUMN} = 'audio'
              AND {TEXT_COLUMN} IS NULL
              AND {MEDIA_URL_COLUMN} IS NOT NULL
            ORDER BY {ID_COLUMN}
            LIMIT {BATCH_SIZE}
        """
        df = pd.read_sql_query(query, con=conn)
        conn.close()
        print(f"📥 Rows fetched: {len(df)}")
        return df
    except Exception as e:
        logging.error(f"Error fetching batch: {e}")
        print(f"❌ Error fetching batch: {e}")
        return pd.DataFrame()

def update_transcriptions(df):
    if df.empty:
        return
    with engine.begin() as conn:
        for _, row in df.iterrows():
            try:
                stmt = text(f"""
                    UPDATE {SCHEMA_NAME}.{TABLE_NAME}
                    SET {TEXT_COLUMN} = :transcription,
                        {PHONETIC_COLUMN} = :phonetic
                    WHERE {ID_COLUMN} = :id
                """)
                conn.execute(stmt, {
                    "transcription": row[TEXT_COLUMN],
                    "phonetic": row[PHONETIC_COLUMN],
                    "id": row[ID_COLUMN]
                })
                print(f"✅ Updated ID {row[ID_COLUMN]}")
            except Exception as e:
                logging.error(f"DB update failed for ID {row[ID_COLUMN]}: {e}")

# =================== PROCESSING LOOP ===================

def process_all_rows():
    print("🚀 Starting multilingual transcription process...")
    total_fetched = 0
    total_success = 0
    total_failures = []

    while True:
        df = fetch_batch()
        if df.empty:
            break

        total_fetched += len(df)
        df_transcribed, failures = parallel_transcribe(df)
        df_transcribed = df_transcribed[df_transcribed[TEXT_COLUMN].notnull()]
        update_transcriptions(df_transcribed)

        total_success += len(df_transcribed)
        total_failures.extend(failures)

        print(f"✅ Batch complete: {len(df_transcribed)} transcribed, {len(failures)} failed.\n")
        sleep(SLEEP_BETWEEN_BATCHES)

    # Final Summary
    print("\n================== 📊 SUMMARY ==================")
    print(f"Total rows fetched:       {total_fetched}")
    print(f"Successfully transcribed: {total_success}")
    print(f"Failed transcriptions:    {len(total_failures)}")
    if total_failures:
        print(f"❌ Failed IDs: {total_failures}")
    print("================================================")

# =================== ENTRY POINT ===================

if __name__ == "__main__":
    process_all_rows()
