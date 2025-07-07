import os
import pandas as pd
import openai
import logging
import psycopg2
from openai import OpenAI
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from time import sleep

# =================== CONFIGURATION ===================

TABLE_NAME = "glific_messages_funnel"
SCHEMA_NAME = "prod"
TEXT_COLUMN = "body_final_phonetic"
CLASS_COLUMN = "question_type"
MESSAGE_COLUMN = "message_type"
ID_COLUMN = "id"
BATCH_SIZE = 50
SLEEP_BETWEEN_BATCHES = 1  # seconds
LOG_FILE = "small_talk_classify_errors.log"
CLASSIFY_LOG_FILE = "classification_log.csv"

# =================== ENV + LOGGING ===================

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
FINE_TUNED_MODEL = os.getenv("OPENAI_FINE_TUNED_MODEL")

if not FINE_TUNED_MODEL:
    raise ValueError("Missing OPENAI_FINE_TUNED_MODEL in environment or .env file")

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

# =================== CLASSIFICATION ===================

def classify_text(text):
    if not text or not isinstance(text, str) or text.strip() == "":
        return None
    try:
        response = client.chat.completions.create(
            model=FINE_TUNED_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Classify the following message as 'query', 'small-talk', or anything else meaningful. Return just your classification as a single word or short phrase."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        print(f"[CLASSIFY] Text: {text[:30]} → {result}")
        return result
    except Exception as e:
        short_text = text[:30].replace("\n", " ") if text else "<empty>"
        logging.error(f"OpenAI error for text [{short_text}]: {e}")
        return None

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
            SELECT id, {TEXT_COLUMN}, message_type
            FROM {SCHEMA_NAME}.{TABLE_NAME}
            WHERE {CLASS_COLUMN} IS NULL
            AND {MESSAGE_COLUMN} IN ('text','audio')
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

def update_classifications(df):
    if df.empty:
        return
    with engine.begin() as conn:
        for _, row in df.iterrows():
            try:
                stmt = text(f"""
                    UPDATE {SCHEMA_NAME}.{TABLE_NAME}
                    SET {CLASS_COLUMN} = :classification
                    WHERE {ID_COLUMN} = :id
                """)
                conn.execute(stmt, {
                    "classification": row[CLASS_COLUMN],
                    "id": row[ID_COLUMN]
                })
                print(f"✅ Updated ID {row[ID_COLUMN]} → {row[CLASS_COLUMN]}")
            except Exception as e:
                logging.error(f"DB update failed for ID {row[ID_COLUMN]}: {e}")

# =================== PROCESSING LOOP ===================

def process_all_rows():
    print("🚀 Starting classification process...")
    classification_logs = []

    while True:
        df = fetch_batch()
        if df.empty:
            print("✅ All rows processed.")
            break

        def decide_classification(row):
            text_val = row.get(TEXT_COLUMN)
            is_audio = row.get("message_type") == "audio"
            auto_classified = False

            if is_audio and (pd.isna(text_val) or str(text_val).strip() == ""):
                classification = "small_talk"
                auto_classified = True
            else:
                classification = classify_text(text_val)
                auto_classified = False

            classification_logs.append({
                "id": row.get(ID_COLUMN),
                "message_type": row.get("message_type"),
                "text": str(text_val)[:30],
                "classification": classification,
                "method": "auto" if auto_classified else "openai"
            })

            return classification

        df[CLASS_COLUMN] = df.apply(decide_classification, axis=1)
        
        # ✅ Ensure even auto-classified (text-null) rows are updated
        df_to_update = df[df[CLASS_COLUMN].astype(str).str.strip() != ""]

        if df_to_update.empty:
            print("⚠️ No rows to update in this batch. Skipping to next.")
            sleep(SLEEP_BETWEEN_BATCHES)
            continue

        print("📊 Classifications returned:", df[CLASS_COLUMN].dropna().unique().tolist())
        print(f"💾 Rows to update: {len(df_to_update)}")
        update_classifications(df_to_update)

        print(f"✅ Batch complete.\n")
        sleep(SLEEP_BETWEEN_BATCHES)

    # 🔍 Save classification logs to CSV
    if classification_logs:
        pd.DataFrame(classification_logs).to_csv(CLASSIFY_LOG_FILE, index=False)
        print(f"📝 Classification log saved to {CLASSIFY_LOG_FILE}")

# =================== ENTRY POINT ===================

if __name__ == "__main__":
    process_all_rows()
