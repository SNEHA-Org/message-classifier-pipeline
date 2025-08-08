import os
import re
import time
import logging
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import openai
import psycopg2
from datetime import datetime
import concurrent.futures
from psycopg2.extras import RealDictCursor

# === CONFIGURATION ===
DRY_RUN = False  # Set to True for testing
BATCH_SIZE = 50
MAX_WORKERS = 5  # Threads for GPT calls
RETRY_ATTEMPTS = 3

# Timestamps for log naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# === LOGGING SETUP ===
LOG_PATH = f"reclassification_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)

# === LOAD ENV VARS ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

# === DB & TABLE CONFIG ===
TABLE_NAME = "glific_messages_funnel"
SCHEMA_NAME = "prod"
ID_COLUMN = "id"
THEME_COLUMN = "theme"
TEXT_COLUMN = "body_final_phonetic"
QUESTION_COLUMN = "question_type"  # <-- configurable identifier column

# === THEME DEFINITIONS ===
ALLOWED_THEMES = [
    "General", "Breastfeeding", "Delivery", "Fetal Movement",
    "Diet and Nutrition", "Newborn Baby Health", "Maternal Pain or Sickness",
    "Government Schemes"
]

THEME_RULES = [
    ("Delivery", ["delivery", "labour", "operation", "normal", "cesarean", "birth", "hospital"]),
    ("Diet and Nutrition", ["food", "vitamin", "calcium", "iron", "khana", "pani", "protein"]),
    ("Maternal Pain or Sickness", ["dard", "pain", "fever", "vomit", "sick", "dawai", "medicine", "goli", "tablet"]),
    ("Fetal Movement", ["movement", "kick", "hilna", "chal", "hil", "baby move", "fetal"]),
    ("Breastfeeding", ["breast", "milk", "feeding", "doodh"]),
    ("Newborn Baby Health", ["newborn", "infant", "crying", "jaundice", "ro raha", "baby", "bacha", "baccha"]),
    ("Government Schemes", ["scheme", "yojana", "ration", "card", "free", "modi", "sarkar"]),
]


def match_theme_rule(text):
    text = text.lower()
    for theme, keywords in THEME_RULES:
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", text):
                return theme
    return None


def gpt_theme_classifier(message):
    prompt = f"""
Classify the following health-related message into one of these themes:

{', '.join(ALLOWED_THEMES)}

Message:
"{message}"

Only return the theme from the list above. No explanation.
"""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            theme = response.choices[0].message.content.strip()
            if theme in ALLOWED_THEMES:
                return theme
        except openai.RateLimitError:
            wait = 2 ** attempt
            logging.warning(f"⚠️ Rate limited. Retrying in {wait}s...")
            time.sleep(wait)
        except Exception as e:
            logging.error(f"❌ GPT classification error: {e}")
            break
    return "General"


def reclassify_with_rules_and_gpt(batch_size=BATCH_SIZE):
    last_id = 0
    batch_num = 1

    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.autocommit = True
        logging.info("✅ Connected to PostgreSQL database.")
    except Exception as e:
        logging.error(f"❌ DB connection failed: {e}")
        return

    try:
        while True:
            logging.info(f"\n📦 Processing batch {batch_num} (last_id: {last_id})...")

            query = f"""
                SELECT {ID_COLUMN}, {TEXT_COLUMN}, {THEME_COLUMN}
                FROM {SCHEMA_NAME}.{TABLE_NAME}
                WHERE 
                {ID_COLUMN} > %s
                AND ({THEME_COLUMN} LIKE '%%,%%' OR {THEME_COLUMN} IS NULL OR TRIM({THEME_COLUMN}) = '')
                AND {TEXT_COLUMN} IS NOT NULL AND TRIM({TEXT_COLUMN}) <> ''
                AND {QUESTION_COLUMN} = 'query'
                ORDER BY {ID_COLUMN}
                LIMIT {batch_size}
            """
            df = pd.read_sql_query(query, con=conn, params=(last_id,))


            if df.empty:
                logging.info("✅ No more rows to classify.")
                break

            # Apply rules
            df["new_theme"] = df[TEXT_COLUMN].apply(match_theme_rule)
            rule_hits = df["new_theme"].notnull().sum()

            # GPT fallback
            null_indices = df[df["new_theme"].isnull()].index
            texts_to_classify = df.loc[null_indices, TEXT_COLUMN].tolist()
            logging.info(f"🤖 GPT fallback for {len(texts_to_classify)} rows...")

            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                results = list(tqdm(
                    executor.map(gpt_theme_classifier, texts_to_classify),
                    total=len(texts_to_classify),
                    desc=f"Batch {batch_num} GPT"
                ))

            for i, idx in enumerate(null_indices):
                df.at[idx, "new_theme"] = results[i]

            gpt_hits = df["new_theme"].notnull().sum() - rule_hits
            updates = df[(df["new_theme"] != df[THEME_COLUMN]) | df[THEME_COLUMN].isnull()].copy()
            updated_count = len(updates)

            logging.info(f"✅ Rule matched: {rule_hits}")
            logging.info(f"🧠 GPT matched: {gpt_hits}")
            logging.info(f"🔁 Records to update: {updated_count}")

            # Log CSV
            if updated_count > 0:
                log_rows = []
                for _, row in updates.iterrows():
                    log_rows.append({
                        "id": row[ID_COLUMN],
                        "old_theme": row[THEME_COLUMN] if pd.notnull(row[THEME_COLUMN]) else "<null>",
                        "new_theme": row["new_theme"],
                        "text": row[TEXT_COLUMN],
                        "batch": batch_num
                    })
                log_df = pd.DataFrame(log_rows)
                log_file = f"theme_reclassification_log_batch_{batch_num}_{timestamp}.csv"
                log_df["timestamp"] = datetime.now().isoformat()
                log_df.to_csv(log_file, index=False)
                logging.info(f"📄 Log written: {log_file}")

            # Update DB
            if DRY_RUN:
                logging.info("🔒 DRY RUN – skipping DB update.")
            else:
                with conn.cursor() as cur:
                    for _, row in tqdm(updates.iterrows(), total=updated_count, desc=f"Updating DB (batch {batch_num})"):
                        cur.execute(
                            f"""
                            UPDATE {SCHEMA_NAME}.{TABLE_NAME}
                            SET {THEME_COLUMN} = %s
                            WHERE {ID_COLUMN} = %s
                            """,
                            (row["new_theme"].strip(), row[ID_COLUMN])
                        )

            # Set last processed ID
            last_id = int(df[ID_COLUMN].max())
            batch_num += 1

    finally:
        conn.close()
        logging.info("🔌 PostgreSQL connection closed.")

# === ENTRY POINT ===
if __name__ == "__main__":
    reclassify_with_rules_and_gpt()
