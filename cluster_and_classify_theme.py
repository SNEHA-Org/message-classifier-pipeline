import os
import textwrap
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import hdbscan
from sklearn.decomposition import PCA
from sqlalchemy import create_engine, text
from umap import UMAP
from dotenv import load_dotenv
import openai
import psycopg2
import re
from nltk.util import ngrams

# ========== CONFIGURATION ==========

TABLE_NAME = "glific_messages_funnel"
SCHEMA_NAME = "prod"
TEXT_COLUMN = "body"
CLUSTER_COLUMN = "theme"
ID_COLUMN = "id"
CLASS_COLUMN = "question_type"
BATCH_SIZE = 5000
LOG_FILE = "cluster_and_classify_theme_errors.log"
EMBED_MODEL = "text-embedding-3-large"
LABEL_MODEL = "gpt-4o"

# ========== ENV + LOGGING ==========

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# ========== TEXT PREPROCESSING ==========

CUSTOM_STOPWORDS = set([
    "hai", "ka", "ke", "ki", "ha", "ho", "se", "ko", "tha", "the", "hota", "hoti", 
    "kar", "raha", "rha", "nahi", "haan", "ji", "ab", "jab", "tak", "bahut", "bohot",
    "kyunki", "aur", "lekin", "par", "bhi", "toh", "to"
])

ROMANIZED_MAP = {
    "pait": "stomach",
    "dard": "pain",
    "baccha": "baby",
    "bacche": "baby",
    "bacha": "baby",
    "bache": "baby",
    "tablet": "tablet",
    "dawai": "medicine",
    "pet": "stomach",
    "fever": "fever",
    "vomit": "vomit",
    "sardi": "cold",
    "khansi": "cough",
    "tika": "vaccine",
    "injection": "vaccine",
    "delivery": "delivery",
    "pain": "pain",
    "maa": "mother",
    "garbh": "pregnancy",
    "tez": "severe",
    "khana": "food",
    "yojana": "scheme",
    "nasika": "vaccine",
}

def normalize_romanized(text):
    tokens = text.lower().split()
    normalized = [ROMANIZED_MAP.get(token, token) for token in tokens]
    return " ".join(normalized)

def remove_stopwords(text):
    tokens = text.lower().split()
    return " ".join([t for t in tokens if t not in CUSTOM_STOPWORDS])

def extract_ngrams(text, n_range=(1, 2)):
    tokens = text.lower().split()
    all_ngrams = []
    for n in range(n_range[0], n_range[1]+1):
        ngs = ngrams(tokens, n)
        all_ngrams.extend(["_".join(g) for g in ngs])
    return " ".join(all_ngrams)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = normalize_romanized(text)
    text = remove_stopwords(text)
    return extract_ngrams(text, n_range=(1, 2))

# ========== EMBEDDING + CLUSTERING ==========

def get_embedding(raw_text):
    try:
        cleaned = clean_text(raw_text)
        response = openai.embeddings.create(input=cleaned, model=EMBED_MODEL)
        return response.data[0].embedding
    except Exception as e:
        logging.warning(f"Embedding failed: {raw_text[:30]} | {e}")
        return None

def get_cluster_theme(messages):
    prompt = textwrap.dedent(f"""
        Classify the following user query typically in Hindi into possibly one of these themes: Delivery,Newborn Baby Health,Diet and Nutrition,Maternal Pain or Sickness,Breastfeeding,Government Schemes,General
If any other specific maternal and child health-related theme is evident as a cluster, return it instead. Make sure all themes returned are unique
Only return the theme and limit as much as possible to categorizing as General.
        {messages}
    """).strip()

    try:
        response = openai.chat.completions.create(
            model=LABEL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.warning(f"Theme labeling failed: {e}")
        return "Unknown"

# ========== DB OPERATIONS ==========

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
            SELECT {ID_COLUMN}, {TEXT_COLUMN}
            FROM {SCHEMA_NAME}.{TABLE_NAME}
            WHERE {CLASS_COLUMN} = 'query'
              AND {TEXT_COLUMN} IS NOT NULL
              AND TRIM({TEXT_COLUMN}) <> ''
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

def update_cluster_themes(df):
    if df.empty:
        return
    with engine.begin() as conn:
        for _, row in df.iterrows():
            try:
                stmt = text(f"""
                    UPDATE {SCHEMA_NAME}.{TABLE_NAME}
                    SET {CLUSTER_COLUMN} = :theme
                    WHERE {ID_COLUMN} = :id
                """)
                conn.execute(stmt, {
                    "theme": row[CLUSTER_COLUMN],
                    "id": row[ID_COLUMN]
                })
                print(f"✅ Updated ID {row[ID_COLUMN]} → {row[CLUSTER_COLUMN]}")
            except Exception as e:
                logging.error(f"DB update failed for ID {row[ID_COLUMN]}: {e}")

# ========== MAIN CLUSTERING PROCESS ==========

def cluster_and_label():
    df = fetch_batch()
    if df.empty:
        print("🚫 No data to process.")
        return

    print("🧠 Generating embeddings...")
    df["embedding"] = df[TEXT_COLUMN].apply(get_embedding)
    df = df[df["embedding"].notnull()]
    X = np.array(df["embedding"].tolist())

    print("📉 Reducing dimensionality with UMAP...")
    reducer = UMAP(n_neighbors=15, n_components=10, metric='cosine', random_state=42)
    X_umap = reducer.fit_transform(X)

    print("🔀 Clustering UMAP-reduced embeddings with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric='euclidean')
    df["cluster_id"] = clusterer.fit_predict(X_umap)

    n_clusters = len(set(df["cluster_id"])) - (1 if -1 in df["cluster_id"].values else 0)
    print(f"✅ Identified {n_clusters} clusters (plus noise if any).")

    print("🏷️ Generating cluster themes with GPT-4...")
    themes = {}
    for cid in sorted(df["cluster_id"].unique()):
        if cid == -1:
            themes[cid] = "General"
            continue

        sample = df[df["cluster_id"] == cid][TEXT_COLUMN].sample(min(10, len(df[df["cluster_id"] == cid])), random_state=42)
        formatted = "\n".join(f"- {m}" for m in sample)
        theme = get_cluster_theme(formatted)
        themes[cid] = theme
        print(f"🧠 Cluster {cid}: {theme}")

    df[CLUSTER_COLUMN] = df["cluster_id"].map(themes)

    print("\n📊 Final Output:")
    for cid, group in df.groupby("cluster_id"):
        print(f"\n--- Cluster {cid}: {themes[cid]} ---")
        for msg in group[TEXT_COLUMN].sample(min(10, len(group)), random_state=1):
            print(f"• {msg}")

    print("\n💾 Saving cluster themes back to DB...")
    update_cluster_themes(df)
    print("✅ Done.")

    # Optional: 2D plot using PCA (for visualization only)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_umap)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df["cluster_id"], cmap='tab10', s=10)
    plt.title("Query Message Clusters (UMAP + HDBSCAN)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label='Cluster ID')
    plt.grid(True)
    plt.show()

# ========== ENTRY POINT ==========

if __name__ == "__main__":
    cluster_and_label()
