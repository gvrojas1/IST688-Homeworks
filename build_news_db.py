"""
It reads news.csv, cleans it, scores each article for "legal/business
significance", embeds every article with OpenAI, and saves everything into
a persistent ChromaDB folder (./ChromaDB_for_HW7). HW7.py then just LOADS
that folder, so the app starts fast and never re-embeds.
"""
import os, re, sys, tomllib
import pandas as pd

# Same sqlite3 fix we used in Lab 4 / HW4 / HW5 
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules['pysqlite3']
except ImportError:
    pass

import chromadb
from openai import OpenAI

CSV_PATH = "./news.csv"
DB_PATH = "./ChromaDB_for_HW7"
COLLECTION = "HW7_News"

# 1. "Interestingness" rules (what a law firm would care about)
# (weight, regex). Higher weight = more legally significant.

CATEGORIES = {
    "Litigation":              (5, r"lawsuit|\bsued\b|\bsues\b|suing|litigation|class action|\bcourt\b|\bjudge\b|\bjury\b|verdict|plaintiff|legal action|legal options"),
    "Regulatory / Government": (5, r"regulator|\bcfpb\b|\bsec\b|\bftc\b|\bdoj\b|antitrust|\bprobe\b|investigat|\bfined\b|\bfines?\b|penalt|sanction|compliance|congress|senate|european commission|\bban\b|tariff|export control|\birs\b"),
    "M&A / Deals":             (4, r"acqui|merger|takeover|buyout|\bdeal\b|\bstake\b|divest|spin.?off|\bipo\b|private equity"),
    "Risk / Crisis":           (4, r"breach|hack|cyberattack|outage|recall|scandal|fraud|bankrupt|downgrade|plunge|sell.?off|layoff|job cuts|violation"),
    "Leadership":              (3, r"\bceo\b|resign|steps down|appoint|chief executive|succession"),
    "Financial Results":       (2, r"earnings|revenue|profit|quarterly|guidance|forecast|dividend"),
}

def score_article(document):
    """Keyword hit in the HEADLINE = full weight; hit only in the description/body = half weight."""
    title = document.split(" Description:")[0].lower()
    body = document.lower()
    score, hits = 0.0, []
    for cat, (weight, pattern) in CATEGORIES.items():
        if re.search(pattern, title):
            score += weight
            hits.append(cat)
        elif re.search(pattern, body):
            score += weight / 2
            hits.append(cat)
    return score, hits

# 2. Load + clean the CSV
def load_clean(path=CSV_PATH):
    df = pd.read_csv(path)
    df["company_name"] = df["company_name"].str.strip()      # some names have extra spaces, e.g. " TPG "
    df["doc_len"] = df["Document"].str.len()
    df = (df.sort_values("doc_len", ascending=False)
            .drop_duplicates("URL")                            # same article listed twice -> keep the fuller copy
            .drop(columns="doc_len")
            .reset_index(drop=True))
    return df

# 3. Get the API key (env variable, or the same secrets.toml Streamlit uses)

def get_api_key():
    if os.getenv("OPENAI_API_KEY"):
        return os.getenv("OPENAI_API_KEY")
    with open(".streamlit/secrets.toml", "rb") as f:
        return tomllib.load(f)["OPENAI_API_KEY"]


# 4. Build the vector DB

def build():
    client = OpenAI(api_key=get_api_key())
    df = load_clean()
    print(f"{len(df)} unique articles after removing duplicate URLs")

    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    try:
        chroma_client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = chroma_client.create_collection(COLLECTION)

    batch_size = 100
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        docs = batch["Document"].tolist()

        emb = client.embeddings.create(input=docs, model="text-embedding-3-small")
        embeddings = [e.embedding for e in emb.data]

        metadatas = []
        for _, row in batch.iterrows():
            score, hits = score_article(row["Document"])
            metadatas.append({
                "company": row["company_name"],
                "date": str(row["Date"])[:10],
                "days_since_2000": int(row["days_since_2000"]),
                "url": row["URL"],
                "legal_score": float(score),
                "categories": ", ".join(hits) if hits else "None",
            })

        collection.add(
            ids=[f"article_{i}" for i in batch.index],
            documents=docs,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"  added {start + len(batch)} / {len(df)}")

    print(f"Done. Collection '{COLLECTION}' has {collection.count()} articles in {DB_PATH}")

if __name__ == "__main__":
    build()