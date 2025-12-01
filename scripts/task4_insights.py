# scripts/task4_insights_from_existing.py

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from scripts.config import DATA_PATHS

# Download stopwords
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

# ---------- CONFIG ----------

# <-- Your actual file
VISUALS_DIR = Path("visuals")
OUTPUTS_DIR = Path("outputs")

VISUALS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# thresholds for insights
MIN_COUNT_DRIVER = 20
MIN_COUNT_PAIN = 10
DRIVER_SENT_THRESH = 0.20
DRIVER_RATING_THRESH = 4.0
PAIN_SENT_THRESH = -0.10
PAIN_RATING_THRESH = 3.0


# ---------- LOAD DATA (NO REVIEW_DATE) ----------

def load_data(path):
    df = pd.read_csv(path)

    # Required columns you actually have
    required = {
        'review_id', 'review_text', 'clean_text', 'rating',
        'bank_name', 'bank_code',
        'transf_sentiment', 'transf_score',
        'vader_sentiment', 'vader_compound',
        'tb_polarity', 'tb_subjectivity',
        'tokens_nostop'
    }

    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing expected columns from {path}: {missing}")

    # Fix types
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['sentiment_score'] = pd.to_numeric(df['vader_compound'], errors='coerce')
    df['sentiment_label'] = df['vader_sentiment'].astype(str)

    df['clean_text'] = df['clean_text'].fillna(df['review_text']).astype(str)
    df['tokens_nostop'] = df['tokens_nostop'].fillna("").astype(str)

    return df


# ---------- BANK LIST ----------

def bank_list(df):
    return sorted(df["bank_name"].dropna().unique())


# ---------- TEXT ANALYTICS ----------

def top_terms_tfidf(corpus, topn=20, ngram_range=(1,2), min_df=5):
    v = TfidfVectorizer(stop_words='english', ngram_range=ngram_range, min_df=min_df)
    X = v.fit_transform(corpus)

    if X.shape[1] == 0:
        return []

    terms = np.array(v.get_feature_names_out())
    scores = np.asarray(X.sum(axis=0)).ravel()
    idx = scores.argsort()[::-1][:topn]

    return list(zip(terms[idx], scores[idx]))


def top_terms_count(corpus, topn=20, ngram_range=(1,2), min_df=5):
    v = CountVectorizer(stop_words='english', ngram_range=ngram_range, min_df=min_df)
    X = v.fit_transform(corpus)

    if X.shape[1] == 0:
        return []

    terms = np.array(v.get_feature_names_out())
    counts = np.asarray(X.sum(axis=0)).ravel()
    idx = counts.argsort()[::-1][:topn]

    return list(zip(terms[idx], counts[idx]))


# ---------- WORDCLOUD ----------

def wordcloud_from_text(text, out_path, max_words=150):
    wc = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        stopwords=STOPWORDS,
        max_words=max_words
    ).generate(text)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------- TOPIC MODELING ----------

def lda_topics(corpus, n_topics=6, n_top_words=8):
    v = CountVectorizer(stop_words="english", min_df=5, max_df=0.9)
    X = v.fit_transform(corpus)

    if X.shape[1] < 10:
        return []

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="batch",
        random_state=42
    ).fit(X)

    terms = v.get_feature_names_out()
    topics = []

    for comp in lda.components_:
        idx = comp.argsort()[::-1][:n_top_words]
        topics.append([terms[i] for i in idx])

    return topics


# ---------- THEME STATS ----------

def theme_stats(df_subset, mask):
    s = df_subset[mask]
    return {
        "count": len(s),
        "avg_rating": s["rating"].mean() if len(s) else np.nan,
        "avg_sentiment": s["sentiment_score"].mean() if len(s) else np.nan,
        "examples": s["review_text"].head(5).tolist()
    }

def compare_banks(df, bank1, bank2):
    b1 = df[df["bank_name"] == bank1]
    b2 = df[df["bank_name"] == bank2]

    comparison = {
        "bank_1": bank1,
        "bank_2": bank2,
        "avg_rating": {
            bank1: b1["rating"].mean(),
            bank2: b2["rating"].mean()
        },
        "avg_sentiment": {
            bank1: b1["vader_compound"].mean(),
            bank2: b2["vader_compound"].mean()
        },
        "total_reviews": {
            bank1: len(b1),
            bank2: len(b2)
        }
    }
    return comparison


def recommend_improvements(drivers, pains):
    recs = []

    # Based on pain points
    for p in pains[:2]:
        term = p["term"]
        recs.append(f"Improve issues related to '{term}'—this is a frequent customer complaint.")

    # Based on drivers (enhance what works)
    for d in drivers[:2]:
        term = d["term"]
        recs.append(f"Expand and promote strong area '{term}' to improve customer satisfaction.")

    # fallback
    if not recs:
        recs = ["Improve app performance", "Enhance UI/UX and reduce crashes"]

    return recs




# ---------- DETECT DRIVERS & PAIN POINTS ----------

def detect_drivers_and_pains(df, bank_name, top_k_terms=30):
    s = df[df['bank_name'] == bank_name]

    result = {
        "bank": bank_name,
        "drivers": [],
        "pains": [],
        "top_terms": []
    }

    if s.empty:
        return result

    tfidf_terms = top_terms_tfidf(s['clean_text'], top_k_terms)
    count_terms = top_terms_count(s['clean_text'], top_k_terms)
    result["top_terms"] = tfidf_terms[:15]

    for term, _ in count_terms:
        mask = s['clean_text'].str.contains(rf"\b{term}\b", case=False)
        stats = theme_stats(s, mask)

        # DRIVER
        if (
            stats["count"] >= MIN_COUNT_DRIVER and
            stats["avg_sentiment"] >= DRIVER_SENT_THRESH and
            stats["avg_rating"] >= DRIVER_RATING_THRESH
        ):
            result["drivers"].append({"term": term, **stats})

        # PAIN
        if (
            stats["count"] >= MIN_COUNT_PAIN and
            stats["avg_sentiment"] <= PAIN_SENT_THRESH and
            stats["avg_rating"] <= PAIN_RATING_THRESH
        ):
            result["pains"].append({"term": term, **stats})

    # Fallback: if none detected, pick top positive & negative
    if not result["drivers"]:
        candidates = [(t, s['sentiment_score'].mean(), t) for t, _ in count_terms]
        top_pos = sorted(candidates, key=lambda x: x[1], reverse=True)[:2]
        for t, _, _ in top_pos:
            m = s['clean_text'].str.contains(rf"\b{t}\b", case=False)
            result["drivers"].append({"term": t, **theme_stats(s, m)})

    if not result["pains"]:
        candidates = [(t, s['sentiment_score'].mean(), t) for t, _ in count_terms]
        top_neg = sorted(candidates, key=lambda x: x[1])[:2]
        for t, _, _ in top_neg:
            m = s['clean_text'].str.contains(rf"\b{t}\b", case=False)
            result["pains"].append({"term": t, **theme_stats(s, m)})

    return result
