import os
import re
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np

# NLP & sentiment
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import spacy

# transformers
from transformers import pipeline

# sklearn & gensim
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import AgglomerativeClustering
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

# ensure downloads (call once)
try:
    nltk.data.find("corpora/stopwords")
except Exception:
    nltk.download("stopwords")

# load small spaCy model (download separately if missing)
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    # user must run: python -m spacy download en_core_web_sm
    nlp = None

class SentimentThemeAnalyzer:
    """
    Pipeline to compute sentiment (VADER/TextBlob + DistilBERT) and perform
    thematic keyword extraction and simple theme grouping per bank.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_paths: Optional[Dict[str, str]] = None,
        transformer_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
        device: int = -1,  # use -1 for CPU, or GPU index if available
        neutral_threshold: float = 0.70,
    ):
        """
        Args:
            df: input DataFrame with at least columns ['review_id','review_text','rating','bank_name','bank_code']
            data_paths: optional dict for saving outputs (matching your DATA_PATHS)
            transformer_model: HuggingFace model for sentiment analysis
            device: device index for pipeline (use -1 for CPU)
            neutral_threshold: min confidence required to mark pos/neg. If below -> neutral.
        """
        self.df = df.copy()
        self.data_paths = data_paths or {}
        self.transformer_model = transformer_model
        self.neutral_threshold = neutral_threshold

        # Initialize lexicon & transformer analyzers
        self.vader = SentimentIntensityAnalyzer()
        self.sia = self.vader  # alias
        self.tb = TextBlob

        # Transformer pipeline (lazy init)
        self._transformer_pipe = None
        self.device = device

        # Precompute stopwords
        self.stop_words = set(stopwords.words("english"))

        # safe columns defaults
        for col in ["review_id", "review_text", "rating", "bank_name", "bank_code"]:
            if col not in self.df.columns:
                raise ValueError(f"Input df must contain column: '{col}'")

    # ---------------------------
    # Model / pipeline utilities
    # ---------------------------
    def _init_transformer(self):
        if self._transformer_pipe is None:
            self._transformer_pipe = pipeline(
                "sentiment-analysis",
                model=self.transformer_model,
                device=self.device
            )
        return self._transformer_pipe

    # ---------------------------
    # Preprocessing
    # ---------------------------
    @staticmethod
    def _basic_clean(text: str) -> str:
        if pd.isna(text):
            return ""
        txt = str(text)
        txt = txt.replace("\n", " ").replace("\r", " ")
        txt = re.sub(r"\s+", " ", txt)
        txt = txt.strip()
        return txt

    def preprocess(self, lemmatize: bool = True, keep_stopwords: bool = False):
        """Add columns: clean_text, tokens, tokens_nostop, lemmas (optional)"""
        print("Preprocessing text...")
        self.df["clean_text"] = self.df["review_text"].apply(self._basic_clean).str.lower()

        # Tokenize simple (spaCy if available)
        if nlp is not None:
            def spacy_tokenize(s: str):
                doc = nlp(s)
                tokens = [t.text for t in doc if not t.is_space]
                return tokens

            def spacy_lemmas(s: str):
                doc = nlp(s)
                lem = [t.lemma_.lower() for t in doc if (t.is_alpha and (keep_stopwords or t.text.lower() not in self.stop_words))]
                return lem

            self.df["tokens"] = self.df["clean_text"].apply(spacy_tokenize)
            self.df["lemmas"] = self.df["clean_text"].apply(spacy_lemmas) if lemmatize else self.df["tokens"]
        else:
            # fallback: split on whitespace, remove punctuation
            self.df["tokens"] = self.df["clean_text"].str.findall(r"\b\w+\b")
            if lemmatize:
                # no lemma fallback -> use tokens
                self.df["lemmas"] = self.df["tokens"]
            else:
                self.df["lemmas"] = self.df["tokens"]

        # remove stopwords for a column used for TF-IDF / topics
        self.df["tokens_nostop"] = self.df["lemmas"].apply(lambda toks: [t for t in toks if t not in self.stop_words])

    # ---------------------------
    # Lexicon sentiment
    # ---------------------------
    def compute_lexicon_sentiment(self):
        print("Computing lexicon sentiment (TextBlob + VADER)...")
        # TextBlob polarity & subjectivity
        self.df["tb_polarity"] = self.df["clean_text"].apply(lambda t: TextBlob(t).sentiment.polarity)
        self.df["tb_subjectivity"] = self.df["clean_text"].apply(lambda t: TextBlob(t).sentiment.subjectivity)

        # VADER compound & label
        self.df["vader_compound"] = self.df["clean_text"].apply(lambda t: self.sia.polarity_scores(t)["compound"])

        def vader_label(c):
            if c >= 0.05:
                return "positive"
            elif c <= -0.05:
                return "negative"
            else:
                return "neutral"

        self.df["vader_sentiment"] = self.df["vader_compound"].apply(vader_label)

    # ---------------------------
    # Transformer sentiment
    # ---------------------------
    def compute_transformer_sentiment(self, batch_size: int = 32):
        """
        Compute transformer-based sentiment and map label+score to positive/negative/neutral.
        SST-2 model returns POSITIVE/NEGATIVE. We map to neutral when model confidence < neutral_threshold.
        """
        print("Computing transformer sentiment using:", self.transformer_model)
        pipe = self._init_transformer()

        texts = self.df["clean_text"].fillna("").tolist()
        results = []

        # process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            out = pipe(batch, truncation=True)
            results.extend(out)

        # results is list of dicts with 'label' and 'score'
        labels = []
        scores = []
        mapped = []
        for r in results:
            label = r["label"].lower()  # 'positive' or 'negative'
            score = float(r["score"])
            # Map to neutral if confidence low
            if score < self.neutral_threshold:
                mapped_label = "neutral"
            else:
                mapped_label = label
            labels.append(label)
            scores.append(score)
            mapped.append(mapped_label)

        self.df["transf_label_raw"] = labels
        self.df["transf_score"] = scores
        self.df["transf_sentiment"] = mapped

    # ---------------------------
    # Aggregation
    # ---------------------------
    def aggregate_by_bank_and_rating(self, sentiment_col: str = "transf_score") -> pd.DataFrame:
        """
        Aggregate mean sentiment score by bank & rating (rating is star rating).
        Returns a DataFrame with grouped mean and counts.
        """
        assert sentiment_col in self.df.columns, f"{sentiment_col} not in dataframe"
        agg = (
            self.df
            .groupby(["bank_name", "rating"])
            .agg(
                mean_sentiment=(sentiment_col, "mean"),
                sd_sentiment=(sentiment_col, "std"),
                count_reviews=("review_id", "count")
            )
            .reset_index()
            .sort_values(["bank_name", "rating"])
        )
        return agg

    # ---------------------------
    # Keywords / TF-IDF
    # ---------------------------
    def extract_keywords_tfidf(self, n_top: int = 15) -> Dict[str, List[Tuple[str, float]]]:
        """
        Extract top TF-IDF keywords per bank.
        Returns dict: {bank_name: [(keyword, score), ...]}
        """
        print("Extracting keywords via TF-IDF per bank...")
        results = {}

        # For each bank, join tokens_nostop into a string and compute TF-IDF on corpus of banks
        banks = self.df["bank_name"].unique()
        bank_texts = []
        bank_keys = []
        for b in banks:
            subset = self.df[self.df["bank_name"] == b]
            joined = subset["tokens_nostop"].apply(lambda toks: " ".join(toks)).str.cat(sep=" ")
            bank_texts.append(joined)
            bank_keys.append(b)

        # compute TF-IDF across banks (so scores are comparable)
        tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
        X = tfidf.fit_transform(bank_texts)
        feature_names = tfidf.get_feature_names_out()

        for idx, b in enumerate(bank_keys):
            row = X[idx].toarray().flatten()
            top_idxs = np.argsort(row)[::-1][:n_top]
            keywords_scores = [(feature_names[i], float(row[i])) for i in top_idxs if row[i] > 0]
            results[b] = keywords_scores

        return results

    # ---------------------------
    # Topic modeling (LDA) - optional
    # ---------------------------
    def lda_topics_per_bank(self, num_topics: int = 3, num_words: int = 8) -> Dict[str, List[List[Tuple[str, float]]]]:
        """
        Simple LDA per bank on tokens_nostop -> returns top words per topic
        """
        print("Running LDA per bank...")
        bank_topics = {}
        for b in self.df["bank_name"].unique():
            subset = self.df[self.df["bank_name"] == b]
            token_lists = subset["tokens_nostop"].tolist()
            if len(token_lists) < 5:
                bank_topics[b] = []
                continue
            dict_ = Dictionary(token_lists)
            corpus = [dict_.doc2bow(toks) for toks in token_lists]
            lda = LdaModel(corpus=corpus, id2word=dict_, num_topics=min(num_topics, max(1, len(dict_))), passes=10, random_state=42)
            topics = []
            for t in range(min(num_topics, lda.num_topics)):
                topics.append(lda.show_topic(t, topn=num_words))
            bank_topics[b] = topics
        return bank_topics

    # ---------------------------
    # Simple theme grouping (rule-based + optional clustering)
    # ---------------------------
    def build_theme_dictionary(self) -> Dict[str, List[str]]:
        """
        Provide default theme -> keyword mapping (you can expand).
        We return a dict where keys are theme names and values are lists of regex tokens.
        """
        return {
            "Account Access Issues": ["login", "signin", "sign in", "password", "otp", "two-factor", "two factor", "locked"],
            "Transaction Performance": ["slow", "timeout", "failed", "transfer", "transaction", "pending", "delay", "processing"],
            "User Interface & Experience": ["ui", "ux", "design", "layout", "button", "crash", "bug", "lag", "screen"],
            "Customer Support": ["support", "customer service", "call", "email", "help", "agent", "response"],
            "Feature Requests": ["feature", "add", "request", "option", "would like", "please add"],
            "Positive Feedback": ["good", "nice", "great", "excellent", "amazing", "wow", "love", "well", "super", "fast", "easy"],
            "Negative Feedback": ["bad", "poor", "slow", "crash", "problem", "fix"],
        }

    def assign_themes_rule_based(self, top_k: int = 20, top_n_themes: int = 2) -> pd.DataFrame:
      """
      For each bank, take top TF-IDF keywords and assign top N themes based on matches.
      Returns a DataFrame of themes per bank with associated keywords.
      """
      tfidf_keywords = self.extract_keywords_tfidf(n_top=top_k)
      theme_dict = self.build_theme_dictionary()

      theme_assignments = []

      for bank, kws in tfidf_keywords.items():
        kws_only = [k for k, s in kws]
        theme_matches = {theme: [] for theme in theme_dict.keys()}

        for kw in kws_only:
            for theme, patterns in theme_dict.items():
                for pat in patterns:
                    if re.search(r"\b" + re.escape(pat) + r"\b", kw, flags=re.I):
                        theme_matches[theme].append(kw)

        # Count matches per theme
        theme_counts = {t: len(v) for t, v in theme_matches.items() if len(v) > 0}

        # Pick top N themes by matched keywords count
        sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        top_themes = [t for t, c in sorted_themes[:top_n_themes]]

        theme_assignments.append({
            "bank_name": bank,
            "keywords": kws_only,
            "themes_matched": top_themes
        })

      return pd.DataFrame(theme_assignments)
    # ---------------------------
    # Save results
    # ---------------------------
    def save_results(self, out_csv: Optional[str] = None):
        """
        Save final df with chosen columns to CSV. out_csv overrides data_paths
        """
        out_path = out_csv or self.data_paths.get("sentiment_results") or self.data_paths.get("processed_reviews")
        if out_path is None:
            raise ValueError("No output path specified in args or DATA_PATHS")

        # ensure folder exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # choose columns to save (if available)
        save_cols = [c for c in [
            "review_id", "review_text", "clean_text", "rating",
            "bank_name", "bank_code",
            "transf_sentiment", "transf_score",
            "vader_sentiment", "vader_compound",
            "tb_polarity", "tb_subjectivity",
            "tokens_nostop"
        ] if c in self.df.columns]
        self.df.to_csv(out_path, index=False, columns=save_cols)
        print(f"Saved results to {out_path}")

    # ---------------------------
    # Full run pipeline
    # ---------------------------
    def run_full_pipeline(self, do_transformer: bool = True, do_lexicon: bool = True, do_topics: bool = True):
        """
        Run the full pipeline and return a summary dict
        """
        # preprocess
        self.preprocess(lemmatize=True, keep_stopwords=False)

        # lexicon
        if do_lexicon:
            self.compute_lexicon_sentiment()

        # transformer
        if do_transformer:
            try:
                self.compute_transformer_sentiment()
            except Exception as e:
                print("Transformer sentiment failed (will continue). Error:", str(e))
                # fallback mapping from vader polarity
                self.df["transf_sentiment"] = self.df["vader_sentiment"]
                self.df["transf_score"] = self.df["vader_compound"]

        # aggregate
        agg = self.aggregate_by_bank_and_rating(sentiment_col="transf_score" if "transf_score" in self.df.columns else "vader_compound")

        # keywords & themes
        keywords = self.extract_keywords_tfidf(n_top=15)
        theme_df = self.assign_themes_rule_based(top_k=20)

        # optional LDA topics
        lda_topics = self.lda_topics_per_bank(num_topics=3, num_words=6) if do_topics else {}

        # save to CSV if configured
        try:
            self.save_results()
        except Exception as e:
            print("Save results failed:", e)

        return {
            "aggregates": agg,
            "keywords": keywords,
            "theme_assignments": theme_df,
            "lda_topics": lda_topics
        }
