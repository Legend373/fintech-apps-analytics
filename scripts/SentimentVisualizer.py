import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from scripts.config import DATA_PATHS


class SentimentVisualizer:
    def __init__(self, data_paths: dict = DATA_PATHS):
        self.data_paths = data_paths

        # Load processed sentiment CSV
        self.df = pd.read_csv(self.data_paths["sentiment_results"])

        # Try loading keyword JSON
        self.keyword_path = os.path.join(
            os.path.dirname(self.data_paths["sentiment_results"]), 
            "tfidf_keywords.json"
        )
        self.keywords = self._load_keywords()

        # Theme CSV file
        self.theme_path = os.path.join(
            os.path.dirname(self.data_paths["sentiment_results"]),
            "theme_assignments.csv"
        )

    # ---------------------------------------------------------
    # Utility: load keywords JSON
    # ---------------------------------------------------------
    def _load_keywords(self):
        try:
            with open(self.keyword_path, "r") as f:
                return json.load(f)
        except:
            return None

    # ---------------------------------------------------------
    # 1. Sentiment Score Distribution
    # ---------------------------------------------------------
    def plot_sentiment_distribution(self):
        plt.figure(figsize=(10, 5))
        sns.histplot(self.df["transf_score"], bins=30, kde=True)
        plt.title("Transformer Sentiment Score Distribution")
        plt.xlabel("Sentiment Score")
        plt.ylabel("Frequency")
        plt.show()

    # ---------------------------------------------------------
    # 2. Mean Sentiment per Bank
    # ---------------------------------------------------------
    def plot_bank_sentiment(self):
        plt.figure(figsize=(10, 5))
        sns.barplot(
            data=self.df.groupby("bank_name")["transf_score"].mean().reset_index(),
            x="bank_name",
            y="transf_score"
        )
        plt.title("Average Sentiment Score per Bank")
        plt.xlabel("Bank")
        plt.ylabel("Mean Sentiment Score")
        plt.xticks(rotation=30)
        plt.show()

    # ---------------------------------------------------------
    # 3. Rating × Sentiment Heatmap
    # ---------------------------------------------------------
    def plot_rating_sentiment_heatmap(self):
        pivot = self.df.pivot_table(
            index="rating",
            columns="bank_name",
            values="transf_score",
            aggfunc="mean"
        )
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, cmap="Blues")
        plt.title("Mean Transformer Sentiment by Rating & Bank")
        plt.show()

    # ---------------------------------------------------------
    # 4. TF-IDF Keyword Plots
    # ---------------------------------------------------------
    def plot_tfidf_keywords(self):
        if not self.keywords:
            print("⚠ No TF-IDF keyword file found. Skipping...")
            return

        for bank, kw_list in self.keywords.items():
            words = [k[0] for k in kw_list]
            scores = [k[1] for k in kw_list]

            plt.figure(figsize=(10, 5))
            sns.barplot(x=scores, y=words)
            plt.title(f"Top TF-IDF Keywords for {bank}")
            plt.xlabel("TF-IDF Score")
            plt.ylabel("Keyword")
            plt.show()

    # ---------------------------------------------------------
    # 5. Themes Frequency Bar Chart
    # ---------------------------------------------------------
    def plot_theme_frequency(self):
        try:
            theme_df = pd.read_csv(self.theme_path)

            theme_df["theme_count"] = theme_df["themes_matched"].apply(
                lambda x: len(eval(x)) if isinstance(x, str) and x.strip() else 0
            )

            plt.figure(figsize=(10, 5))
            sns.barplot(
                data=theme_df,
                x="bank_name",
                y="theme_count",
                palette="viridis"
            )
            plt.title("Themes Detected per Bank")
            plt.ylabel("Number of Themes Matched")
            plt.show()

        except Exception as e:
            print("⚠ Theme visual skipped:", e)

    # ---------------------------------------------------------
    # Run all plots
    # ---------------------------------------------------------
    def run_all(self):
        print("📊 Running all visualizations...")
        self.plot_sentiment_distribution()
        self.plot_bank_sentiment()
        self.plot_rating_sentiment_heatmap()
        self.plot_tfidf_keywords()
        self.plot_theme_frequency()
        print("✅ All visualizations complete.")


# ---------------------------------------------------------
# Usage
# ---------------------------------------------------------
if __name__ == "__main__":
    vis = SentimentVisualizer(DATA_PATHS)
    vis.run_all()
