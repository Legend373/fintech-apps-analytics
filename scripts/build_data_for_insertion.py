import pandas as pd
import os
import sys 

# -----------------------------
# 1. Load input datasets
# -----------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import DATA_PATHS
processed_path ='data/processed/reviews_processed.csv'
sentiment_path =DATA_PATHS["reviews_with_sentiment"]
output_path =DATA_PATHS['final_results']

df_processed = pd.read_csv(processed_path)
df_sentiment = pd.read_csv(sentiment_path)

# -----------------------------
# 2. Select only required columns from each dataset
# -----------------------------
df_proc = df_processed[["review_id", "review_date", "source"]]

df_sent = df_sentiment[
    [
        "review_id",
        "review_text",
        "rating",
        "bank_name",
        "bank_code",
        "vader_sentiment",   # label
        "vader_compound"     # score
    ]
]

# -----------------------------
# 3. Merge datasets on review_id
# -----------------------------
df_final = pd.merge(df_sent, df_proc, on="review_id", how="left")

# -----------------------------
# 4. Rename to match final DB table
# -----------------------------
df_final = df_final.rename(
    columns={
        "vader_sentiment": "sentiment_label",
        "vader_compound": "sentiment_score",
    }
)

# -----------------------------
# 5. Save final unified CSV
# -----------------------------
df_final.to_csv(output_path, index=False)

print("✅ final_results.csv created successfully!")
print(f"Saved to: {output_path}")
