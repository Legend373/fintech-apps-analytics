import psycopg2
import pandas as pd

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import DATA_PATHS, DB_CONFIG

# Load your cleaned dataframe
df = pd.read_csv(DATA_PATHS["final_results"])

# Connect to PostgreSQL
conn = psycopg2.connect(
    host=DB_CONFIG['host'],
    database=DB_CONFIG['database'],
    user=DB_CONFIG['user'],
    password=DB_CONFIG['password']
)

cursor = conn.cursor()

# Insert banks (unique)
unique_banks = df[['bank_name', 'bank_code']].drop_duplicates()

for _, row in unique_banks.iterrows():
    cursor.execute("""
        INSERT INTO banks (bank_name, app_name)
        VALUES (%s, %s)
        ON CONFLICT (bank_name) DO NOTHING;
    """, (row['bank_name'], row['bank_code']))

conn.commit()

# Build bank_id lookup
cursor.execute("SELECT bank_id, bank_name FROM banks;")
bank_map = {name: bid for bid, name in cursor.fetchall()}

# Insert reviews
for _, row in df.iterrows():
    cursor.execute("""
        INSERT INTO reviews 
        (bank_id, review_text, rating, review_date, sentiment_label, sentiment_score, source)
        VALUES (%s, %s, %s, %s, %s, %s, %s);
    """, (
        bank_map.get(row['bank_name']),
        row['review_text'],
        row['rating'],
        row['review_date'],
        row['sentiment_label'],
        row['sentiment_score'],
        row['source']
    ))

conn.commit()
cursor.close()
conn.close()

print("✔ Data inserted successfully!")
