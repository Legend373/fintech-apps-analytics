"""
Configuration file for Bank Reviews Analysis Project
"""
import os

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Google Play Store App IDs
APP_IDS = {
# env
    'CBE': os.getenv('CBE_APP_ID', 'com.combanketh.mobilebanking'),
    'BOA': os.getenv('ABYSSINIA_APP_ID', 'com.boa.boaMobileBanking'),
    'Dashenbank': os.getenv('DASHEN_APP_ID', 'com.dashen.dashensuperapp')

}
#Database Credentials
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'bank_reviews'),
    'user': os.getenv('DB_USER','legend'),
    'password': os.getenv('DB_PASSWORD')
}

# Bank Names Mapping
BANK_NAMES = {
    'CBE': 'Commercial Bank of Ethiopia',
    'BOA': 'Bank of Abyssinia',
    'Dashenbank': 'Dashen Bank'
}

# Scraping Configuration
SCRAPING_CONFIG = {
    'reviews_per_bank': int(os.getenv('REVIEWS_PER_BANK', 500)),
    'max_retries': int(os.getenv('MAX_RETRIES', 3)),
    'lang': 'en',
    'country': 'et'  # Ethiopia
}

# File Paths
DATA_PATHS = {
    'raw': '../../data/raw',
    'processed': '../../data/processed',
    'raw_reviews': '../../data/raw/reviews_raw.csv',
    'processed_reviews': '../../data/processed/reviews_processed.csv',
    'sentiment_results': '../../data/processed/reviews_with_sentiment.csv',
    'reviews_with_sentiment':'data/processed/reviews_with_sentiment.csv',
    'final_results': 'data/processed/reviews_final.csv'
}









