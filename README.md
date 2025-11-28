          
          Customer Experience Analytics for Fintech Apps

This project is mainly for Omega Consultancy a company that is supporting banks to improve their mobile apps to enhance customer retention and satisfaction. 


aiming for :
   - Scrape user reviews from the Google Play Store.
   - Analyze sentiment (positive/negative/neutral) and extract themes (e.g., "bugs", "UI").
   - Identify satisfaction drivers (e.g., speed) and pain points (e.g., crashes).
   - Store cleaned review data in a Postgres database.
   - Deliver a report with visualizations and actionable recommendations.

     
           
           Critical milestones taken

1. Google Play Store Review Scraper(scripts/scraper.py): Data Collection

This script scraped user reviews from Google Play Store for three Ethiopian banks.
Target: 500 reviews per bank (1200 total minimum)

2. Data Preprocessing Script(scripts/preprocessing.py)

This script cleans and preprocesses the scraped reviews data.
- Handles missing values
- Normalizes dates
- Cleans text data