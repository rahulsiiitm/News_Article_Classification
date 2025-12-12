import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Lemmatizer (converts "running" -> "run")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans news article text:
    1. Lowercase
    2. Remove numbers & punctuation
    3. Remove stopwords
    4. Lemmatize (optional but good for news)
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove numbers and punctuation (keep only letters)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Remove Stopwords & Lemmatize
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    
    return " ".join(cleaned_words)

def process_dataset():
    raw_path = 'data/raw/bbc-text.csv'
    processed_path = 'data/processed/bbc_clean.csv'
    
    if not os.path.exists(raw_path):
        print(f"❌ Error: {raw_path} not found.")
        print("Please download 'bbc-text.csv' from Kaggle and put it in 'data/raw/'")
        return

    print("Loading raw data...")
    df = pd.read_csv(raw_path)
    
    # Check if columns are correct (Kaggle BBC dataset usually has 'category' and 'text')
    print(f"Columns found: {df.columns.tolist()}")
    
    print("Cleaning text (this may take a moment)...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Save processed file
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"✅ Success! Cleaned data saved to {processed_path}")
    print(df.head())

if __name__ == "__main__":
    process_dataset()