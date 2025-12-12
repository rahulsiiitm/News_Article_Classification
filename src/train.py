import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_model():
    # 1. Load Clean Data
    data_path = 'data/processed/bbc_clean.csv'
    if not os.path.exists(data_path):
        print("❌ Error: Processed data not found. Run 'src/preprocess.py' first.")
        return

    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Drop rows where text might have become empty after cleaning
    df.dropna(subset=['cleaned_text'], inplace=True)

    # 2. Vectorization (TF-IDF)
    print("Vectorizing text (TF-IDF)...")
    # max_features=5000 keeps the top 5k most important words
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['cleaned_text']).toarray()
    y = df['category']

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train Model
    print("Training Naive Bayes Classifier...")
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # 5. Evaluation
    print("Evaluating...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 6. Save Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    os.makedirs('reports/figures', exist_ok=True)
    plt.savefig('reports/figures/confusion_matrix.png')
    print("Confusion Matrix saved to reports/figures/")

    # 7. Save Model & Vectorizer
    os.makedirs('models', exist_ok=True)
    with open('models/news_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
        
    print("\n🚀 Success! Model saved to 'models/news_classifier.pkl'")

if __name__ == "__main__":
    train_model()