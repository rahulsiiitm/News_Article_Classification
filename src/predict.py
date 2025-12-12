import pickle
import os
import sys

# Add 'src' to path so we can import clean_text
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess import clean_text

def load_artifacts():
    """Loads the saved model and vectorizer."""
    model_path = 'models/news_classifier.pkl'
    vect_path = 'models/tfidf_vectorizer.pkl'

    if not os.path.exists(model_path) or not os.path.exists(vect_path):
        print("❌ Error: Model files not found. Run 'src/train.py' first!")
        return None, None

    print("Loading model...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vect_path, 'rb') as f:
        vectorizer = pickle.load(f)
        
    return model, vectorizer

def predict_news(text, model, vectorizer):
    # 1. Clean the user's input (using the same logic as training)
    cleaned_text = clean_text(text)
    
    # 2. Convert to Numbers (TF-IDF)
    # Note: Use .transform(), NOT .fit_transform()
    features = vectorizer.transform([cleaned_text]).toarray()
    
    # 3. Predict
    prediction = model.predict(features)[0]
    
    # Get probabilities (confidence)
    probs = model.predict_proba(features)[0]
    max_prob = max(probs) * 100
    
    return prediction, max_prob

if __name__ == "__main__":
    model, vectorizer = load_artifacts()
    
    if model:
        print("\n📰 --- BBC News Classifier --- 📰")
        print("Type 'exit' to quit.\n")
        
        while True:
            user_input = input("Enter a news headline: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            category, confidence = predict_news(user_input, model, vectorizer)
            
            print(f"👉 Prediction: {category.upper()} ({confidence:.2f}%)")
            print("-" * 30)