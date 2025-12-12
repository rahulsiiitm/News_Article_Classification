import streamlit as st
import pickle
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add 'src' to path to import clean_text
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from preprocess import clean_text

# --- Page Config ---
st.set_page_config(page_title="BBC News Classifier", page_icon="📰", layout="centered")

# --- Load Model & Vectorizer ---
@st.cache_resource
def load_artifacts():
    model_path = 'models/news_classifier.pkl'
    vect_path = 'models/tfidf_vectorizer.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(vect_path):
        st.error("❌ Model not found. Please run 'src/train.py' first.")
        return None, None

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vect_path, 'rb') as f:
        vectorizer = pickle.load(f)
        
    return model, vectorizer

model, vectorizer = load_artifacts()

# --- UI Layout ---
st.title("📰 AI News Classifier")
st.markdown("Classifies news articles into **Sports, Tech, Business, Politics, or Entertainment**.")

st.divider()

# Input
user_input = st.text_area("Paste a news headline or article here:", height=150, placeholder="Ex: Apple releases new iPhone with advanced AI features...")

if st.button("🔍 Classify News", type="primary"):
    if user_input and model:
        # 1. Preprocess
        cleaned_text = clean_text(user_input)
        
        # 2. Vectorize & Predict
        features = vectorizer.transform([cleaned_text]).toarray()
        prediction = model.predict(features)[0]
        probs = model.predict_proba(features)[0]
        
        # 3. Get Confidence Score
        classes = model.classes_
        max_prob = max(probs)
        
        # --- Display Results ---
        st.subheader(f"Category: {prediction.upper()}")
        
        # Dynamic Color based on Category
        color_map = {
            'sport': 'green',
            'tech': 'blue',
            'business': 'orange',
            'politics': 'red',
            'entertainment': 'purple'
        }
        color = color_map.get(prediction, 'gray')
        
        st.markdown(f":{color}[**Confidence: {max_prob*100:.2f}%**]")
        st.progress(max_prob)
        
        # --- Visualization (Bar Chart) ---
        st.write("---")
        st.write("📊 **Confidence Breakdown:**")
        
        # Create a DataFrame for the chart
        prob_df = pd.DataFrame({
            'Category': classes,
            'Probability': probs
        }).sort_values(by='Probability', ascending=False)
        
        # Plot using Streamlit's native bar chart
        st.bar_chart(prob_df.set_index('Category'))
        
    elif not user_input:
        st.warning("⚠️ Please enter some text to classify.")