# BBC News Classification System

## Project Overview
This project is a Natural Language Processing (NLP) application designed to classify news articles into five categories: Business, Entertainment, Politics, Sport, and Tech. It uses TF-IDF for feature extraction and a Multinomial Naive Bayes classifier trained on the BBC News dataset.

## Key Features
- Multi-class classification with five news categories.
- Text preprocessing including lowercasing, stopword removal, punctuation removal, and lemmatization.
- Streamlit-based interactive dashboard for real-time predictions.
- Command-line interface for quick testing.
- Modular source code for preprocessing, training, and inference.

## Folder Structure
```
News-Article-Classification/
│
├── data/
│   ├── raw/                 # 'bbc-text.csv' (Original Dataset)
│   └── processed/           # 'bbc_clean.csv' (Preprocessed Data)
│
├── src/
│   ├── preprocess.py        # Cleaning pipeline
│   ├── train.py             # Model training script
│   ├── predict.py           # CLI prediction tool
│   └── convert_to_csv.py    # Utility for dataset formatting
│
├── models/
│   ├── news_classifier.pkl  # Saved model
│   └── tfidf_vectorizer.pkl # Saved vectorizer
│
├── app.py                   # Streamlit dashboard
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```

## Setup and Installation

### 1. Clone the Repository
```
git clone <your-repo-link>
cd News-Article-Classification
```

### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Prepare the Dataset
Download the BBC News Classification dataset from Kaggle and place the `bbc-text.csv` file into:
```
data/raw/
```

## How to Run

### 1. Preprocess the Data
```
python src/preprocess.py
```
This generates the processed dataset at `data/processed/bbc_clean.csv`.

### 2. Train the Model
```
python src/train.py
```
This saves the trained model and vectorizer into the `models/` directory.

### 3. Launch the Streamlit Dashboard
```
streamlit run app.py
```

### 4. Test Using CLI
```
python src/predict.py
```

## Technical Approach

### Preprocessing
- Lowercasing and punctuation removal.
- Stopword removal using NLTK.
- Lemmatization for reducing words to base form.
- TF-IDF vectorization (up to 5000 features).

### Model
- Multinomial Naive Bayes classifier.
- Optimized for multi-class text classification.

## Results
- Achieved approximately 96% accuracy on the test data.
- Strong performance across clearly separated categories.
- Minor overlap observed between Business and Politics.

## Tech Stack
- Python 3.x
- Scikit-Learn
- NLTK
- Pandas and NumPy
- Streamlit
