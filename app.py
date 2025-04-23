import streamlit as st
import pandas as pd
import numpy as np
import re
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from wordcloud import STOPWORDS
import joblib

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Define custom transformer
class PreprocessingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.nlp = nlp

    def clean_text(self, text):
        text = re.sub(r'\b\w{1,2}\b', '', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def lemmatize_text(self, text):
        doc = self.nlp(text)
        return ' '.join([token.lemma_ for token in doc])

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['combined_text'] = X['combined_text'].apply(self.clean_text).apply(self.lemmatize_text)
        return X

# Build pipeline
preprocessing_pipeline = Pipeline([
    ('preprocessing', PreprocessingTransformer())
])

feature_extraction_pipeline = ColumnTransformer([
    ('tfidf', TfidfVectorizer(
        token_pattern=r"\b[a-zA-Z]{3,}\b",
        stop_words=list(STOPWORDS),
        min_df=5,
        max_df=0.8), 'combined_text')
])

dimensionality_reduction_pipeline = Pipeline([
    ('svd', TruncatedSVD(n_components=300, algorithm='arpack'))
])

transformation_pipeline = Pipeline([
    ('preprocessing', preprocessing_pipeline),
    ('features', feature_extraction_pipeline),
    ('dim_reduction', dimensionality_reduction_pipeline)
])

# Load trained model (replace with your actual model path if using joblib)
model = LogisticRegression(
    C=4.573408970457499,
    penalty='l1',
    solver='saga',
    class_weight=None,
    random_state=42
)

# Streamlit UI
st.title("Fake News Detection App")
st.markdown("Enter a **title** and **news article text** to determine if it's Real or Fake.")

title_input = st.text_input("Enter News Title")
text_input = st.text_area("Enter News Content")

if st.button("Predict"):
    if not title_input or not text_input:
        st.warning("Please enter both title and text.")
    else:
        input_df = pd.DataFrame({
            'combined_text': [title_input + " " + text_input]
        })

        processed_input = transformation_pipeline.fit_transform(input_df)
        prediction = model.predict(processed_input)[0]
        label = 'Fake' if prediction == 1 else 'Real'
        st.success(f"Prediction: {label}")
