import streamlit as st
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


def initialize_session_state():
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'preprocessing_done' not in st.session_state:
        st.session_state.preprocessing_done = False
    if 'feature_engineering_done' not in st.session_state:
        st.session_state.feature_engineering_done = False
    if 'models_results' not in st.session_state:
        st.session_state.models_results = []
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None


def validate_data(df):
    issues = []
    if df.empty:
        issues.append("Dataset is empty")
    if df.isnull().all().any():
        issues.append("Some columns are entirely null")
    if len(df) < 10:
        issues.append("Dataset is very small (< 10 rows)")
    text_cols = df.select_dtypes(include=['object']).columns
    if len(text_cols) == 0:
        issues.append("No text columns found for analysis")
    return issues


def preprocess_with_progress(df, text_col, stop_words):
    progress_bar = st.progress(0)
    status_text = st.empty()
    processed_texts = []
    total_rows = len(df)
    for i, text in enumerate(df[text_col]):
        processed_text = clean_text(text, stop_words)
        processed_texts.append(processed_text)
        progress = (i + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(f'Processing: {i+1}/{total_rows} texts')
    progress_bar.empty()
    status_text.empty()
    return processed_texts


def compare_models(models_results):
    if not models_results:
        st.warning("No models trained yet for comparison")
        return
    comparison_df = pd.DataFrame(models_results)
    numeric_cols = comparison_df.select_dtypes(include=['number']).columns
    display_df = pd.concat([comparison_df[['Model']], comparison_df[numeric_cols]], axis=1)
    styled_df = display_df.style.highlight_max(axis=0, color='lightgreen')
    st.dataframe(styled_df, use_container_width=True)
    with st.expander("🔧 View Model Parameters"):
        for result in models_results:
            st.write(f"**{result['Model']}**: `{result.get('Parameters', {})}`")
    if 'Accuracy' in comparison_df.columns:
        best_idx = comparison_df['Accuracy'].idxmax()
        best_model = comparison_df.loc[best_idx, 'Model']
        best_accuracy = comparison_df.loc[best_idx, 'Accuracy']
        st.success(f"🎯 Recommended Model: {best_model} (Accuracy: {best_accuracy:.4f})")


def get_stopwords():
    custom_stopwords = set(stopwords.words('english'))
    custom_stopwords.discard('not')
    custom_stopwords.update({'would', 'shall', 'could', 'might'})
    neutral_words = {'organization', 'company', 'work', 'worked', 'employee', 'employer', 'working', 'firm'}
    return custom_stopwords.union(neutral_words)


def contraction_expansion(content):
    content = re.sub(r"won\'t", "would not", content)
    content = re.sub(r"can\'t", "can not", content)
    content = re.sub(r"don\'t", "do not", content)
    content = re.sub(r"shouldn\'t", "should not", content)
    content = re.sub(r"needn\'t", "need not", content)
    content = re.sub(r"hasn\'t", "has not", content)
    content = re.sub(r"haven\'t", "have not", content)
    content = re.sub(r"weren\'t", "were not", content)
    content = re.sub(r"mightn\'t", "might not", content)
    content = re.sub(r"didn\'t", "did not", content)
    content = re.sub(r"n\'t", " not", content)
    return content


def clean_text(text, stop_words):
    if not isinstance(text, str):
        return ''
    text = contraction_expansion(text)
    text = contractions.fix(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w) for w in filtered]
    return ' '.join(lemmatized)
