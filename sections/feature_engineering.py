import streamlit as st
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from utils import get_stopwords


def feature_engineering_section():
    st.subheader("Feature Engineering")

    if st.session_state.data_loaded and st.session_state.current_df is not None:
        df = st.session_state.current_df.copy()
        st.success("✅ Using previously loaded data")
    else:
        st.warning("⚠️ Please upload data in the Data Preprocessing section.")
        return

    st.dataframe(df.head(50), use_container_width=True)

    st.sidebar.header("Column Selection")
    text_col = st.sidebar.selectbox("Select Text Column", df.columns)

    rating_cols = [col for col in df.columns if 'rating' in col.lower() or 'score' in col.lower()]
    sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower() or 'label' in col.lower()]

    st.subheader("🧹 Preprocessing & Lemmatization (Optimized)")
    stop_words = get_stopwords()
    lemmatizer = WordNetLemmatizer()

    @st.cache_data(show_spinner=False)
    def fast_preprocess(series):
        pattern = re.compile(r'[^a-zA-Z\s]')
        return series.astype(str).str.lower()\
            .str.replace(pattern, '', regex=True)\
            .apply(lambda x: ' '.join([lemmatizer.lemmatize(w) for w in x.split() if w not in stop_words]))

    df['processed_text'] = fast_preprocess(df[text_col])
    st.dataframe(df[['processed_text']].head(10), use_container_width=True)

    st.subheader("🏷️ Binary Labeling")
    label_method = st.selectbox("Labeling strategy:",
        ["Keyword-based (good/excellent/positive)", "From rating column (7-10 pos, 1-4 neg)",
         "From existing 0/1 column"])
    if 'rating' in label_method.lower() and rating_cols:
        target_col = rating_cols[0]
        if df[target_col].max() > 1:
            df['Label_temp'] = df[target_col].apply(lambda x: 1 if x >= 7 else (0 if x <= 4 else 2))
            dropped = (df['Label_temp'] == 2).sum()
            df = df[df['Label_temp'] < 2].copy()
            df['label'] = df['Label_temp'].astype(int)
            df.drop(columns=['Label_temp'], inplace=True)
            st.info(f"Dropped {dropped} neutral reviews (rating 5-6)")
        else:
            df['label'] = df[target_col].astype(int)
    elif '0/1' in label_method.lower() and rating_cols:
        df['label'] = df[rating_cols[0]].astype(int)
    else:
        df['label'] = np.where(
            df['processed_text'].str.contains(r'\b(good|excellent|positive|amazing|great|wonderful)\b', case=False, na=False),
            1, 0
        )
    st.dataframe(df[['processed_text', 'label']].head(10), use_container_width=True)

    st.sidebar.header("Vectorizer Settings")
    vectorizer_type = st.sidebar.selectbox("Choose Vectorizer", ["CountVectorizer", "TF-IDF"])
    ngram_min = st.sidebar.slider("N-gram Range Start", 1, 5, 1)
    ngram_max = st.sidebar.slider("N-gram Range End", ngram_min, 7, 3)
    min_df = st.sidebar.slider("Min Document Frequency", 1, 20, 10)
    max_features = st.sidebar.slider("Max Features", 100, 10000, 10000)

    st.sidebar.header("Train/Test Split")
    split_option = st.sidebar.radio("Set split manually?", ("Use default (80/20)", "Custom split"))

    if split_option == "Custom split":
        train_size = st.sidebar.slider("Training Set Size (%)", 50, 95, 80, step=5)
        test_size = 1.0 - (train_size / 100)
    else:
        test_size = 0.2

    clean_df = df.dropna(subset=['processed_text', 'label'])

    try:
        kw = dict(ngram_range=(ngram_min, ngram_max), min_df=min_df, max_features=max_features)

        if vectorizer_type == "CountVectorizer":
            vect = CountVectorizer(**kw)
        else:
            vect = TfidfVectorizer(**kw)

        X = vect.fit_transform(clean_df['processed_text'])
        y = clean_df['label']
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        st.session_state.x_train = x_train
        st.session_state.x_test = x_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.vectorizer = vect
        st.session_state.vectorizer_type = vectorizer_type
        st.session_state.feature_engineering_done = True

        _, test_idx, _, _ = train_test_split(np.arange(len(clean_df)), y, test_size=test_size, random_state=42)
        st.session_state.x_test_texts = clean_df['processed_text'].iloc[test_idx].values

        st.success("✅ Vectorization and labeling complete!")
        st.write(f"X_train shape: {x_train.shape}")
        st.write(f"y_train distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    except ValueError as e:
        st.error(f"❌ Vectorization failed: {e}")
        st.stop()

    st.subheader("📊 Feature Scores")
    feature_names = vect.get_feature_names_out()
    if vectorizer_type == "CountVectorizer":
        feature_scores = np.asarray(x_train.sum(axis=0)).flatten()
        score_type = "Frequency (CountVectorizer)"
    else:
        feature_scores = np.asarray(x_train.mean(axis=0)).flatten()
        score_type = "Mean TF-IDF Score"

    score_df = pd.DataFrame({"Feature": feature_names, "Score": feature_scores})
    top_scores = score_df.sort_values(by="Score", ascending=False).head(20)
    st.write(f"🔢 Top 20 Features by {score_type}")
    st.dataframe(top_scores.reset_index(drop=True), use_container_width=True)

    if st.checkbox("🔎 Show Top Features by Chi-Squared Score"):
        st.subheader("📈 Chi-Squared Feature Scores")
        chi_scores, _ = chi2(x_train, y_train)
        chi_df = pd.DataFrame({"Feature": feature_names, "Chi2 Score": chi_scores})
        top_chi2 = chi_df.sort_values(by="Chi2 Score", ascending=False).head(20)

        st.dataframe(top_chi2.reset_index(drop=True), use_container_width=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Chi2 Score", y="Feature", data=top_chi2, ax=ax)
        ax.set_title("Top 20 Features by Chi-Squared Score")
        st.pyplot(fig)
