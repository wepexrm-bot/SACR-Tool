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
from sklearn.preprocessing import LabelEncoder
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

    st.subheader("🏷️ Labeling")
    include_neutral = st.checkbox("Keep neutral class (ratings 5-6) as 3rd class", value=False,
                                  help="If checked, ratings 5-6 become a neutral class instead of being dropped.")

    label_method = st.selectbox("Labeling strategy:",
        ["Keyword-based (good/excellent/positive)", "From rating column (7-10 pos, 1-4 neg)",
         "From existing 0/1 column"])

    if 'rating' in label_method.lower() and rating_cols:
        target_col = rating_cols[0]
        if df[target_col].max() > 1:
            def map_rating(x):
                if x >= 7:
                    return 'positive'
                elif x <= 4:
                    return 'negative'
                else:
                    return 'neutral'
            df['label_raw'] = df[target_col].apply(map_rating)
            if not include_neutral:
                n_neutral = (df['label_raw'] == 'neutral').sum()
                df = df[df['label_raw'] != 'neutral'].copy()
                st.info(f"Dropped {n_neutral} neutral reviews (rating 5-6). Check 'Keep neutral class' to keep them.")
            else:
                st.info("Keeping neutral reviews (ratings 5-6) as a 3rd class.")
        else:
            df['label_raw'] = df[target_col].map({0: 'negative', 1: 'positive'})
    elif '0/1' in label_method.lower() and rating_cols:
        df['label_raw'] = df[rating_cols[0]].map({0: 'negative', 1: 'positive'})
    else:
        df['label_raw'] = np.where(
            df['processed_text'].str.contains(r'\b(good|excellent|positive|amazing|great|wonderful)\b', case=False, na=False),
            'positive', 'negative'
        )

    df = df.dropna(subset=['label_raw']).copy()

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label_raw'])
    class_names = list(le.classes_)
    st.session_state.class_names = class_names
    st.write(f"**Classes detected:** {class_names}")
    st.write(f"**Label distribution:** {df['label_raw'].value_counts().to_dict()}")
    st.dataframe(df[['processed_text', 'label_raw', 'label']].head(10), use_container_width=True)

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

    clean_df = df.dropna(subset=['processed_text', 'label']).copy()
    clean_df = clean_df[clean_df['processed_text'].str.strip() != '']

    try:
        kw = dict(ngram_range=(ngram_min, ngram_max), min_df=min_df, max_features=max_features)

        if vectorizer_type == "CountVectorizer":
            vect = CountVectorizer(**kw)
        else:
            vect = TfidfVectorizer(**kw)

        # Split BEFORE vectorization to prevent data leakage
        train_df, test_df = train_test_split(
            clean_df, test_size=test_size, random_state=42, stratify=clean_df['label']
        )

        x_train = vect.fit_transform(train_df['processed_text'])
        x_test = vect.transform(test_df['processed_text'])
        y_train = train_df['label'].values
        y_test = test_df['label'].values

        st.session_state.x_train = x_train
        st.session_state.x_test = x_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.vectorizer = vect
        st.session_state.vectorizer_type = vectorizer_type
        st.session_state.feature_engineering_done = True

        st.session_state.x_test_texts = test_df['processed_text'].values

        st.success("✅ Vectorization and labeling complete! (No data leakage — vectorizer fit on train only)")
        st.write(f"X_train shape: {x_train.shape}, X_test shape: {x_test.shape}")
        st.write(f"y_train distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        st.write(f"y_test distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

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
