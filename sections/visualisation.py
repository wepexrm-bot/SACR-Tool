import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer


def visualisation_section():
    st.subheader("Data Visualization")

    if not st.session_state.get("preprocessing_done", False):
        st.warning("⚠️ Please complete the Data Preprocessing section before accessing Visualization.")
        st.stop()

    if st.session_state.data_loaded and st.session_state.current_df is not None:
        df_vis = st.session_state.current_df.copy()
        st.success("✅ Using previously loaded data")
    else:
        data = st.file_uploader("Upload dataset:", type=['csv', 'xlsx', 'txt', 'json'])
        if data is None:
            st.info("👆 Please upload a dataset to continue")
            return
        try:
            if data.name.endswith('.csv'):
                df_vis = pd.read_csv(data)
            elif data.name.endswith('.xlsx'):
                df_vis = pd.read_excel(data)
            elif data.name.endswith('.json'):
                df_vis = pd.read_json(data)
            else:
                df_vis = pd.read_csv(data, sep='\t')
            st.session_state.current_df = df_vis.copy()
            st.session_state.data_loaded = True
            st.success("✅ Data successfully loaded")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return

    st.dataframe(df_vis.head(50), use_container_width=True)

    if st.checkbox("Select Multiple columns to plot"):
        selected_columns = st.multiselect(
            "Select your preferred columns", df_vis.columns
        )
        if selected_columns:
            df_subset = df_vis[selected_columns].copy()
            st.dataframe(df_subset, use_container_width=True)
        else:
            st.warning("Please pick at least one column to continue.")
            return
    else:
        selected_columns = list(df_vis.columns)
        df_subset = df_vis.copy()

    if st.checkbox("Class Imbalance Check"):
        class_column = st.selectbox("Select target/class column", df_vis.columns)
        class_counts = df_vis[class_column].value_counts(dropna=False)
        class_props = class_counts / class_counts.sum()
        st.write(class_props.to_frame("Proportion"))

        fig, ax = plt.subplots()
        sns.barplot(x=class_props.index.astype(str), y=class_props.values, ax=ax)
        ax.set_title(f"Class Distribution for {class_column}")
        ax.set_ylabel("Proportion")
        ax.set_xlabel("Class Labels")
        st.pyplot(fig)

    if st.checkbox("Basic Text Statistics"):
        text_column = st.selectbox(
            "Select a text column:", df_subset.select_dtypes(include="object").columns
        )
        word_counts = df_subset[text_column].astype(str).apply(lambda x: len(x.split()))
        st.write("Average Word Count:", word_counts.mean())
        st.write("Max Word Count:", word_counts.max())

    if st.checkbox("Average Word Length in Text Column"):
        text_col_len = st.selectbox(
            "Choose text column for word length analysis",
            df_subset.select_dtypes(include="object").columns,
        )
        avg_word_len_series = df_subset[text_col_len].astype(str).apply(
            lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
        )
        fig, ax = plt.subplots()
        sns.histplot(avg_word_len_series, kde=True, ax=ax)
        ax.set_title(f"Average Word Length in {text_col_len}")
        st.pyplot(fig)

    if st.checkbox("Compare Word Count in Positive vs Negative Reviews"):
        rating_column = st.selectbox("Select rating column:", df_vis.columns)
        text_column_wc = st.selectbox(
            "Select text column for comparison:",
            df_subset.select_dtypes(include="object").columns,
        )

        numeric_ratings = pd.to_numeric(df_vis[rating_column], errors="coerce")
        median_rating = numeric_ratings.median()
        word_counts_all = df_vis[text_column_wc].astype(str).apply(
            lambda x: len(x.split())
        )

        pos_mask = numeric_ratings >= median_rating
        neg_mask = numeric_ratings < median_rating
        pos_wc = word_counts_all[pos_mask]
        neg_wc = word_counts_all[neg_mask]

        fig, ax = plt.subplots()
        sns.kdeplot(pos_wc, label="Positive", shade=True)
        sns.kdeplot(neg_wc, label="Negative", shade=True)
        ax.set_title("Word Count Distribution: Positive vs Negative")
        ax.legend()
        st.pyplot(fig)

    if st.checkbox("N-Gram Analysis"):
        try:
            ngram_col = st.selectbox(
                "Select text column for N-Gram analysis:",
                df_subset.select_dtypes(include="object").columns,
            )
            n_val = st.slider("Choose N for N-grams:", 1, 5, 2)
            num_top = st.slider("Number of top N-grams to display:", 5, 50, 20)

            vectorizer = CountVectorizer(
                ngram_range=(n_val, n_val), stop_words="english"
            )
            ngram_matrix = vectorizer.fit_transform(
                df_subset[ngram_col].dropna().astype(str)
            )
            ngram_counts = ngram_matrix.sum(axis=0).A1
            ngram_vocab = vectorizer.get_feature_names_out()
            ngram_freq = dict(zip(ngram_vocab, ngram_counts))

            top_ngrams = Counter(ngram_freq).most_common(num_top)
            ngrams_df = pd.DataFrame(top_ngrams, columns=["N-gram", "Frequency"])
            st.dataframe(ngrams_df, use_container_width=True)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=ngrams_df, x="Frequency", y="N-gram", ax=ax)
            ax.set_title(f"Top {num_top} {n_val}-grams in {ngram_col}")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error in N-Gram analysis: {e}")
