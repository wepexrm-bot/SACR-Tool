import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def eda_section():
    st.subheader("Exploratory Data Analysis")

    if not st.session_state.get("preprocessing_done", False):
        st.warning("⚠️ Please complete the Data Preprocessing section before accessing EDA.")
        st.stop()

    if st.session_state.data_loaded and st.session_state.current_df is not None:
        df = st.session_state.current_df
        st.success("✅ Using previously loaded data")
    else:
        data = st.file_uploader("Upload dataset:", type=['csv', 'xlsx', 'txt', 'json'])
        if data is not None:
            try:
                if data.name.endswith('.csv'):
                    df = pd.read_csv(data)
                elif data.name.endswith('.xlsx'):
                    df = pd.read_excel(data)
                elif data.name.endswith('.json'):
                    df = pd.read_json(data)
                else:
                    df = pd.read_csv(data, sep='\t')
                st.success("✅ Data successfully loaded")
                st.session_state.current_df = df
                st.session_state.data_loaded = True
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                return
        else:
            st.info("👆 Please upload a dataset to continue")
            return

    st.write(f"**Shape:** {df.shape}")
    st.dataframe(df.head(10), use_container_width=True)

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        st.write("**Missing Values:**")
        st.dataframe(missing.to_frame('Missing Values'), use_container_width=True)
    else:
        st.write("No missing values found.")

    text_col = None
    for col in df.columns:
        if df[col].dtype == 'object':
            avg_len = df[col].astype(str).str.len().mean()
            if avg_len > 50:
                text_col = col
                break

    potential_sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower() or 'label' in col.lower()]
    rating_cols = [col for col in df.columns if 'rating' in col.lower() or 'score' in col.lower()]

    st.write(f"**Detected — Text column:** `{text_col}`  |  **Sentiment columns:** {potential_sentiment_cols}  |  **Rating columns:** {rating_cols}")

    if text_col is None:
        st.info("No text column with avg length > 50 auto-detected. Select one manually:")
        text_col = st.selectbox("Text column:", [c for c in df.columns if df[c].dtype == 'object'])

    before = len(df)
    df = df.dropna(subset=[text_col])
    if before - len(df) > 0:
        st.write(f"Dropped {before - len(df)} rows with null text.")

    df['text_length'] = df[text_col].astype(str).apply(len)
    df['word_count'] = df[text_col].astype(str).apply(lambda x: len(x.split()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df['text_length'], bins=50, edgecolor='black')
    axes[0].set_title('Text Length Distribution')
    axes[0].set_xlabel('Length (chars)')
    axes[0].set_ylabel('Frequency')

    axes[1].hist(df['word_count'], bins=50, edgecolor='black', color='orange')
    axes[1].set_title('Word Count Distribution')
    axes[1].set_xlabel('Word Count')
    axes[1].set_ylabel('Frequency')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    target_col = None
    if rating_cols:
        target_col = rating_cols[0]
    elif potential_sentiment_cols:
        target_col = potential_sentiment_cols[0]

    if target_col is not None and target_col in df.columns:
        st.subheader(f"Class Distribution — {target_col}")
        if df[target_col].dtype == 'object':
            vals = df[target_col].astype(str).str.lower()
            st.write(vals.value_counts())
            fig, ax = plt.subplots(figsize=(6, 4))
            vals.value_counts().plot(kind='bar', ax=ax)
            ax.set_title(f'Class Distribution — {target_col}')
            ax.set_xlabel(target_col)
            ax.set_ylabel('Count')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            num_vals = pd.to_numeric(df[target_col], errors='coerce')
            st.write(num_vals.value_counts().sort_index())
            fig, ax = plt.subplots(figsize=(6, 4))
            num_vals.value_counts().sort_index().plot(kind='bar', ax=ax)
            ax.set_title(f'Class Distribution — {target_col}')
            ax.set_xlabel(target_col)
            ax.set_ylabel('Count')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    if target_col is not None and target_col in df.columns:
        st.subheader("Word Cloud by Sentiment")
        try:
            # Determine classes present from session or auto-detect
            class_names = st.session_state.get("class_names", None)
            neutral_present = class_names is not None and 'neutral' in class_names

            class_dfs = {}
            if df[target_col].dtype == 'object':
                cat_vals = df[target_col].astype(str).str.lower()
                pos_cats = {'positive', 'pos', 'good'}
                neg_cats = {'negative', 'neg', 'bad'}
                neu_cats = {'neutral', 'neu'}
                class_dfs['positive'] = df[cat_vals.isin(pos_cats)]
                class_dfs['negative'] = df[cat_vals.isin(neg_cats)]
                if neutral_present:
                    class_dfs['neutral'] = df[cat_vals.isin(neu_cats)]
            else:
                num_vals = pd.to_numeric(df[target_col], errors='coerce')
                uniq = sorted(num_vals.dropna().unique())
                if neutral_present:
                    class_dfs['positive'] = df[num_vals >= 7]
                    class_dfs['negative'] = df[num_vals <= 4]
                    class_dfs['neutral'] = df[(num_vals >= 5) & (num_vals <= 6)]
                elif num_vals.nunique() == 2 and set(num_vals.dropna().unique()).issubset({0, 1}):
                    class_dfs['positive'] = df[num_vals == 1]
                    class_dfs['negative'] = df[num_vals == 0]
                else:
                    med = num_vals.median()
                    class_dfs['positive'] = df[num_vals >= med]
                    class_dfs['negative'] = df[num_vals < med]

            sentiment_order = ['positive', 'neutral', 'negative']
            present = [s for s in sentiment_order if s in class_dfs and not class_dfs[s].empty]
            if present:
                colors = {'positive': 'Greens', 'neutral': 'Blues', 'negative': 'Reds'}
                n = len(present)
                fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
                if n == 1:
                    axes = [axes]
                for i, cls in enumerate(present):
                    text = ' '.join(class_dfs[cls][text_col].astype(str).head(500))
                    wc = WordCloud(width=500, height=400, background_color='white',
                                   colormap=colors[cls], max_words=100).generate(text)
                    axes[i].imshow(wc, interpolation='bilinear')
                    axes[i].axis('off')
                    axes[i].set_title(f'{cls.title()} Reviews', fontsize=14, color=colors[cls].replace('s', ''))
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Could not split data into sentiment classes for word cloud.")
        except Exception as e:
            st.error(f"Word cloud generation failed: {e}")
