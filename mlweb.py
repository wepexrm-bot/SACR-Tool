import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd
import nltk
from nltk.corpus import stopwords
import contractions
import re
import time
from fpdf import FPDF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2
from wordcloud import WordCloud
from collections import Counter

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
import joblib
import io

# Enhanced session state management
def initialize_session_state():
    """Initialize session state variables"""
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

# Data validation function
def validate_data(df):
    """Validate uploaded data and return issues"""
    issues = []
    if df.empty:
        issues.append("Dataset is empty")
    if df.isnull().all().any():
        issues.append("Some columns are entirely null")
    if len(df) < 10:
        issues.append("Dataset is very small (< 10 rows)")
    
    # Check for text columns
    text_cols = df.select_dtypes(include=['object']).columns
    if len(text_cols) == 0:
        issues.append("No text columns found for analysis")
    
    return issues

# Progress bar preprocessing function
def preprocess_with_progress(df, text_col, stop_words):
    """Preprocess text with progress bar"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    processed_texts = []
    total_rows = len(df)
    
    for i, text in enumerate(df[text_col]):
        processed_text = clean_text(text, stop_words)
        processed_texts.append(processed_text)
        
        # Update progress
        progress = (i + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(f'Processing: {i+1}/{total_rows} texts')
    
    progress_bar.empty()
    status_text.empty()
    
    return processed_texts

# Model comparison function
def compare_models(models_results):
    """Compare model performance and recommend best model"""
    if not models_results:
        st.warning("No models trained yet for comparison")
        return

    comparison_df = pd.DataFrame(models_results)


    # Separate out non-numeric columns that can't be used for highlighting
    numeric_cols = comparison_df.select_dtypes(include=['number']).columns
    display_df = pd.concat([comparison_df[['Model']], comparison_df[numeric_cols]], axis=1)

    # Style numeric columns
    styled_df = display_df.style.highlight_max(axis=0, color='lightgreen')
    st.dataframe(styled_df, use_container_width=True)

    # Show parameters separately (optional)
    with st.expander("🔧 View Model Parameters"):
        for result in models_results:
            st.write(f"**{result['Model']}**: `{result.get('Parameters', {})}`")

    # Best model recommendation
    if 'Accuracy' in comparison_df.columns:
        best_idx = comparison_df['Accuracy'].idxmax()
        best_model = comparison_df.loc[best_idx, 'Model']
        best_accuracy = comparison_df.loc[best_idx, 'Accuracy']
        st.success(f"🎯 Recommended Model: {best_model} (Accuracy: {best_accuracy:.4f})")




st.title("SACR Tool (Sentiment Analysis on Customer Review)")


def get_stopwords():
    custom_stopwords = set(stopwords.words('english'))
    # Keep 'not' (critical for sentiment negation)
    custom_stopwords.discard('not')
    # Add modal verbs that don't carry sentiment
    custom_stopwords.update({'would', 'shall', 'could', 'might'})
    neutral_words = {'organization', 'company', 'work', 'worked', 'employee', 'employer', 'working', 'firm'}
    return custom_stopwords.union(neutral_words)

# Define text preprocessing function
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
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w) for w in filtered]
    return ' '.join(lemmatized)

class LemmaTokenizer:
    def __init__(self):
        self.wordnetlemma = WordNetLemmatizer()
    def __call__(self, reviews):
        return [self.wordnetlemma.lemmatize(word) for word in word_tokenize(reviews)]


def web():

    initialize_session_state()  # Add this line first
    activities = ['Data Preprocessing','EDA', 'Visualisation', 'Feature Engineering','Models', 'Model Comparison',  'About Us']

    option = st.sidebar.selectbox("Selection Option:", activities)

    st.sidebar.title("📚 Help Center")

    with st.sidebar.expander("🔰 How to Use This App"):
        st.markdown("""
                    **Steps:**
        1. Upload your dataset in the **Data Preprocessing** section.
        2. Explore your data in **EDA**.
        3. Visualize relationships under **Visualization**.
        4. Use **Feature Engineering** to preprocess and vectorize.
        5. Choose and train a model in **Models**.
        6. Choose a small dataset for faster analysis
                    
        Click on the image below to see a visual walkthrough of how to use the application.
        """)

        # Use columns to make the image look clickable
        col1, col2, col3 = st.sidebar.columns([1, 4, 1])
        with col2:
            if st.button("🖼️ Show Tutorial"):
                st.session_state.show_tutorial = True

    if st.session_state.get("show_tutorial", False):
        st.markdown("## Getting Started Tutorial")
        st.image("Sentiment analysis diagram.png", caption="SACR Tool Walkthrough", use_container_width=True)
        if st.button("❌ Hide Tutorial"):
            st.session_state.show_tutorial = False




    if option == 'Data Preprocessing':
        st.subheader("Data Preprocessing")

        data = st.file_uploader("Upload dataset:", type=['csv', 'xlsx', 'txt', 'json'])
        if data is not None:
            try:
                # Handle different file types
                if data.name.endswith('.csv'):
                    df = pd.read_csv(data)
                elif data.name.endswith('.xlsx'):
                    df = pd.read_excel(data)
                elif data.name.endswith('.json'):
                    df = pd.read_json(data)
                else:
                    df = pd.read_csv(data, sep='\t')  # For txt files
                
                st.success("✅ Data successfully loaded")

                # Validate data
                issues = validate_data(df)
                if issues:
                    st.warning("⚠️ Data Quality Issues:")
                    for issue in issues:
                        st.write(f"- {issue}")
                
                # Save to session state
                st.session_state.current_df = df
                st.session_state.data_loaded = True

                # Dataset Overview
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                with col2:
                    st.info(f"💾 Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
                st.dataframe(df.head(50), use_container_width=True)

                # Check for missing values
                st.subheader("Missing Values Overview")
                missing_info = df.isnull().sum()
                missing_info = missing_info[missing_info > 0]
                if not missing_info.empty:
                    st.write("Columns with missing values:")
                    st.dataframe(missing_info.to_frame('Missing Values'), use_container_width=True)
                else:
                    st.info("No missing values detected.")

                # Optional: Convert existing categorical sentiment into binary
                potential_sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower() or 'label' in col.lower()]

                if potential_sentiment_cols:
                    st.subheader("🧠 Sentiment Conversion (Categorical to Binary)")
                    selected_sent_col = st.selectbox("Select sentiment column to convert to binary:", potential_sentiment_cols)

                    if df[selected_sent_col].dtype == 'object':
                        df['binary_sentiment_label'] = df[selected_sent_col].astype(str).str.lower().map(
                            lambda x: 1 if any(p in x for p in ['positive', 'pos', 'good', 'excellent']) 
                                    else 0 if any(n in x for n in ['negative', 'neg', 'bad', 'poor']) 
                                    else np.nan
                        )

                        # Preview mapping results
                        st.write("📋 Mapping Preview")
                        st.dataframe(df[[selected_sent_col, 'binary_sentiment_label']].dropna().head(10), use_container_width=True)

                        unmapped = df['binary_sentiment_label'].isna().sum()
                        if unmapped > 0:
                            st.warning(f"⚠️ {unmapped} rows could not be mapped and have NaN in the new label column.")
                        else:
                            st.success("✅ All sentiment values successfully mapped to binary!")

                # Text column selection
                text_columns = [col for col in df.columns if df[col].dtype == 'object']
                selected_cols = st.multiselect("Select columns to use for text analysis:", text_columns)

                if selected_cols:
                    st.subheader("🛠️ Preprocessing Options")
                    use_progress = st.checkbox("Show progress bar (recommended for large datasets)", value=True)
                    
                    if st.button("🚀 Start Preprocessing"):
                        with st.spinner("Processing text data..."):
                            df['analysis_text'] = df[selected_cols].fillna('').agg(' '.join, axis=1)
                            stop_words = get_stopwords()
                            
                            if use_progress:
                                processed_texts = preprocess_with_progress(df, 'analysis_text', stop_words)
                                df['analysis_text_clean'] = processed_texts
                            else:
                                df['analysis_text_clean'] = df['analysis_text'].apply(lambda x: clean_text(x, stop_words))
                            
                            st.session_state.current_df = df
                            st.session_state.preprocessing_done = True

                        st.success("✅ Text preprocessing completed!")

                        # Preview processed results
                        preview_df = df[['analysis_text', 'analysis_text_clean']].head(10)
                        st.subheader("📄 Preprocessing Results Preview")
                        st.dataframe(preview_df, use_container_width=True)

                        # Text statistics
                        avg_original_length = df['analysis_text'].str.len().mean()
                        avg_clean_length = df['analysis_text_clean'].str.len().mean()
                        reduction_pct = ((avg_original_length - avg_clean_length) / avg_original_length * 100)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Avg Original Length", f"{avg_original_length:.1f} chars")
                        with col2:
                            st.metric("Avg Clean Length", f"{avg_clean_length:.1f} chars")
                        with col3:
                            st.metric("Reduction", f"{reduction_pct:.1f}%")

                else:
                    st.warning("⚠️ Please select at least one text column for analysis.")

            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")


    elif option == 'EDA':
        st.subheader("Exploratory Data Analysis")

        # ⛔ Block access unless preprocessing is done
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
                    # Handle different file types
                    if data.name.endswith('.csv'):
                        df = pd.read_csv(data)
                    elif data.name.endswith('.xlsx'):
                        df = pd.read_excel(data)
                    elif data.name.endswith('.json'):
                        df = pd.read_json(data)
                    else:
                        df = pd.read_csv(data, sep='\t')  # For txt files
                    
                    st.success("✅ Data successfully loaded")
                    st.session_state.current_df = df
                    st.session_state.data_loaded = True
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
                    return
            else:
                st.info("👆 Please upload a dataset to continue")
                return
                
        # Use container_width instead of column_width
        st.dataframe(df.head(50), use_container_width=True)  # Fixed here

        if st.checkbox("Display Shape"):
            st.write(df.shape)
        
        if st.checkbox("Display columns"):
            st.write(df.columns)
        
        selected_columns = st.multiselect("Select Preferred columns for EDA:", df.columns)
        if selected_columns:
            df1 = df[selected_columns]
            st.dataframe(df1, use_container_width=True)

        if st.checkbox("Display summary statistics"):
            st.write(df1.describe(include='all').T)

        if st.checkbox("Show missing values in selected columns"):
            missing = df1.isnull().sum()
            st.dataframe(missing[missing > 0].to_frame('Missing Values'), use_container_width=True)

        if st.checkbox("Generate WordCloud for Text Columns by Sentiment"):
            text_columns = [col for col in selected_columns if df1[col].dtype == 'object']
            rating_column = st.selectbox("Select the rating/sentiment column:", df.columns)

            if rating_column:
                try:
                    # Check if the column contains categorical sentiment values
                    unique_values = df[rating_column].dropna().unique()
                    
                    # Check for common sentiment categories (case-insensitive)
                    categorical_sentiments = set(['positive', 'negative', 'pos', 'neg', 'good', 'bad'])
                    is_categorical = any(str(val).lower() in categorical_sentiments for val in unique_values)
                    
                    if is_categorical:
                        # Handle categorical sentiment values
                        df[rating_column] = df[rating_column].astype(str).str.lower()
                        
                        # Define positive and negative categories
                        positive_categories = ['positive', 'pos', 'good']
                        negative_categories = ['negative', 'neg', 'bad']
                        
                        # Create masks for positive and negative sentiments
                        pos_mask = df[rating_column].isin(positive_categories)
                        neg_mask = df[rating_column].isin(negative_categories)
                        
                        pos_df = df[pos_mask]
                        neg_df = df[neg_mask]
                        
                        st.write(f"Found categorical sentiments. Positive: {pos_df.shape[0]} records, Negative: {neg_df.shape[0]} records")
                        
                    else:
                        # Handle numeric rating values
                        df[rating_column] = pd.to_numeric(df[rating_column], errors='coerce')
                        median_rating = df[rating_column].median()
                        pos_df = df[df[rating_column] >= median_rating]
                        neg_df = df[df[rating_column] < median_rating]
                        
                        st.write(f"Using median split at {median_rating}. Positive: {pos_df.shape[0]} records, Negative: {neg_df.shape[0]} records")

                    # Generate word clouds for each text column
                    for col in text_columns:
                        if pos_df.empty or neg_df.empty:
                            st.warning(f"Insufficient data for sentiment split in column {col}")
                            continue
                        
                        # Combine text for positive sentiment
                        pos_text = pos_df[col].dropna().astype(str).str.cat(sep=' ')
                        # Combine text for negative sentiment
                        neg_text = neg_df[col].dropna().astype(str).str.cat(sep=' ')
                        
                        # Check if there's enough text to generate word clouds
                        if len(pos_text.strip()) == 0 or len(neg_text.strip()) == 0:
                            st.warning(f"Insufficient text data for word cloud generation in column {col}")
                            continue

                        # Generate word clouds
                        pos_wc = WordCloud(width=800, height=400, background_color='white', 
                                        colormap='Greens').generate(pos_text)
                        neg_wc = WordCloud(width=800, height=400, background_color='white', 
                                        colormap='Reds').generate(neg_text)

                        # Display positive word cloud
                        st.write(f"**Positive WordCloud for {col}:**")
                        fig1, ax1 = plt.subplots(figsize=(10, 5))
                        ax1.imshow(pos_wc, interpolation='bilinear')
                        ax1.axis('off')
                        ax1.set_title('Positive Sentiment', fontsize=16, fontweight='bold')
                        st.pyplot(fig1)
                        plt.close(fig1)

                        # Display negative word cloud
                        st.write(f"**Negative WordCloud for {col}:**")
                        fig2, ax2 = plt.subplots(figsize=(10, 5))
                        ax2.imshow(neg_wc, interpolation='bilinear')
                        ax2.axis('off')
                        ax2.set_title('Negative Sentiment', fontsize=16, fontweight='bold')
                        st.pyplot(fig2)
                        plt.close(fig2)


            

                except Exception as e:
                    st.error(f"Failed to process sentiment word clouds: {e}")
                    st.write("Please ensure your data contains either:")
                    st.write("- Numeric ratings for median-based splitting, or")
                    st.write("- Categorical values like 'positive', 'negative', 'pos', 'neg', 'good', 'bad'")
        

        

    elif option == 'Visualisation':
        st.subheader("Data Visualization")

        # ⛔ Block access unless preprocessing is done
        if not st.session_state.get("preprocessing_done", False):
            st.warning("⚠️ Please complete the Data Preprocessing section before accessing Visualization.")
            st.stop()

        # ── 1. Load data safely ────────────────────────────────────────────────────
        if st.session_state.data_loaded and st.session_state.current_df is not None:
            # Work on a COPY so we never mutate the global dataframe
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

        # Preview (read-only)
        st.dataframe(df_vis.head(50), use_container_width=True)

        # ── 2. Column subset for visualisation ────────────────────────────────────
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

        # ── 3. Optional EDA helpers (work ONLY on df_subset or temporary Series) ──
        # 3-a Class imbalance
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

        # 3-b Basic text statistics
        if st.checkbox("Basic Text Statistics"):
            text_column = st.selectbox(
                "Select a text column:", df_subset.select_dtypes(include="object").columns
            )
            word_counts = df_subset[text_column].astype(str).apply(lambda x: len(x.split()))
            st.write("Average Word Count:", word_counts.mean())
            st.write("Max Word Count:", word_counts.max())

        # 3-c Average word length
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

        # 3-d Word-count distribution for positive vs. negative
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

        # 3-e N-gram analysis (no mutation)
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



                        
    elif option == 'Feature Engineering':
        st.subheader("Feature Engineering")

        if st.session_state.data_loaded and st.session_state.current_df is not None:
            df = st.session_state.current_df.copy()
            st.success("✅ Using previously loaded data")
        else:
            st.warning("⚠️ Please upload data in the Data Preprocessing section.")
            return

        st.dataframe(df.head(50), use_container_width=True)

        # Select text column
        st.sidebar.header("Column Selection")
        text_col = st.sidebar.selectbox("Select Text Column", df.columns)

        # Detect rating/sentiment columns
        rating_cols = [col for col in df.columns if 'rating' in col.lower() or 'score' in col.lower()]
        sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower() or 'label' in col.lower()]

        # --- Fast Preprocessing ---
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

        # --- Fast Labeling ---
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

        # --- Vectorizer Settings ---
        st.sidebar.header("Vectorizer Settings")
        vectorizer_type = st.sidebar.selectbox("Choose Vectorizer", ["CountVectorizer", "TF-IDF"])
        use_lemma_tokenizer = st.sidebar.checkbox("Use LemmaTokenizer (lemmatizes during vectorization)", value=True)
        ngram_min = st.sidebar.slider("N-gram Range Start", 1, 5, 1)
        ngram_max = st.sidebar.slider("N-gram Range End", ngram_min, 7, 3)
        min_df = st.sidebar.slider("Min Document Frequency", 1, 20, 10)
        max_features = st.sidebar.slider("Max Features", 100, 10000, 10000)
        # --- Train/Test Split Configuration ---
        st.sidebar.header("Train/Test Split")
        split_option = st.sidebar.radio("Set split manually?", ("Use default (80/20)", "Custom split"))

        if split_option == "Custom split":
            train_size = st.sidebar.slider("Training Set Size (%)", 50, 95, 80, step=5)
            test_size = 1.0 - (train_size / 100)
        else:
            test_size = 0.2  # Default 80% training, 20% testing


        # Cleaned Data
        clean_df = df.dropna(subset=['processed_text', 'label'])

        # Vectorization
        try:
            tok = LemmaTokenizer() if use_lemma_tokenizer else None
            kw = dict(ngram_range=(ngram_min, ngram_max), min_df=min_df, max_features=max_features)
            if use_lemma_tokenizer:
                kw['analyzer'] = 'word'
                kw['tokenizer'] = LemmaTokenizer()

            if vectorizer_type == "CountVectorizer":
                vect = CountVectorizer(**kw)
            else:
                vect = TfidfVectorizer(**kw)

            # Fit and split
            X = vect.fit_transform(clean_df['processed_text'])
            y = clean_df['label']
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Save in session state
            st.session_state.x_train = x_train
            st.session_state.x_test = x_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.vectorizer = vect
            st.session_state.vectorizer_type = vectorizer_type
            st.session_state.feature_engineering_done = True

            # Store original test texts for FP/FN analysis
            _, test_idx, _, _ = train_test_split(np.arange(len(clean_df)), y, test_size=test_size, random_state=42)
            st.session_state.x_test_texts = clean_df['processed_text'].iloc[test_idx].values

            st.success("✅ Vectorization and labeling complete!")
            st.write(f"X_train shape: {x_train.shape}")

        except ValueError as e:
            st.error(f"❌ Vectorization failed: {e}")
            st.stop()

        # --- Feature Scores ---
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

        # --- Chi2 Scores (Optional) ---
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



                
    elif option == 'Models':
        st.subheader("🤖 Machine Learning Models")

        # ✅ Ensure session state is initialized
        if "feature_engineering_done" not in st.session_state:
            st.session_state.feature_engineering_done = False

        if "models_results" not in st.session_state:
            st.session_state.models_results = []

        if not st.session_state.feature_engineering_done:
            st.warning("⚠️ Please complete Feature Engineering first to generate training and testing data.")
        else:
            x_train = st.session_state.x_train
            x_test = st.session_state.x_test
            y_train = st.session_state.y_train
            y_test = st.session_state.y_test

            # Enhanced model selection
            st.sidebar.header("🎛️ Model Configuration")
            seed = st.sidebar.slider('Random Seed', 1, 200, 42)
            classifier_name = st.sidebar.selectbox('Select your preferred classifier:',
                                                ('Logistic Regression', 'Decision Tree', 'Random Forest', 'AdaBoost', 'Naive Bayes'))

            def add_parameters(name_of_clf):
                params = {}
                
                if name_of_clf == 'Logistic Regression':
                    params['C'] = st.sidebar.slider('C (Inverse regularization)', 0.01, 10.0, 1.0)
                    params['max_iter'] = st.sidebar.slider('Max iterations', 50, 1000, 100)
                    params['solver'] = st.sidebar.selectbox('Solver', ['liblinear', 'lbfgs', 'newton-cg'])
                elif name_of_clf == 'Decision Tree':
                    params['max_depth'] = st.sidebar.slider('Max depth', 1, 30, 5)
                    params['min_samples_split'] = st.sidebar.slider('Min samples split', 2, 20, 2)
                    params['criterion'] = st.sidebar.selectbox('Criterion', ('gini', 'entropy'))
                elif name_of_clf == 'Random Forest':
                    params['n_estimators'] = st.sidebar.slider('Number of trees', 10, 200, 100)
                    params['max_depth'] = st.sidebar.slider('Max depth', 1, 30, 10)
                    params['min_samples_split'] = st.sidebar.slider('Min samples split', 2, 10, 2)
                elif name_of_clf == 'AdaBoost':
                    params['n_estimators'] = st.sidebar.slider('Number of estimators', 10, 200, 50)
                    params['learning_rate'] = st.sidebar.slider('Learning rate', 0.01, 2.0, 1.0)
                elif name_of_clf == 'Naive Bayes':
                    params['alpha'] = st.sidebar.slider('Alpha (smoothing)', 0.01, 10.0, 1.0)
                return params

            params = add_parameters(classifier_name)

            def get_classifier(name_of_clf, params):
                if name_of_clf == 'Logistic Regression':
                    return LogisticRegression(
                        C=params['C'], 
                        max_iter=params.get('max_iter', 100), 
                        solver=params.get('solver', 'liblinear'),
                        random_state=seed
                    )
                elif name_of_clf == 'Decision Tree':
                    return DecisionTreeClassifier(
                        max_depth=params['max_depth'], 
                        min_samples_split=params.get('min_samples_split', 2), 
                        criterion=params.get('criterion', 'gini'),
                        random_state=seed
                    )
                elif name_of_clf == 'Random Forest':
                    return RandomForestClassifier(
                        n_estimators=params['n_estimators'], 
                        max_depth=params['max_depth'],
                        min_samples_split=params.get('min_samples_split', 2),
                        random_state=seed
                    )
                elif name_of_clf == 'AdaBoost':
                    return AdaBoostClassifier(
                        n_estimators=params['n_estimators'], 
                        learning_rate=params['learning_rate'], 
                        random_state=seed
                    )
                elif name_of_clf == 'Naive Bayes':
                    return MultinomialNB(alpha=params.get('alpha', 1.0))
                return None

            if st.button(f"🚀 Train and Test {classifier_name}"):
                with st.spinner(f"Training {classifier_name}..."):
                    # Train model
                    clf = get_classifier(classifier_name, params)
                    start_time = time.time()
                    clf.fit(x_train, y_train)
                    # ✅ Store model and dependencies for XAI
                    st.session_state.trained_model = clf
                    st.session_state.X_test = x_test
                    st.session_state.y_test = y_test
                    st.session_state.tfidf_vectorizer = st.session_state.vectorizer

                    training_time = time.time() - start_time
                    
                    # Make predictions
                    y_pred = clf.predict(x_test)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')

                    # Display results
                    st.success(f"✅ {classifier_name} Training Completed!")
                    
                    # Enhanced metrics display
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.4f}")
                    with col2:
                        st.metric("Precision", f"{precision:.4f}")
                    with col3:
                        st.metric("Recall", f"{recall:.4f}")
                    with col4:
                        st.metric("F1-Score", f"{f1:.4f}")
                    
                    st.info(f"⏱️ Training Time: {training_time:.2f} seconds")
                    
                    # Detailed classification report
                    st.subheader("📋 Detailed Classification Report")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df, use_container_width=True)
                    
                    # Store results for comparison
                    model_result = {
                        'Model': classifier_name,
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'Recall': recall,
                        'F1_Score': f1,
                        'Training_Time': training_time,
                        'Parameters': params
                    }
                    
                    # Add to session state for comparison
                    st.session_state.models_results.append(model_result)
                    
                    # Export results
                    export_data = {
                        'model_type': classifier_name,
                        'vectorizer_type': type(st.session_state.vectorizer).__name__,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    results_df = pd.DataFrame([export_data])
                    csv = results_df.to_csv(index=False)

                    st.download_button(
                        label="📅 Download Results CSV",
                        data=csv,
                        file_name=f"sacr_results_{classifier_name.lower().replace(' ', '_')}.csv",
                        mime="text/csv",
                        key=f"download_csv_{classifier_name}"
                    )

                    # PDF download
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, txt="Sentiment Analysis Model Report", ln=True, align='C')
                    for k, v in export_data.items():
                        pdf.cell(200, 10, txt=f"{k}: {v}", ln=True, align='L')
                    pdf_output = f"sacr_results_{classifier_name.lower().replace(' ', '_')}.pdf"
                    pdf.output(pdf_output)
                    with open(pdf_output, "rb") as f:
                        st.download_button(
                            label="🔖 Download Results PDF",
                            data=f.read(),
                            file_name=pdf_output,
                            mime="application/pdf",
                            key=f"download_pdf_{classifier_name}"
                        )

                    # --- Confusion Matrix ---
                    with st.expander("📊 Confusion Matrix", expanded=False):
                        cm = confusion_matrix(y_test, y_pred)
                        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                        ConfusionMatrixDisplay(cm, display_labels=['Negative', 'Positive']).plot(ax=ax_cm)
                        plt.tight_layout()
                        st.pyplot(fig_cm)

                    # --- ROC-AUC Curve ---
                    with st.expander("📈 ROC-AUC Curve", expanded=False):
                        if hasattr(clf, "predict_proba"):
                            y_prob = clf.predict_proba(x_test)[:, 1]
                            fpr, tpr, _ = roc_curve(y_test, y_prob)
                            auc_score = roc_auc_score(y_test, y_prob)

                            fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
                            ax_roc.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.4f})")
                            ax_roc.plot([0, 1], [0, 1], "k--", label="Random")
                            ax_roc.set_xlabel("False Positive Rate")
                            ax_roc.set_ylabel("True Positive Rate")
                            ax_roc.set_title("ROC Curve")
                            ax_roc.legend()
                            plt.tight_layout()
                            st.pyplot(fig_roc)
                            st.metric("AUC Score", f"{auc_score:.4f}")
                        else:
                            st.info(f"{classifier_name} does not support probability predictions.")

                    # --- False Positive / False Negative Analysis ---
                    with st.expander("🔍 False Positive / False Negative Analysis", expanded=False):
                        fp = np.where((y_pred == 1) & (y_test == 0))[0]
                        fn = np.where((y_pred == 0) & (y_test == 1))[0]
                        st.write(f"**False Positives:** {len(fp)}  |  **False Negatives:** {len(fn)}")
                        if st.checkbox("Show False Positive examples", key=f"show_fp_{classifier_name}"):
                            x_test_texts = st.session_state.get("x_test_texts", None)
                            if x_test_texts is not None:
                                for idx in fp[:10]:
                                    st.text(f"[{idx}] {x_test_texts[idx][:300]}")
                            else:
                                st.info("Test texts not available. Re-run Feature Engineering.")
                        if st.checkbox("Show False Negative examples", key=f"show_fn_{classifier_name}"):
                            x_test_texts = st.session_state.get("x_test_texts", None)
                            if x_test_texts is not None:
                                for idx in fn[:10]:
                                    st.text(f"[{idx}] {x_test_texts[idx][:300]}")
                            else:
                                st.info("Test texts not available. Re-run Feature Engineering.")

                    # --- Download Trained Model ---
                    with st.expander("💾 Download Trained Model", expanded=False):
                        buf_model = io.BytesIO()
                        joblib.dump(clf, buf_model)
                        buf_model.seek(0)
                        st.download_button(
                            label="⬇️ Download Model (.joblib)",
                            data=buf_model,
                            file_name=f"{classifier_name.lower().replace(' ', '_')}_model.joblib",
                            mime="application/octet-stream",
                            key=f"download_model_{classifier_name}"
                        )
                        buf_vect = io.BytesIO()
                        joblib.dump(st.session_state.vectorizer, buf_vect)
                        buf_vect.seek(0)
                        st.download_button(
                            label="⬇️ Download Vectorizer (.joblib)",
                            data=buf_vect,
                            file_name=f"{classifier_name.lower().replace(' ', '_')}_vectorizer.joblib",
                            mime="application/octet-stream",
                            key=f"download_vect_{classifier_name}"
                        )


            # --- Real-time Prediction ---
            if st.session_state.get("trained_model") is not None:
                with st.expander("🔮 Test with Custom Text", expanded=True):
                    st.markdown("Type a review below to see what the trained model predicts.")
                    sample_text = st.text_area("Enter your text:", "", height=100,
                                                key=f"sample_text_{classifier_name}")
                    if st.button("Predict Sentiment", key=f"predict_btn_{classifier_name}"):
                        if sample_text.strip():
                            stop_words = get_stopwords()
                            cleaned = clean_text(sample_text, stop_words)
                            vec = st.session_state.vectorizer.transform([cleaned])
                            model = st.session_state.trained_model
                            pred = model.predict(vec)[0]
                            if hasattr(model, "predict_proba"):
                                proba = model.predict_proba(vec)[0]
                                confidence = proba[int(pred)]
                            else:
                                confidence = None

                            label = "✅ Positive" if pred == 1 else "❌ Negative"
                            st.success(f"**Prediction:** {label}")
                            if confidence is not None:
                                st.metric("Confidence", f"{confidence:.2%}")
                            st.caption(f"Cleaned text: _{cleaned[:200]}{'...' if len(cleaned) > 200 else ''}_")
                        else:
                            st.warning("Please enter some text to analyze.")


    elif option == 'Model Comparison':
        st.subheader("Model Performance Comparison")
        
        if st.session_state.models_results:
            compare_models(st.session_state.models_results)
            
            # Enhanced comparison visualizations
            if st.checkbox("📈 Show Performance Charts"):
                results_df = pd.DataFrame(st.session_state.models_results)
                
                # Performance metrics comparison
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.ravel()
                
                for i, metric in enumerate(metrics):
                    if metric in results_df.columns:
                        sns.barplot(data=results_df, x='Model', y=metric, ax=axes[i])
                        axes[i].set_title(f'{metric} Comparison')
                        axes[i].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Training time comparison
                if 'Training_Time' in results_df.columns:
                    st.subheader("⏱️ Training Time Comparison")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=results_df, x='Model', y='Training_Time', ax=ax)
                    ax.set_title('Training Time Comparison (seconds)')
                    ax.tick_params(axis='x', rotation=45)
                    st.pyplot(fig)
            
            # Clear results option
            if st.button("🗑️ Clear All Results"):
                st.session_state.models_results = []
                st.success("✅ All model results cleared!")
                st.rerun()

                
        else:
            st.info("🤔 No model results available yet. Train some models first!")
            
            # Quick training option
            if st.button("🚀 Quick Train and Test All Models"):
                if st.session_state.feature_engineering_done:
                    x_train = st.session_state.x_train
                    x_test = st.session_state.x_test
                    y_train = st.session_state.y_train
                    y_test = st.session_state.y_test

                    quick_models = {
                        'Logistic Regression': LogisticRegression(max_iter=100, solver='liblinear'),
                        'Decision Tree': DecisionTreeClassifier(),
                        'Random Forest': RandomForestClassifier(),
                        'AdaBoost': AdaBoostClassifier(),
                        'Naive Bayes': MultinomialNB()
                    }

                    for model_name, clf in quick_models.items():
                        with st.spinner(f"Training {model_name}..."):
                            clf.fit(x_train, y_train)
                            y_pred = clf.predict(x_test)

                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, average='weighted')
                            recall = recall_score(y_test, y_pred, average='weighted')
                            f1 = f1_score(y_test, y_pred, average='weighted')
                            training_time = None  # Skipped for simplicity

                            model_result = {
                                'Model': model_name,
                                'Accuracy': accuracy,
                                'Precision': precision,
                                'Recall': recall,
                                'F1_Score': f1,
                                'Training_Time': training_time,
                                'Parameters': 'Default'
                            }

                            st.session_state.models_results.append(model_result)

                    st.success("✅ All models trained and added to comparison!")
                    st.rerun()
            else:
                    st.warning("Please complete model training to get customized model comparision !")




    elif option == 'About Us':
        st.subheader("About Us")
    
    
        st.markdown("""
### Welcome to Sentiment Analyzer!

We are a team of developers passionate about Natural Language Processing and Machine Learning.  
Our mission is to help users analyze sentiments in text data using intuitive tools and powerful backend models.

What This App Does:
- Clean and preprocess text from various datasets
- Perform EDA and visualize patterns
- Train models using TF-IDF, CountVectorizer, and Logistic Regression
- Evaluate feature importance and sentiment patterns


---
### Made with 💙 using Python, Streamlit
""")


if __name__ == '__main__':
    web()

with st.sidebar:
    st.markdown("---")
    st.markdown("<div style='height: 250px;'></div>", unsafe_allow_html=True)  # Spacer
    if st.button("🔄 Reset All", help="Clear all session data and start fresh"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

