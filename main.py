from altair import param
from sklearn.base import is_classifier
from sklearn.discriminant_analysis import StandardScaler
import streamlit as st 
import numpy as np 
import seaborn as sns
import pandas as pd
import nltk
from nltk.corpus import stopwords
import contractions
import string
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2
from wordcloud import WordCloud
from collections import Counter


try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score, precision_score, recall_score, accuracy_score,precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn import model_selection


from PIL import Image

st.title("SACR Tool")

def get_stopwords():
    custom_stopwords = set(stopwords.words('english'))
    neutral_words = {'organization', 'company', 'work', 'worked', 'employee', 'employer', 'working', 'firm'}
    return custom_stopwords.union(neutral_words)

# Define text preprocessing function
def clean_text(text, stop_words):
    if not isinstance(text, str):
        return ''

    text = contractions.fix(text)
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered)


def web():
    activities = ['Data Preprocessing','EDA', 'Visualisation', 'Feature Engineering','models','About Us']
    option = st.sidebar.selectbox("Selection Option:", activities)

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
                
                st.success("Data successfully loaded")
                
                # Use container_width instead of column_width
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
                
                text_columns = [col for col in df.columns if df[col].dtype == 'object']
                selected_cols = st.multiselect("Select columns to use for text analysis:", text_columns)

                if selected_cols:
                    df['analysis_text'] = df[selected_cols].fillna('').agg(' '.join, axis=1)
                    stop_words = get_stopwords()
                    df['analysis_text_clean'] = df['analysis_text'].apply(lambda x: clean_text(x, stop_words))

                    st.success("Text preprocessing completed.")
                    st.dataframe(df[['analysis_text', 'analysis_text_clean']].head(30),use_container_width=True)
                else:
                    st.warning("Please select at least one text column.")

            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

    elif option == 'EDA':
        st.subheader("Exploratory Data Analysis")

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
                
                st.success("Data successfully loaded")
                
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
                    rating_column = st.selectbox("Select the rating column for sentiment split:", df.columns)

                    if rating_column:
                        try:
                            df[rating_column] = pd.to_numeric(df[rating_column], errors='coerce')
                            median_rating = df[rating_column].median()
                            pos_df = df[df[rating_column] >= median_rating]
                            neg_df = df[df[rating_column] < median_rating]

                            for col in text_columns:
                                pos_text = pos_df[col].dropna().astype(str).str.cat(sep=' ')
                                neg_text = neg_df[col].dropna().astype(str).str.cat(sep=' ')

                                pos_wc = WordCloud(width=800, height=400, background_color='white').generate(pos_text)
                                neg_wc = WordCloud(width=800, height=400, background_color='white').generate(neg_text)

                                st.write(f"Positive WordCloud for {col} (Rating >= {median_rating}):")
                                fig1, ax1 = plt.subplots(figsize=(10, 5))
                                ax1.imshow(pos_wc, interpolation='bilinear')
                                ax1.axis('off')
                                st.pyplot(fig1)

                                st.write(f"Negative WordCloud for {col} (Rating < {median_rating}):")
                                fig2, ax2 = plt.subplots(figsize=(10, 5))
                                ax2.imshow(neg_wc, interpolation='bilinear')
                                ax2.axis('off')
                                st.pyplot(fig2)

                        except Exception as e:
                            st.error(f"Failed to process sentiment word clouds: {e}")


                if st.checkbox("Distribution Plots for Numerical Columns"):
                    num_columns = [col for col in selected_columns if df1[col].dtype != 'object']
                    for col in num_columns:
                        fig, ax = plt.subplots()
                        sns.histplot(df1[col].dropna(), kde=True, ax=ax)
                        ax.set_title(f"Distribution of {col}")
                        st.pyplot(fig)
                else:
                    st.warning("Please select columns for EDA.")

            except Exception as e:
                st.error(f"Error loading file: {str(e)}")


    elif option == 'Visualisation':
        st.subheader("Data Visualization")

        data = st.file_uploader("Upload dataset:", type=['csv', 'xlsx', 'txt', 'json'])
        if data is not None:
            if data.name.endswith('.csv'):
                df = pd.read_csv(data)
            elif data.name.endswith('.xlsx'):
                df = pd.read_excel(data)
            elif data.name.endswith('.json'):
                df = pd.read_json(data)
            else:
                df = pd.read_csv(data, sep='\t')

            st.dataframe(df.head(50))

            if st.checkbox('Select Multiple columns to plot'):
                selected_columns = st.multiselect('Select your preferred columns', df.columns)
                df1 = df[selected_columns]
                st.dataframe(df1)

                # Heatmap
                if st.checkbox('Display Heatmap'):
                    fig, ax = plt.subplots()
                    sns.heatmap(df1.corr(), vmax=1, square=True, cmap='viridis', ax=ax)
                    st.pyplot(fig)

                # Pairplot
                if st.checkbox('Display Pairplot'):
                    fig = sns.pairplot(df1, diag_kind='kde')
                    st.pyplot(fig.fig)

                # Class Imbalance Check
                if st.checkbox('Class Imbalance Check'):
                    class_column = st.selectbox("Select target/class column", df.columns)
                    class_counts = df[class_column].value_counts()
                    class_proportions = class_counts / class_counts.sum()
                    st.write(class_proportions.to_frame('Proportion'))

                    # Display bar chart for class imbalance
                    fig, ax = plt.subplots()
                    sns.barplot(x=class_proportions.index.astype(str), y=class_proportions.values, ax=ax)
                    ax.set_title(f"Class Distribution for {class_column}")
                    ax.set_ylabel("Proportion")
                    ax.set_xlabel("Class Labels")
                    st.pyplot(fig)

                # Basic Stats of Text Data
                if st.checkbox('Basic Text Statistics'):
                    text_column = st.selectbox("Select a text column:", df.select_dtypes(include='object').columns)
                    df['word_count'] = df[text_column].astype(str).apply(lambda x: len(x.split()))
                    st.write("Average Word Count:", df['word_count'].mean())
                    st.write("Max Word Count:", df['word_count'].max())

                # Average Word Length in Text Column
                if st.checkbox('Average Word Length in Text Column'):
                    text_col = st.selectbox("Choose text column for word length analysis", df.select_dtypes(include='object').columns)
                    df['avg_word_len'] = df[text_col].astype(str).apply(lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0)
                    fig, ax = plt.subplots()
                    sns.histplot(df['avg_word_len'], kde=True, ax=ax)
                    ax.set_title(f"Average Word Length in {text_col}")
                    st.pyplot(fig)

                # Word Count in Positive vs Negative
                if st.checkbox('Compare Word Count in Positive vs Negative Reviews'):
                    rating_column = st.selectbox("Select rating column:", df.columns)
                    df[rating_column] = pd.to_numeric(df[rating_column], errors='coerce')
                    median_rating = df[rating_column].median()
                    df['word_count'] = df[text_col].astype(str).apply(lambda x: len(x.split()))
                    pos = df[df[rating_column] >= median_rating]['word_count']
                    neg = df[df[rating_column] < median_rating]['word_count']

                    fig, ax = plt.subplots()
                    sns.kdeplot(pos, label='Positive', shade=True)
                    sns.kdeplot(neg, label='Negative', shade=True)
                    ax.set_title('Word Count Distribution: Positive vs Negative')
                    ax.legend()
                    st.pyplot(fig)

                 # N-Gram Analysis
                if st.checkbox('N-Gram Analysis'):
                    try:
                        

                        ngram_col = st.selectbox("Select text column for N-Gram analysis:", df.select_dtypes(include='object').columns)
                        n_val = st.slider("Choose N for N-grams:", 1, 5, 2)
                        num_top = st.slider("Number of top N-grams to display:", 5, 50, 20)

                        vectorizer = CountVectorizer(ngram_range=(n_val, n_val), stop_words='english')
                        ngram_matrix = vectorizer.fit_transform(df[ngram_col].dropna().astype(str))
                        ngram_counts = ngram_matrix.sum(axis=0).A1
                        ngram_vocab = vectorizer.get_feature_names_out()
                        ngram_freq = dict(zip(ngram_vocab, ngram_counts))

                        top_ngrams = Counter(ngram_freq).most_common(num_top)
                        ngrams_df = pd.DataFrame(top_ngrams, columns=['N-gram', 'Frequency'])
                        st.dataframe(ngrams_df)

                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(data=ngrams_df, x='Frequency', y='N-gram', ax=ax)
                        ax.set_title(f"Top {num_top} {n_val}-grams in {ngram_col}")
                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Error in N-Gram analysis: {e}")

                        
    elif option == 'Feature Engineering':
        st.subheader("Feature Engineering")

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
                
                st.success("Data successfully loaded")
                
                # Use container_width instead of column_width
                st.dataframe(df.head(50), use_container_width=True)  # Fixed here
                 # Column selection
                st.sidebar.header("Column Selection")
                text_col = st.sidebar.selectbox("Select Text Column", df.columns)

                # --- Data Preprocessing ---
                st.subheader("🧹 Data Preprocessing & Lemmatization")
                stop_words = set(stopwords.words('english'))
                lemmatizer = WordNetLemmatizer()

                def preprocess_text(text):
                    text = str(text).lower()
                    text = re.sub(r'[^a-zA-Z\s]', '', text)
                    tokens = text.split()
                    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
                    return ' '.join(tokens)

                df['processed_text'] = df[text_col].apply(preprocess_text)
                st.write(df[['processed_text']].head())

                # Binary labeling (auto)
                st.subheader("🏷️ Binary Labeling")
                df['label'] = df['processed_text'].apply(lambda x: 1 if 'good' in x or 'excellent' in x or 'positive' in x else 0)
                st.write(df[['processed_text', 'label']].head())

                # Vectorizer settings
                st.sidebar.header("Vectorizer Settings")
                vectorizer_type = st.sidebar.selectbox("Choose Vectorizer", ["CountVectorizer", "TF-IDF"])
                ngram_min = st.sidebar.slider("N-gram Range Start", 1, 4, 1)
                ngram_max = st.sidebar.slider("N-gram Range End", ngram_min, 4, ngram_min)
                min_df = st.sidebar.slider("Min Document Frequency", 1, 20, 5)
                max_features = st.sidebar.slider("Max Features", 100, 10000, 1000)
                test_size = st.sidebar.slider("Test Size (fraction)", 0.1, 0.5, 0.3)

                

                # Clean and split
                df = df[['processed_text', 'label']].dropna()
                train, test = train_test_split(df, test_size=test_size, random_state=42)

                # Vectorization
                if vectorizer_type == "CountVectorizer":
                    vect = CountVectorizer(ngram_range=(ngram_min, ngram_max), min_df=min_df, max_features=max_features)
                else:
                    vect = TfidfVectorizer(ngram_range=(ngram_min, ngram_max), min_df=min_df, max_features=max_features)

                x_train = vect.fit_transform(train['processed_text'])
                x_test = vect.transform(test['processed_text'])
                y_train = train['label']
                y_test = test['label']
                st.session_state.x_train = x_train
                st.session_state.x_test = x_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test

                st.success("✅ Vectorization and labeling complete!")
                st.write(f"X_train shape: {x_train.shape}")


                # Show feature scores
                st.subheader("📊 Feature Scores")
                feature_names = vect.get_feature_names_out()
                if vectorizer_type == "CountVectorizer":
                    feature_scores = np.asarray(x_train.sum(axis=0)).flatten ()
                    score_type = "Frequency (CountVectorizer)"
                else:
                    feature_scores = np.asarray(x_train.mean(axis=0)).flatten()
                    score_type = "Mean TF-IDF Score"

                score_df = pd.DataFrame({"Feature": feature_names, "Score": feature_scores})
                score_df_sorted = score_df.sort_values(by="Score", ascending=False).head(20)
                st.write(f"🔢 Top 20 Features by {score_type}")
                st.write(score_df_sorted.reset_index(drop=True))

                # Checkbox for Chi2 Feature Importance
                compute_chi2 = st.checkbox("🔎 Show Top Features by Chi-Squared Score")

                # Show feature importance using Chi-Squared score
                if compute_chi2:
                    st.subheader("📈 Chi-Squared Feature Scores")
                    chi_scores, _ = chi2(x_train, y_train)
                    chi_df = pd.DataFrame({"Feature": feature_names, "Chi2 Score": chi_scores})
                    chi_df_sorted = chi_df.sort_values(by="Chi2 Score", ascending=False).head(20)
                    st.write(chi_df_sorted.reset_index(drop=True))

                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x="Chi2 Score", y="Feature", data=chi_df_sorted, ax=ax)
                    ax.set_title("Top 20 Features by Chi-Squared Score")
                    st.pyplot(fig)



            
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

                
    elif option=='models':
        st.subheader("Choose a classifier Model")

        if all(k in st.session_state for k in ['x_train', 'x_test', 'y_train', 'y_test']):
            x_train = st.session_state.x_train
            x_test = st.session_state.x_test
            y_train = st.session_state.y_train
            y_test = st.session_state.y_test

            seed = st.sidebar.slider('Seed', 1, 200, 42)
            classifier_name = st.sidebar.selectbox('Select your preferred classifier:',
                                                ( 'Logistic Regression',  'Decision Tree', 'Random Forest', 'AdaBoost'))

            def add_parameters(name_of_clf):
                params = {}
                
                if name_of_clf == 'Logistic Regression':
                    params['C'] = st.sidebar.slider('C (Inverse regularization)', 0.01, 10.0, 1.0)
                    params['max_iter'] = st.sidebar.slider('Max iterations', 50, 500, 100)
                elif name_of_clf == 'Decision Tree':
                    params['max_depth'] = st.sidebar.slider('Max depth', 1, 30, 5)
                    params['min_samples_split'] = st.sidebar.slider('Min samples split', 2, 20, 2)
                    params['criterion'] = st.sidebar.selectbox('Criterion', ('gini', 'entropy'))
                elif name_of_clf == 'Random Forest':
                    params['n_estimators'] = st.sidebar.slider('Number of trees', 10, 200, 100)
                    params['max_depth'] = st.sidebar.slider('Max depth', 1, 30, 10)
                elif name_of_clf == 'AdaBoost':
                    params['n_estimators'] = st.sidebar.slider('Number of estimators', 10, 200, 50)
                    params['learning_rate'] = st.sidebar.slider('Learning rate', 0.01, 2.0, 0.1)
                return params

            params = add_parameters(classifier_name)

            def get_classifier(name_of_clf, params):
                
                if name_of_clf == 'Logistic Regression':
                    return LogisticRegression(C=params['C'], max_iter=params.get('max_iter', 100), solver='liblinear')

                elif name_of_clf == 'Decision Tree':
                    return DecisionTreeClassifier(max_depth=params['max_depth'], min_samples_split=params.get('min_samples_split', 2), criterion=params.get('criterion', 'gini'))
                elif name_of_clf == 'Random Forest':
                    return RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=seed)
                elif name_of_clf == 'AdaBoost':
                    return AdaBoostClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'], random_state=seed)
                return None

            clf = get_classifier(classifier_name, params)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)

            st.write(f"Classifier: {classifier_name}")
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
            st.text("Classification Report")
            st.text(classification_report(y_test, y_pred))
        else:
            st.warning("Please complete Feature Engineering first to generate training and testing data.")




    
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



    st.sidebar.title("📚 Help Center")

    with st.sidebar.expander("🔰 How to Use This App"):
        st.markdown("""
                    **Steps:**
        1. Upload your dataset in the **Data Preprocessing** section.
        2. Explore your data in **EDA**.
        3. Visualize relationships under **Visualization**.
        4. Use **Feature Engineering** to preprocess and vectorize.
        5. Choose and train a model in **Models**.
                    
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
