import streamlit as st

st.set_page_config(page_title="SACR Tool — Complete Pipeline", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
import traceback
import sys as _sacr_sys
import warnings
import io
from collections import Counter

warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import chi2
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import nltk

# Module-level stopwords + cleaning (used both in the cached pipeline and for custom predictions)
_MODEL_STOP_WORDS = None

def _ensure_nltk():
    global _MODEL_STOP_WORDS
    if _MODEL_STOP_WORDS is not None:
        return
    # Support both punkt_tab (NLTK 3.9+) and punkt (older NLTK)
    for res_name in ['punkt_tab', 'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']:
        try:
            if res_name in ('punkt_tab', 'punkt'):
                try:
                    _ = nltk.data.find(f'tokenizers/{res_name}')
                except LookupError:
                    nltk.download(res_name, quiet=True)
            else:
                _ = nltk.data.find(f'corpora/{res_name}') if res_name in ('stopwords', 'wordnet') else \
                    nltk.data.find(f'taggers/{res_name}')
        except LookupError:
            nltk.download(res_name, quiet=True)
    try:
        _MODEL_STOP_WORDS = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        _MODEL_STOP_WORDS = set(stopwords.words('english'))
    _MODEL_STOP_WORDS.discard('not')
    _MODEL_STOP_WORDS.update(['would', 'shall', 'could', 'might'])

def _contraction_expansion(content):
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

class _LemmaTokenizer(object):
    def __init__(self):
        _ensure_nltk()
        self.wordnetlemma = WordNetLemmatizer()
    def __call__(self, reviews):
        _ensure_nltk()
        return [self.wordnetlemma.lemmatize(word) for word in word_tokenize(reviews)]

def _clean_text(content):
    _ensure_nltk()
    if not isinstance(content, str):
        return ''
    content = _contraction_expansion(content)
    content = re.sub(r'\W+', ' ', content)
    content = re.sub(r'http\S+', '', content)
    tokens = []
    for w in content.split():
        wl = w.strip().lower()
        if wl not in _MODEL_STOP_WORDS and wl.isalpha():
            tokens.append(wl)
    return ' '.join(tokens)

_ensure_nltk()

st.title("SACR Tool — Complete Sentiment Analysis Pipeline")
st.markdown("Mirrors the full 5‑phase notebook. Upload → automatic run → test at the bottom.")

if st.sidebar.button("Reset All — Start Fresh", type="primary", use_container_width=True):
    st.cache_data.clear()
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# ────────────────────────────────────────────────────────────
#  CONFIGURATION
# ────────────────────────────────────────────────────────────
st.sidebar.header("Mode")
mode = st.sidebar.radio("View:", ["Train Pipeline", "Test Predictions"], horizontal=True)

st.sidebar.header("Configuration")
INCLUDE_NEUTRAL = st.sidebar.checkbox("Keep neutral class (ratings 5‑6)", value=True,
    help="When ON: ratings ≥7→pos, ≤4→neg, 5‑6→neutral. When OFF: neutral rows are dropped.")
TEST_SIZE = st.sidebar.selectbox("Test set size", [0.2, 0.3, 0.4], index=0)
RANDOM_STATE = st.sidebar.number_input("Random seed", 1, 200, 42)
VECTORIZER_TYPE = st.sidebar.radio("Vectorizer", ["TF-IDF", "CountVectorizer"], horizontal=True)
NGRAM_MIN, NGRAM_MAX = 1, 3
MIN_DF = 10
MAX_FEATURES = 10000
NROWS = 500_000

# ────────────────────────────────────────────────────────────
#  CACHED PIPELINE — runs once per (file, config) combo
# ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline(file_bytes, include_neutral, test_size, random_state, vectorizer_type):
    _ensure_nltk()
    import io as _io
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(_io.BytesIO(file_bytes), encoding=enc, nrows=NROWS)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        df = pd.read_csv(_io.BytesIO(file_bytes), encoding='utf-8', errors='replace', nrows=NROWS)
    # ... (full pipeline from Phase 1 through Phase 4, returning all artifacts)
    # Detect text column
    text_col = None
    for col in df.columns:
        if df[col].dtype == 'object':
            avg_len = df[col].astype(str).str.len().mean()
            if avg_len > 50:
                text_col = col
                break
    if text_col is None:
        for col in df.columns:
            if df[col].dtype == 'object':
                text_col = col
                break
    if text_col is None:
        for col in df.columns:
            if df[col].astype(str).str.len().mean() > 50:
                text_col = col
                break
    if text_col is None:
        for col in df.columns:
            if df[col].nunique() > 10:
                text_col = col
                break
    if text_col is None:
        col_info = {col: f"dtype={df[col].dtype}, nunique={df[col].nunique()}" for col in df.columns}
        return {'error': f'No text column found. Columns: {col_info}'}

    potential_sentiment_cols = [c for c in df.columns if 'sentiment' in c.lower() or 'label' in c.lower()]
    NUMERIC_LABEL_KEYWORDS = ['rating', 'score', 'star', 'target', 'polarity', 'class', 'sentiment', 'label']
    rating_cols = [c for c in df.columns if any(k in c.lower() for k in NUMERIC_LABEL_KEYWORDS)]

    # Fallback: if nothing matched by name, any numeric column with 2-10 distinct values
    if not potential_sentiment_cols and not rating_cols:
        for col in df.columns:
            if col != text_col and pd.api.types.is_numeric_dtype(df[col]) and 2 <= df[col].nunique() <= 10:
                rating_cols = [col]
                break

    # Drop null text
    df = df.dropna(subset=[text_col]).copy()

    # Label Creation
    target_col = None
    label_source = None
    if potential_sentiment_cols:
        target_col = potential_sentiment_cols[0]
        label_source = 'categorical'
    elif rating_cols:
        target_col = rating_cols[0]
        label_source = 'numeric'

    if target_col is None:
        df['label_raw'] = np.where(
            df[text_col].astype(str).str.contains(
                r'\b(good|excellent|positive|amazing|great|wonderful)\b', case=False, na=False),
            'positive', 'negative'
        )
    elif label_source == 'categorical':
        df['label_raw'] = df[target_col].astype(str).str.strip().str.lower()
    else:
        vals = df[target_col].dropna()
        unique_vals = sorted(vals.unique())
        n_unique = len(unique_vals)

        if n_unique == 2:
            lo, hi = unique_vals
            df['label_raw'] = df[target_col].map({lo: 'negative', hi: 'positive'})
        elif n_unique == 3:
            lo, mid, hi = unique_vals
            df['label_raw'] = df[target_col].map({lo: 'negative', mid: 'neutral', hi: 'positive'})
            if not include_neutral:
                df = df[df['label_raw'] != 'neutral']
        else:
            vmin, vmax = unique_vals[0], unique_vals[-1]
            rng = vmax - vmin
            pos_cut = vmin + (2 / 3) * rng
            neg_cut = vmin + (1 / 3) * rng

            def map_rating(x):
                if x >= pos_cut:
                    return 'positive'
                elif x <= neg_cut:
                    return 'negative'
                else:
                    return 'neutral'
            df['label_raw'] = df[target_col].apply(map_rating)
            if not include_neutral:
                df = df[df['label_raw'] != 'neutral']

    df = df.dropna(subset=['label_raw']).copy()

    # Encode labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label_raw'])
    class_names = list(le.classes_)
    n_classes = len(class_names)

    df['processed_text'] = df[text_col].astype(str).apply(_clean_text)
    clean_df = df.dropna(subset=['processed_text', 'label']).copy()
    clean_df = clean_df[clean_df['processed_text'].str.strip() != '']

    # Train/Test split BEFORE vectorization
    train_df, test_df = train_test_split(
        clean_df, test_size=test_size, random_state=random_state, stratify=clean_df['label']
    )

    if vectorizer_type == 'CountVectorizer':
        vect = CountVectorizer(analyzer='word', tokenizer=_LemmaTokenizer(),
                               ngram_range=(NGRAM_MIN, NGRAM_MAX), min_df=MIN_DF, max_features=MAX_FEATURES)
    else:
        vect = TfidfVectorizer(analyzer='word', tokenizer=_LemmaTokenizer(),
                                ngram_range=(NGRAM_MIN, NGRAM_MAX), min_df=MIN_DF, max_features=MAX_FEATURES)

    x_train = vect.fit_transform(train_df['processed_text'])
    x_test = vect.transform(test_df['processed_text'])
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    feature_names = vect.get_feature_names_out()

    # Train all models
    SEED = random_state
    USE_GRID_SEARCH = True
    NB_ALPHA = 0.5
    USE_CV_SCORES = True
    USE_VOTING = True
    USE_CALIBRATION = True

    # Optional GridSearch for Logistic Regression
    if USE_GRID_SEARCH:
        from sklearn.model_selection import GridSearchCV
        param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
        gs = GridSearchCV(LogisticRegression(max_iter=200, solver='lbfgs', random_state=SEED, class_weight='balanced'),
                          param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
        gs.fit(x_train, y_train)
        best_c = gs.best_params_['C']
    else:
        best_c = 10

    # SMOTE resampling before training
    try:
        from imblearn.over_sampling import SMOTE as _SMOTE
        if _SMOTE is not None:
            smote = _SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
            x_train, y_train = smote.fit_resample(x_train, y_train)
    except ImportError:
        pass

    classifiers = {
        'Logistic Regression': LogisticRegression(C=best_c, max_iter=200, solver='lbfgs', random_state=SEED, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_split=2, criterion='gini', random_state=SEED, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, random_state=SEED, class_weight='balanced'),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=SEED),
        'Naive Bayes': MultinomialNB(alpha=NB_ALPHA)
    }

    results = []
    trained_models = {}
    raw_models = {}
    for name, clf in classifiers.items():
        start = time.time()
        clf.fit(x_train, y_train)
        training_time = time.time() - start
        raw_clf = clf

        if USE_CALIBRATION:
            from sklearn.calibration import CalibratedClassifierCV
            cal_clf = CalibratedClassifierCV(clf, cv=3)
            cal_clf.fit(x_train, y_train)
            clf = cal_clf

        y_pred = clf.predict(x_test)
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'F1_Weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'F1_Macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'Training_Time': training_time
        })
        trained_models[name] = clf
        raw_models[name] = raw_clf

    # Voting Ensemble
    if USE_VOTING and len(trained_models) > 1:
        from sklearn.ensemble import VotingClassifier
        voting_clf = VotingClassifier(
            estimators=[(name.replace(' ', '_'), clf) for name, clf in trained_models.items()],
            voting='soft' if all(hasattr(clf, 'predict_proba') for clf in trained_models.values()) else 'hard'
        )
        voting_clf.fit(x_train, y_train)
        y_pred_vote = voting_clf.predict(x_test)
        trained_models['Voting Ensemble'] = voting_clf
        raw_models['Voting Ensemble'] = voting_clf
        results.append({
            'Model': 'Voting Ensemble',
            'Accuracy': accuracy_score(y_test, y_pred_vote),
            'Precision': precision_score(y_test, y_pred_vote, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, y_pred_vote, average='weighted', zero_division=0),
            'F1_Weighted': f1_score(y_test, y_pred_vote, average='weighted', zero_division=0),
            'F1_Macro': f1_score(y_test, y_pred_vote, average='macro', zero_division=0),
            'Training_Time': 0.0
        })

    # Exclude Voting Ensemble from best-model selection
    best_results = pd.DataFrame(results)
    base_results = best_results[best_results['Model'] != 'Voting Ensemble']
    if base_results.empty:
        base_results = best_results
    best_model_name = base_results.sort_values('F1_Weighted', ascending=False).iloc[0]['Model']
    best_clf = trained_models[best_model_name]
    best_pipeline = Pipeline(steps=[('vect', vect), ('clf', best_clf)])

    # Find misclassified indices
    y_pred_all = best_clf.predict(x_test)
    misclassified_idx = np.where(y_pred_all != y_test)[0]

    return {
        'df': df,
        'text_col': text_col,
        'class_names': class_names,
        'n_classes': n_classes,
        'le': le,
        'vect': vect,
        'feature_names': feature_names,
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_df': train_df,
        'test_df': test_df,
        'results': results,
        'results_df': base_results.sort_values('F1_Weighted', ascending=False),
        'trained_models': trained_models,
        'raw_models': raw_models,
        'best_model_name': best_model_name,
        'best_clf': best_clf,
        'best_pipeline': best_pipeline,
        'y_pred_best': best_clf.predict(x_test),
        'y_pred_all': y_pred_all,
        'misclassified_idx': misclassified_idx,
        'test_texts': test_df['processed_text'].values,
    }


# ────────────────────────────────────────────────────────────
#  LOAD DATASET & RUN PIPELINE
# ────────────────────────────────────────────────────────────
data_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])

# Allow loading a pre-trained model from a zip or standalone joblib
model_zip = None
model_joblib = None
if data_file is None and mode == "Test Predictions":
    col1, col2 = st.columns(2)
    with col1:
        model_zip = st.file_uploader("Upload model package (.zip)", type=["zip"],
                                      help="Zip with best_pipeline.joblib, vectorizer.joblib, "
                                           "label_encoder.joblib, meta.json (from sacr_cli.py train)")
    with col2:
        model_joblib = st.file_uploader("Or upload pipeline (.joblib)", type=["joblib"],
                                         help="A single best_pipeline.joblib file (Pipeline with vectorizer + classifier)")

if data_file is None and model_zip is None and model_joblib is None:
    if mode == "Test Predictions":
        st.warning("Upload a CSV to train, or upload a saved model (.joblib / .zip).", icon="⚠️")
    st.stop()

if model_zip is not None:
    import zipfile, io as _io, json
    with zipfile.ZipFile(_io.BytesIO(model_zip.getvalue())) as z:
        z.extractall("_loaded_model")
    import joblib as _joblib
    best_pipeline = _joblib.load("_loaded_model/best_pipeline.joblib")
    vect = _joblib.load("_loaded_model/vectorizer.joblib")
    le = _joblib.load("_loaded_model/label_encoder.joblib")
    trained_models = _joblib.load("_loaded_model/trained_models.joblib")
    raw_models = _joblib.load("_loaded_model/raw_models.joblib")
    with open("_loaded_model/meta.json") as f:
        meta = json.load(f)
    class_names = meta['class_names']
    n_classes = meta['n_classes']
    best_model_name = meta['best_model_name']
    best_clf = trained_models[best_model_name]
    pipe = {
        'class_names': class_names, 'n_classes': n_classes, 'le': le, 'vect': vect,
        'trained_models': trained_models, 'raw_models': raw_models,
        'best_model_name': best_model_name, 'best_clf': best_clf,
        'best_pipeline': best_pipeline, 'results': [], 'results_df': pd.DataFrame(),
    }

elif model_joblib is not None:
    import joblib as _joblib
    best_pipeline = _joblib.load(io.BytesIO(model_joblib.getvalue()))
    # If it's a Pipeline (vect + clf), extract components by checking each step
    if hasattr(best_pipeline, 'named_steps'):
        clf_step = None
        vect_step = None
        for step_name in best_pipeline.named_steps:
            step = best_pipeline.named_steps[step_name]
            if hasattr(step, 'transform') and not hasattr(step, 'predict'):
                vect_step = step_name
            elif hasattr(step, 'predict'):
                clf_step = step_name
        if clf_step:
            best_clf = best_pipeline.named_steps[clf_step]
        else:
            best_clf = list(best_pipeline.named_steps.values())[-1]
        vect = best_pipeline.named_steps.get(vect_step) if vect_step else None
    else:
        # Standalone classifier (e.g. logistic_regression_model.joblib) — no vectorizer
        best_clf = best_pipeline
        vect = None
        st.warning("⚠️ This .joblib file is a standalone classifier without a vectorizer. "
                   "Upload a full Pipeline (.joblib with vectorizer + classifier) to make predictions, "
                   "or use a model package (.zip) from `sacr_cli.py train`.", icon="⚠️")
    # Detect class names from the classifier
    if hasattr(best_clf, 'classes_'):
        class_names = list(best_clf.classes_)
    elif hasattr(best_clf, 'calibrated_classifiers_'):
        class_names = list(best_clf.calibrated_classifiers_[0].classes_)
    else:
        class_names = ['negative', 'positive']
    n_classes = len(class_names)
    best_model_name = "Uploaded Model"
    le = None
    trained_models = {best_model_name: best_clf}
    raw_models = trained_models
    pipe = {
        'class_names': class_names, 'n_classes': n_classes, 'le': le, 'vect': vect,
        'trained_models': trained_models, 'raw_models': raw_models,
        'best_model_name': best_model_name, 'best_clf': best_clf,
        'best_pipeline': best_pipeline, 'results': [], 'results_df': pd.DataFrame(),
    }

else:
    with st.spinner("Running full pipeline (cleaning, vectorization, training 5 models)..."):
        pipe = run_pipeline(
            data_file.getvalue(), INCLUDE_NEUTRAL, TEST_SIZE, RANDOM_STATE, VECTORIZER_TYPE
        )

if 'error' in pipe:
    st.error(pipe['error'])
    st.stop()

# Unpack (always needed for both modes)
df = pipe.get('df', pd.DataFrame()); text_col = pipe.get('text_col', '')
class_names = pipe['class_names']; n_classes = pipe['n_classes']
le = pipe['le']; vect = pipe['vect']
feature_names = pipe.get('feature_names', vect.get_feature_names_out() if hasattr(vect, 'get_feature_names_out') else [])
x_train = pipe.get('x_train', None); x_test = pipe.get('x_test', None)
y_train = pipe.get('y_train', None); y_test = pipe.get('y_test', None)
train_df = pipe.get('train_df', None); test_df = pipe.get('test_df', None)
results = pipe.get('results', []); results_df = pipe.get('results_df', pd.DataFrame())
trained_models = pipe['trained_models']
raw_models = pipe.get('raw_models', trained_models)
best_model_name = pipe['best_model_name']; best_clf = pipe['best_clf']
best_pipeline = pipe['best_pipeline']
y_pred_best = pipe.get('y_pred_best', best_clf.predict(x_test) if x_test is not None else [])
y_pred_all = pipe.get('y_pred_all', y_pred_best)
misclassified_idx = pipe.get('misclassified_idx', [])
test_texts = pipe.get('test_texts', [])

if mode == "Train Pipeline":
    # ── Full pipeline UI (heavy, rendered only in Train mode) ──

    # ═══════════════════════════════════════════════════════════
    #  PHASE 1 : UI — DATA PREPROCESSING & EDA
    # ═══════════════════════════════════════════════════════════
    st.header("Phase 1 — Data Preprocessing & EDA")

    with st.expander("Dataset Info & Missing Values", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Shape", f"{df.shape[0]} rows × {df.shape[1]} cols")
            buf = io.StringIO()
            df.info(buf=buf)
            st.text(buf.getvalue())
        with col2:
            missing = df.isnull().sum()
            missing = missing[missing > 0]
            if not missing.empty:
                st.write("Columns with missing values:")
                st.dataframe(missing.to_frame("Missing"))
            else:
                st.success("No missing values.")

    potential_sentiment_cols = [c for c in df.columns if 'sentiment' in c.lower() or 'label' in c.lower()]
    rating_cols = [c for c in df.columns if 'rating' in c.lower() or 'score' in c.lower()]
    st.info(f"**Text column:** `{text_col}`  |  **Sentiment cols:** {potential_sentiment_cols}  |  **Rating cols:** {rating_cols}")

    st.write(f"**Classes:** {class_names}  |  **n_classes:** {n_classes}")
    imb = df['label'].value_counts().max() / df['label'].value_counts().min()
    if imb > 1.5:
        st.write(f"⚠️ Imbalanced (ratio {imb:.2f}).")

    col1, col2 = st.columns(2)
    with col1:
        df_viz = df.copy()
        df_viz['text_length'] = df_viz[text_col].astype(str).apply(len)
        df_viz['word_count'] = df_viz[text_col].astype(str).apply(lambda x: len(x.split()))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df_viz['text_length'], bins=50, edgecolor='black')
        ax.set_title('Text Length Distribution')
        ax.set_xlabel('Length (chars)')
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df_viz['word_count'], bins=50, edgecolor='black', color='orange')
        ax.set_title('Word Count Distribution')
        ax.set_xlabel('Word Count')
        st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors_bar = ['green' if c == 'positive' else 'blue' if c == 'neutral' else 'red' for c in class_names]
    df['label_raw'].value_counts().plot(kind='bar', ax=ax, color=colors_bar)
    ax.set_title('Class Distribution')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    st.subheader("Word Cloud per Class")
    n_show = min(n_classes, 3)
    ncols_wc = min(3, n_classes)
    fig, axes = plt.subplots(1, ncols_wc, figsize=(7 * ncols_wc, 6))
    if ncols_wc == 1:
        axes = [axes]
    cmap_map = {'positive': 'Greens', 'neutral': 'Blues', 'negative': 'Reds'}
    for i, cls in enumerate(class_names[:n_show]):
        texts_ = ' '.join(df[df['label_raw'] == cls][text_col].astype(str).head(500))
        cmap_wc = cmap_map.get(cls.lower(), 'viridis')
        wc = WordCloud(width=500, height=400, background_color='white', colormap=cmap_wc, max_words=100).generate(texts_)
        axes[i].imshow(wc, interpolation='bilinear')
        axes[i].axis('off')
        axes[i].set_title(f'{cls.title()} Reviews', color=cmap_wc.replace('s', ''))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    #  PHASE 2 : UI — FEATURE ENGINEERING
    # ═══════════════════════════════════════════════════════════
    st.header("Phase 2 — Feature Engineering")
    st.write(f"Custom stopwords loaded ({len(_MODEL_STOP_WORDS)} words) — kept 'not'.")
    st.dataframe(df[['processed_text']].head(10), use_container_width=True)
    st.write(f"After cleaning: {len(train_df) + len(test_df)} rows.")
    st.write(f"Train: {train_df.shape}, Test: {test_df.shape}")
    st.write(f"Train balance:\n{train_df['label_raw'].value_counts(normalize=True).round(3)}")
    st.write(f"Test balance:\n{test_df['label_raw'].value_counts(normalize=True).round(3)}")
    st.success(f"x_train: {x_train.shape}  |  x_test: {x_test.shape}")
    st.write(f"y_train distribution: {np.bincount(y_train)}")
    st.write(f"y_test distribution: {np.bincount(y_test)}")

    st.subheader("Top Features by Score")
    if VECTORIZER_TYPE == "CountVectorizer":
        scores = np.asarray(x_train.sum(axis=0)).flatten()
    else:
        scores = np.asarray(x_train.mean(axis=0)).flatten()
    score_df = pd.DataFrame({"Feature": feature_names, "Score": scores}).sort_values("Score", ascending=False).head(20)
    st.dataframe(score_df.reset_index(drop=True), use_container_width=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=score_df, x="Score", y="Feature", ax=ax, palette="viridis")
    ax.set_title(f"Top 20 Features ({VECTORIZER_TYPE})")
    st.pyplot(fig)

    st.subheader("Chi-Squared Feature Selection")
    chi_scores, _ = chi2(x_train, y_train)
    chi_df = pd.DataFrame({"Feature": feature_names, "Chi2 Score": chi_scores}).sort_values("Chi2 Score", ascending=False).head(20)
    st.dataframe(chi_df.reset_index(drop=True), use_container_width=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Chi2 Score", y="Feature", data=chi_df, ax=ax)
    ax.set_title("Top 20 Features by Chi-Squared Score")
    st.pyplot(fig)

    # ═══════════════════════════════════════════════════════════
    #  PHASE 3 : UI — MODEL SELECTION
    # ═══════════════════════════════════════════════════════════
    st.header("Phase 3 — Model Selection")
    for r in results:
        st.write(f"**{r['Model']}** — Acc: {r['Accuracy']:.4f} | F1(w): {r['F1_Weighted']:.4f} | F1(m): {r['F1_Macro']:.4f} | Time: {r['Training_Time']:.2f}s")
        if r['Accuracy'] >= 0.999:
            st.error(f"⚠️ {r['Model']}: accuracy {r['Accuracy']:.4f} — possible data leakage!")
    st.subheader("Results Summary")
    st.dataframe(results_df.set_index('Model'), use_container_width=True)

    st.subheader("Performance Comparison Charts")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Weighted', 'F1_Macro']
    for i, metric in enumerate(metrics):
        sns.barplot(x='Model', y=metric, data=results_df, ax=axes[i])
        axes[i].set_title(f'{metric} Comparison')
        axes[i].tick_params(axis='x', rotation=45)
        for j, v in enumerate(results_df[metric]):
            axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    axes[-1].axis('off')
    plt.tight_layout()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='Model', y='Training_Time', data=results_df, ax=ax, palette='viridis')
    ax.set_title('Training Time Comparison')
    ax.set_ylabel('Time (seconds)')
    for i, v in enumerate(results_df['Training_Time']):
        ax.text(i, v + 0.01, f'{v:.2f}s', ha='center', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)

    # ═══════════════════════════════════════════════════════════
    #  PHASE 4 : UI — MODEL EVALUATION & XAI
    # ═══════════════════════════════════════════════════════════
    st.header("Phase 4 — Model Evaluation & XAI")
    st.success(f"**Best model (by weighted F1):** {best_model_name}")
    st.subheader(f"Detailed Classification Report — {best_model_name}")
    report = classification_report(y_test, y_pred_best, target_names=class_names, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix'):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5 + len(class_names), 4 + len(class_names) // 2))
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap='Blues', values_format='d')
        plt.title(title)
        plt.tight_layout()
        return fig
    st.pyplot(plot_confusion_matrix(y_test, y_pred_best, class_names, f'Confusion Matrix — {best_model_name}'))

    st.subheader("ROC-AUC Curves")
    if hasattr(best_clf, 'predict_proba'):
        y_prob = best_clf.predict_proba(x_test)
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            auc_score = roc_auc_score(y_test, y_prob[:, 1])
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.4f})', lw=2)
            ax.plot([0, 1], [0, 1], 'k--', label='Random')
            ax.set_xlabel('FPR'), ax.set_ylabel('TPR')
            ax.set_title(f'ROC Curve — {best_model_name}')
            ax.legend(); plt.tight_layout(); st.pyplot(fig)
        else:
            y_test_bin = label_binarize(y_test, classes=range(n_classes))
            auc_macro = roc_auc_score(y_test_bin, y_prob, average='macro', multi_class='ovr')
            fig, ax = plt.subplots(figsize=(7, 6))
            for i, cname in enumerate(class_names):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                class_auc = roc_auc_score(y_test_bin[:, i], y_prob[:, i])
                ax.plot(fpr, tpr, label=f'{cname} (AUC={class_auc:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', label='Random')
            ax.set_xlabel('FPR'), ax.set_ylabel('TPR')
            ax.set_title(f'ROC Curves (OvR) — {best_model_name}')
            ax.legend(); plt.tight_layout(); st.pyplot(fig)
            st.metric("Macro-average AUC", f"{auc_macro:.4f}")

    st.subheader("Confusion Matrices — All Models")
    n_models = len(trained_models)
    n_cols_m = min(3, n_models)
    n_rows_m = -(-n_models // n_cols_m)
    fig, axes = plt.subplots(n_rows_m, n_cols_m, figsize=(6 * n_cols_m, 5 * n_rows_m))
    axes = np.array(axes).ravel()
    for i, (name, clf) in enumerate(trained_models.items()):
        y_pred = clf.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=axes[i], cmap='Blues', values_format='d')
        axes[i].set_title(name)
    for j in range(n_models, len(axes)):
        axes[j].axis('off')
    plt.tight_layout(); st.pyplot(fig)

    st.subheader("Misclassification Analysis")
    st.write(f"**Total misclassified:** {len(misclassified_idx)} / {len(y_test)} ({len(misclassified_idx)/len(y_test):.1%})")
    for true_cls in range(n_classes):
        for pred_cls in range(n_classes):
            if true_cls == pred_cls:
                continue
            idxs = np.where((y_test == true_cls) & (y_pred_all == pred_cls))[0]
            if len(idxs) == 0:
                continue
            label = f"True: {class_names[true_cls]} → Pred: {class_names[pred_cls]} ({len(idxs)} cases)"
            if st.checkbox(label, key=f"mis_{true_cls}_{pred_cls}"):
                for idx in idxs[:5]:
                    st.text(f"[{idx}] {test_texts[idx][:200]}")

    st.subheader("Feature Importance")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    ax_idx = 0
    for name in ['Random Forest', 'Decision Tree', 'AdaBoost', 'Logistic Regression']:
        clf = raw_models.get(name)
        if clf is None:
            continue
        if hasattr(clf, 'coef_'):
            coef = clf.coef_
            importances = np.abs(coef).mean(axis=0) if coef.shape[0] > 1 else np.abs(coef[0])
        elif hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
        else:
            continue
        top_n = 15
        top_idx = np.argsort(importances)[-top_n:]
        axes[ax_idx].barh(range(top_n), importances[top_idx], color='steelblue')
        axes[ax_idx].set_yticks(range(top_n))
        axes[ax_idx].set_yticklabels([feature_names[i] for i in top_idx])
        axes[ax_idx].set_title(f'Top {top_n} Features — {name}')
        axes[ax_idx].invert_yaxis(); ax_idx += 1
    for j in range(ax_idx, len(axes)):
        axes[j].axis('off')
    plt.tight_layout(); st.pyplot(fig)

    # ═══════════════════════════════════════════════════════════
    #  PHASE 5 : UI — EXPLAINABILITY (XAI)
    # ═══════════════════════════════════════════════════════════
    st.header("Phase 5 — Explainability (XAI)")
    st.write(f"Pipeline ready for: **{best_model_name}**")

    tabs = st.tabs(["SHAP (Global)", "SHAP (Individual)", "LIME", "All 5 Models"])

    with tabs[0]:
        st.markdown("**SHAP beeswarm** — global feature contribution ranking.")
        max_samples = st.slider("SHAP sample size", 50, 500, 200, key="shap_global_samples")
        if st.button("Run SHAP (Global)", key="shap_global_btn"):
            with st.spinner("Computing SHAP..."):
                try:
                    import shap
                    bg = min(max_samples, x_train.shape[0])
                    ts = min(max_samples, x_test.shape[0])
                    x_tr_d = x_train[:bg].toarray()
                    x_te_d = x_test[:ts].toarray()
                    raw_best = raw_models.get(best_model_name, best_clf)
                    if hasattr(raw_best, 'coef_'):
                        se = shap.LinearExplainer(raw_best, x_tr_d, feature_names=feature_names)
                        sv = se(x_te_d)
                    elif hasattr(raw_best, 'feature_importances_'):
                        se = shap.TreeExplainer(raw_best, feature_names=feature_names)
                        sv = se(x_te_d)
                    else:
                        st.error("Unsupported model type."); sv = None
                    if sv is not None:
                        is_linear = isinstance(sv, list)
                        if n_classes == 2:
                            sv_plot = sv[1] if is_linear else sv
                            shap.summary_plot(sv_plot, x_te_d, feature_names=feature_names, show=False)
                            st.pyplot(plt.gcf()); plt.clf()
                        else:
                            for ci, cn in enumerate(class_names):
                                st.write(f"**Class: {cn}**")
                                sv_plot = sv[:, :, ci] if not is_linear else sv[ci]
                                shap.summary_plot(sv_plot, x_te_d, feature_names=feature_names, show=False)
                                st.pyplot(plt.gcf()); plt.clf()
                except ImportError:
                    st.error("`shap` not installed. Run `pip install shap`.")
                except Exception as e:
                    st.error(f"SHAP error: {e}")

    with tabs[1]:
        st.markdown("**SHAP force plot** on a misclassified example.")
        if len(misclassified_idx) == 0:
            st.info("No misclassified examples.")
        else:
            shap_ts = min(300, x_test.shape[0])
            mis_in = misclassified_idx[misclassified_idx < shap_ts]
            if len(mis_in) == 0:
                st.info("No misclassified in SHAP sample.")
            else:
                idx_choice = st.selectbox("Misclassified index:", mis_in[:10], key="shap_indiv_idx",
                    format_func=lambda i: f"Index {i} — True: {class_names[y_test[i]]}, Pred: {class_names[y_pred_all[i]]}")
                if st.button("Explain with SHAP", key="shap_indiv_btn"):
                    try:
                        import shap
                        bg = min(200, x_train.shape[0])
                        x_tr_d = x_train[:bg].toarray()
                        x_te_d = x_test[:shap_ts].toarray()
                        raw_best = raw_models.get(best_model_name, best_clf)
                        if hasattr(raw_best, 'coef_'):
                            ex = shap.LinearExplainer(raw_best, x_tr_d, feature_names=feature_names)
                            sv = ex(x_te_d)
                        else:
                            ex = shap.TreeExplainer(raw_best, feature_names=feature_names)
                            sv = ex(x_te_d)
                        st.text(f"True: {class_names[y_test[idx_choice]]}  Pred: {class_names[y_pred_all[idx_choice]]}")
                        is_linear = isinstance(sv, list)
                        if n_classes == 2:
                            base = ex.expected_value[1] if is_linear else ex.expected_value
                            vals = sv[1][idx_choice].values if is_linear else sv[idx_choice].values
                            shap.force_plot(base, vals, x_te_d[idx_choice], feature_names=feature_names, matplotlib=True, show=False)
                        else:
                            pc = y_pred_all[idx_choice]
                            base = ex.expected_value[:, pc].mean() if is_linear else ex.expected_value[pc]
                            vals = sv[pc][idx_choice].values if is_linear else sv[idx_choice, :, pc].values
                            shap.force_plot(base, vals, x_te_d[idx_choice], feature_names=feature_names, matplotlib=True, show=False)
                        st.pyplot(plt.gcf()); plt.clf()
                    except ImportError:
                        st.error("`shap` not installed.")
                    except Exception as e:
                        st.error(f"SHAP error: {e}")

    with tabs[2]:
        st.markdown("**LIME** — local explanations.")
        try:
            from lime.lime_text import LimeTextExplainer
        except ImportError:
            st.error("`lime` not installed. Run `pip install lime`."); st.stop()
        lime_explainer = LimeTextExplainer(class_names=class_names)

        def explain_with_lime(text, num_features=15):
            return lime_explainer.explain_instance(text, best_pipeline.predict_proba, num_features=num_features, top_labels=n_classes)

        st.markdown("##### LIME: Misclassified Example")
        if len(misclassified_idx) > 0:
            idx_lime = st.selectbox("Pick misclassified index:", misclassified_idx[:10], key="lime_mis_idx",
                format_func=lambda i: f"Index {i} — True: {class_names[y_test[i]]}, Pred: {class_names[y_pred_all[i]]}")
            if st.button("Explain with LIME", key="lime_mis_btn"):
                exp = explain_with_lime(test_texts[idx_lime])
                probs = best_pipeline.predict_proba([test_texts[idx_lime]])[0]
                for ci, cn in enumerate(class_names):
                    st.write(f"P({cn}): {probs[ci]:.2%}")
                fig = exp.as_pyplot_figure(); st.pyplot(fig)

        st.markdown("---")
        st.markdown("##### LIME: Custom Review")
        custom_review = st.text_input("Enter a review to explain with LIME:", "The movie was absolutely fantastic!", key="lime_custom")
        if st.button("Explain with LIME", key="lime_custom_btn") and custom_review.strip():
            cleaned_review = _clean_text(custom_review)
            pred = best_pipeline.predict([cleaned_review])[0]
            st.write(f"**Prediction:** {class_names[pred]}")
            probs = best_pipeline.predict_proba([cleaned_review])[0]
            for ci, cn in enumerate(class_names):
                st.write(f"P({cn}): {probs[ci]:.2%}")
            exp = explain_with_lime(cleaned_review)
            fig = exp.as_pyplot_figure(); st.pyplot(fig)

    with tabs[3]:
        st.markdown("**Compare all 5 models** on your own review.")
        user_review = st.text_input("Enter a review to classify:", "This product is amazing and worked perfectly!", key="all_models_input")
        if st.button("Classify with All Models", key="all_models_btn") and user_review.strip():
            cleaned_review = _clean_text(user_review)
            vec = vect.transform([cleaned_review])
            st.code(f"{'Model':<25} {'Prediction':<15} {'Confidence':<10}")
            st.code("-" * 55)
            for name, clf in trained_models.items():
                p = clf.predict(vec)[0]
                lbl = class_names[p]
                if hasattr(clf, "predict_proba"):
                    proba = clf.predict_proba(vec)[0]
                    conf = proba[int(p)]
                    probs_str = " | ".join([f"{c}: {proba[i]:.1%}" for i, c in enumerate(class_names)])
                    st.code(f"{name:<25} {lbl:<15} {conf:.2%}")
                    st.code(f"{'':<25} {probs_str}")
                else:
                    st.code(f"{name:<25} {lbl:<15} N/A")
                st.code("-" * 55)

    st.success("Pipeline complete! All 5 phases executed successfully.")

else:
    # ── Test Predictions mode: lightweight, no heavy UI ──
    st.info("Switch back to **Train Pipeline** mode to view the full training report.")
    st.toast("Ready for predictions! Type a review below.", icon="🔮")

    with st.expander("Download Trained Models", expanded=False):
        import io as _io
        import joblib as _joblib
        st.markdown("Download individual models for offline use.")
        for name, clf in trained_models.items():
            buf = _io.BytesIO()
            _joblib.dump(clf, buf)
            buf.seek(0)
            st.download_button(
                f"⬇️ {name} (.joblib)", data=buf,
                file_name=f"{name.lower().replace(' ', '_')}_model.joblib",
                mime="application/octet-stream",
                key=f"dl_{name}"
            )
        buf_v = _io.BytesIO()
        _joblib.dump(vect, buf_v)
        buf_v.seek(0)
        st.download_button(
            "⬇️ Vectorizer (.joblib)", data=buf_v,
            file_name="vectorizer.joblib", mime="application/octet-stream",
            key="dl_vect"
        )

        st.markdown("---")
        st.markdown("**Download full Pipeline** (vectorizer + classifier together) — upload this `.joblib` file later to make predictions directly without re-training.")
        buf_pipe = _io.BytesIO()
        _joblib.dump(best_pipeline, buf_pipe)
        buf_pipe.seek(0)
        st.download_button(
            "⬇️ Full Pipeline (best_pipeline.joblib)", data=buf_pipe,
            file_name="best_pipeline.joblib", mime="application/octet-stream",
            key="dl_pipeline",
            type="primary"
        )

    st.markdown("### Test Custom Review")
    st.markdown("Enter text below to classify with all 5 trained models.")

    user_text = st.text_area("Review text:", "", height=120)
    if st.button("Classify", type="primary") and user_text.strip():
        cleaned = _clean_text(user_text)
        # Standalone classifier without vectorizer — can't vectorize text
        if vect is None and not hasattr(best_pipeline, 'named_steps'):
            st.error("Cannot predict: this model file contains only a classifier, "
                     "no vectorizer. Upload a full Pipeline (.joblib) with both "
                     "vectorizer + classifier, or use a model package (.zip).")
            st.stop()
        # Use best_pipeline directly if it's a full Pipeline (includes vectorizer)
        if hasattr(best_pipeline, 'named_steps'):
            st.code(f"{'Model':<25} {'Prediction':<15} {'Confidence':<10}")
            st.code("-" * 55)
            for name, clf in trained_models.items():
                p = clf.predict(vec)[0]
                lbl = class_names[p]
                if hasattr(clf, "predict_proba"):
                    proba = clf.predict_proba(vec)[0]
                    conf = proba[int(p)]
                    probs_str = " | ".join([f"{c}: {proba[i]:.1%}" for i, c in enumerate(class_names)])
                    st.code(f"{name:<25} {lbl:<15} {conf:.2%}")
                    st.code(f"{'':<25} {probs_str}")
                else:
                    st.code(f"{name:<25} {lbl:<15} N/A")
                st.code("-" * 55)
        else:
            # Separate vect + clf (from zip or train)
            vec = vect.transform([cleaned])
            st.code(f"{'Model':<25} {'Prediction':<15} {'Confidence':<10}")
            st.code("-" * 55)
            for name, clf in trained_models.items():
                p = clf.predict(vec)[0]
                lbl = class_names[p]
                if hasattr(clf, "predict_proba"):
                    proba = clf.predict_proba(vec)[0]
                    conf = proba[int(p)]
                    probs_str = " | ".join([f"{c}: {proba[i]:.1%}" for i, c in enumerate(class_names)])
                    st.code(f"{name:<25} {lbl:<15} {conf:.2%}")
                    st.code(f"{'':<25} {probs_str}")
                else:
                    st.code(f"{name:<25} {lbl:<15} N/A")
                st.code("-" * 55)
