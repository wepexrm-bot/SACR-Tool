import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
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
for res_name in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']:
    try:
        nltk.data.find(f'tokenizers/{res_name}') if res_name == 'punkt' else \
        nltk.data.find(f'corpora/{res_name}') if res_name in ('stopwords', 'wordnet') else \
        nltk.data.find(f'taggers/{res_name}')
    except LookupError:
        nltk.download(res_name)


st.set_page_config(page_title="SACR Tool — Complete Pipeline", layout="wide")
st.title("SACR Tool — Complete Sentiment Analysis Pipeline")
st.markdown("Mirrors the full 5‑phase notebook. Upload → automatic run → test at the bottom.")

# ────────────────────────────────────────────────────────────
#  CONFIGURATION
# ────────────────────────────────────────────────────────────
st.sidebar.header("Configuration")
INCLUDE_NEUTRAL = st.sidebar.checkbox("Keep neutral class (ratings 5‑6)", value=True,
    help="When ON: ratings ≥7→pos, ≤4→neg, 5‑6→neutral. When OFF: neutral rows are dropped.")
TEST_SIZE = st.sidebar.selectbox("Test set size", [0.2, 0.3, 0.4], index=0)
RANDOM_STATE = st.sidebar.number_input("Random seed", 1, 200, 42)
VECTORIZER_TYPE = st.sidebar.radio("Vectorizer", ["TF-IDF", "CountVectorizer"], horizontal=True)
NGRAM_MIN, NGRAM_MAX = 1, 3
MIN_DF = 10
MAX_FEATURES = 10000

# ────────────────────────────────────────────────────────────
#  LOAD DATASET
# ────────────────────────────────────────────────────────────
data_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])
if data_file is None:
    st.stop()

with st.spinner("Loading dataset..."):
    df = pd.read_csv(data_file)

# ────────────────────────────────────────────────────────────
#  PHASE 1 : DATA PREPROCESSING & EDA
# ────────────────────────────────────────────────────────────
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

# Detect text & target columns
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

potential_sentiment_cols = [c for c in df.columns if 'sentiment' in c.lower() or 'label' in c.lower()]
rating_cols = [c for c in df.columns if 'rating' in c.lower() or 'score' in c.lower()]

st.info(f"**Text column:** `{text_col}`  |  **Sentiment cols:** {potential_sentiment_cols}  |  **Rating cols:** {rating_cols}")

# Drop null text
before = len(df)
df = df.dropna(subset=[text_col])
dropped = before - len(df)
if dropped:
    st.write(f"Dropped {dropped} rows with null text.")

# Label Creation (mirrors notebook cell 11)
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
    st.warning("No target column found — used keyword‑based labeling (weak fallback).")

elif label_source == 'categorical':
    df['label_raw'] = df[target_col].astype(str).str.strip().str.lower()
    st.write(f"Using categorical target `{target_col}`. Values: {sorted(df['label_raw'].unique())}")

else:
    vals = df[target_col].dropna()
    unique_vals = sorted(vals.unique())
    if set(unique_vals) <= {0, 1}:
        df['label_raw'] = df[target_col].map({0: 'negative', 1: 'positive'})
        st.write(f"Binary 0/1 column `{target_col}` → negative/positive.")
    elif vals.max() > 1:
        def map_rating(x):
            if x >= 7:
                return 'positive'
            elif x <= 4:
                return 'negative'
            else:
                return 'neutral'
        df['label_raw'] = df[target_col].apply(map_rating)
        n_neutral = (df['label_raw'] == 'neutral').sum()
        if INCLUDE_NEUTRAL:
            st.write(f"Rating scale: ≥7→pos, ≤4→neg, 5‑6→neutral. Keeping {n_neutral} neutral rows.")
        else:
            df = df[df['label_raw'] != 'neutral']
            st.write(f"Rating scale: dropped {n_neutral} neutral rows (INCLUDE_NEUTRAL=False).")

df = df.dropna(subset=['label_raw'])

# Encode labels (cell 12)
le = LabelEncoder()
df['label'] = le.fit_transform(df['label_raw'])
class_names = list(le.classes_)
n_classes = len(class_names)
st.write(f"**Classes:** {class_names}  |  **n_classes:** {n_classes}")
imb = df['label'].value_counts().max() / df['label'].value_counts().min()
if imb > 1.5:
    st.write(f"⚠️ Imbalanced (ratio {imb:.2f}). Stratified split + `class_weight='balanced'` will be used.")

# Text length distribution (cell 13)
col1, col2 = st.columns(2)
with col1:
    df['text_length'] = df[text_col].astype(str).apply(len)
    df['word_count'] = df[text_col].astype(str).apply(lambda x: len(x.split()))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df['text_length'], bins=50, edgecolor='black')
    ax.set_title('Text Length Distribution')
    ax.set_xlabel('Length (chars)')
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df['word_count'], bins=50, edgecolor='black', color='orange')
    ax.set_title('Word Count Distribution')
    ax.set_xlabel('Word Count')
    st.pyplot(fig)

# Class distribution (cell 14)
fig, ax = plt.subplots(figsize=(6, 4))
colors_bar = ['green' if c == 'positive' else 'blue' if c == 'neutral' else 'red' for c in class_names]
df['label_raw'].value_counts().plot(kind='bar', ax=ax, color=colors_bar)
ax.set_title('Class Distribution')
ax.set_xlabel('Class')
ax.set_ylabel('Count')
st.pyplot(fig)

# Word cloud per class (cell 15)
st.subheader("Word Cloud per Class")
n_show = min(n_classes, 3)
ncols_wc = min(3, n_classes)
fig, axes = plt.subplots(1, ncols_wc, figsize=(7 * ncols_wc, 6))
if ncols_wc == 1:
    axes = [axes]
cmap_map = {'positive': 'Greens', 'neutral': 'Blues', 'negative': 'Reds'}
for i, cls in enumerate(class_names[:n_show]):
    texts = ' '.join(df[df['label_raw'] == cls][text_col].astype(str).head(500))
    cmap_wc = cmap_map.get(cls.lower(), 'viridis')
    wc = WordCloud(width=500, height=400, background_color='white', colormap=cmap_wc, max_words=100).generate(texts)
    axes[i].imshow(wc, interpolation='bilinear')
    axes[i].axis('off')
    axes[i].set_title(f'{cls.title()} Reviews', color=cmap_wc.replace('s', ''))
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ────────────────────────────────────────────────────────────
#  PHASE 2 : FEATURE ENGINEERING
# ────────────────────────────────────────────────────────────
st.header("Phase 2 — Feature Engineering")

# Customized stopwords (cell 17)
stop_words = stopwords.words('english')
new_stopwords = ['would', 'shall', 'could', 'might']
stop_words.extend(new_stopwords)
stop_words.remove('not')
stop_words = set(stop_words)
st.write(f"Custom stopwords loaded ({len(stop_words)} words) — kept 'not'.")

# Contraction expansion & cleaning (cell 18)
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

def remove_special_chars(content):
    return re.sub(r'\W+', ' ', content)

def remove_urls(content):
    return re.sub(r'http\S+', '', content)

def remove_stopwords_from_text(content):
    clean = []
    for i in content.split():
        if i.strip().lower() not in stop_words and i.strip().lower().isalpha():
            clean.append(i.strip().lower())
    return ' '.join(clean)

def data_cleaning(content):
    if not isinstance(content, str):
        return ''
    content = contraction_expansion(content)
    content = remove_special_chars(content)
    content = remove_urls(content)
    content = remove_stopwords_from_text(content)
    return content

# Apply cleaning (cell 19)
with st.spinner("Cleaning text..."):
    df['processed_text'] = df[text_col].astype(str).apply(data_cleaning)
st.dataframe(df[['processed_text']].head(10), use_container_width=True)

# Drop empty processed text (cell 20)
clean_df = df.dropna(subset=['processed_text', 'label']).copy()
clean_df = clean_df[clean_df['processed_text'].str.strip() != '']
st.write(f"After cleaning: {len(clean_df)} rows (dropped {len(df) - len(clean_df)} with empty text).")

# Train/Test split BEFORE vectorization (cell 21)
train_df, test_df = train_test_split(
    clean_df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=clean_df['label']
)
st.write(f"Train: {train_df.shape}, Test: {test_df.shape}")
st.write(f"Train balance:\n{train_df['label_raw'].value_counts(normalize=True).round(3)}")
st.write(f"Test balance:\n{test_df['label_raw'].value_counts(normalize=True).round(3)}")

# LemmaTokenizer (cell 22)
class LemmaTokenizer(object):
    def __init__(self):
        self.wordnetlemma = WordNetLemmatizer()
    def __call__(self, reviews):
        return [self.wordnetlemma.lemmatize(word) for word in word_tokenize(reviews)]

# Vectorize (cell 22)
if VECTORIZER_TYPE == 'CountVectorizer':
    vect = CountVectorizer(analyzer='word', tokenizer=LemmaTokenizer(),
                           ngram_range=(NGRAM_MIN, NGRAM_MAX), min_df=MIN_DF, max_features=MAX_FEATURES)
else:
    vect = TfidfVectorizer(analyzer='word', tokenizer=LemmaTokenizer(),
                            ngram_range=(NGRAM_MIN, NGRAM_MAX), min_df=MIN_DF, max_features=MAX_FEATURES)

x_train = vect.fit_transform(train_df['processed_text'])
x_test = vect.transform(test_df['processed_text'])
y_train = train_df['label'].values
y_test = test_df['label'].values
feature_names = vect.get_feature_names_out()

st.success(f"x_train: {x_train.shape}  |  x_test: {x_test.shape}")
st.write(f"y_train distribution: {np.bincount(y_train)}")
st.write(f"y_test distribution: {np.bincount(y_test)}")

# Top Features by Score (cell 23)
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

# Chi-Squared (cell 24)
st.subheader("Chi-Squared Feature Selection")
chi_scores, _ = chi2(x_train, y_train)
chi_df = pd.DataFrame({"Feature": feature_names, "Chi2 Score": chi_scores}).sort_values("Chi2 Score", ascending=False).head(20)
st.dataframe(chi_df.reset_index(drop=True), use_container_width=True)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="Chi2 Score", y="Feature", data=chi_df, ax=ax)
ax.set_title("Top 20 Features by Chi-Squared Score")
st.pyplot(fig)

# ────────────────────────────────────────────────────────────
#  PHASE 3 : MODEL SELECTION
# ────────────────────────────────────────────────────────────
st.header("Phase 3 — Model Selection")

SEED = RANDOM_STATE
classifiers = {
    'Logistic Regression': LogisticRegression(C=10, max_iter=200, solver='lbfgs',
                                               random_state=SEED, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_split=2, criterion='gini',
                                             random_state=SEED, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2,
                                             random_state=SEED, class_weight='balanced'),
    'AdaBoost': AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=SEED),
    'Naive Bayes': MultinomialNB(alpha=1.0)
}

results = []
trained_models = {}
prog = st.progress(0, text="Training models...")

for idx, (name, clf) in enumerate(classifiers.items()):
    prog.progress((idx + 1) / len(classifiers), text=f"Training {name}...")
    start = time.time()
    clf.fit(x_train, y_train)
    training_time = time.time() - start
    y_pred = clf.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision_w = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_w = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

    results.append({
        'Model': name, 'Accuracy': accuracy, 'Precision': precision_w,
        'Recall': recall_w, 'F1_Weighted': f1_w, 'F1_Macro': f1_macro,
        'Training_Time': training_time
    })
    trained_models[name] = clf

    st.write(f"**{name}** — Acc: {accuracy:.4f} | F1(w): {f1_w:.4f} | F1(m): {f1_macro:.4f} | Time: {training_time:.2f}s")

    # Leakage sanity check
    if accuracy >= 0.999:
        st.error(f"⚠️ {name}: accuracy {accuracy:.4f} — possible data leakage!")

prog.progress(1.0, text="All models trained!")

# Results Summary (cell 28)
results_df = pd.DataFrame(results).sort_values('F1_Weighted', ascending=False)
st.subheader("Results Summary")
st.dataframe(results_df.set_index('Model'), use_container_width=True)

# Performance Comparison Charts (cell 29)
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

# Training Time Comparison (cell 30)
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x='Model', y='Training_Time', data=results_df, ax=ax, palette='viridis')
ax.set_title('Training Time Comparison')
ax.set_ylabel('Time (seconds)')
for i, v in enumerate(results_df['Training_Time']):
    ax.text(i, v + 0.01, f'{v:.2f}s', ha='center', fontsize=10)
plt.tight_layout()
st.pyplot(fig)

# ────────────────────────────────────────────────────────────
#  PHASE 4 : MODEL EVALUATION & XAI
# ────────────────────────────────────────────────────────────
st.header("Phase 4 — Model Evaluation & XAI")

# Pick best model (cell 32)
best_model_name = results_df.iloc[0]['Model']
best_clf = trained_models[best_model_name]
y_pred_best = best_clf.predict(x_test)
st.success(f"**Best model (by weighted F1):** {best_model_name}")

# Detailed classification report
st.subheader(f"Detailed Classification Report — {best_model_name}")
report = classification_report(y_test, y_pred_best, target_names=class_names,
                               output_dict=True, zero_division=0)
st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

# Confusion Matrix (cell 33)
def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5 + len(class_names), 4 + len(class_names) // 2))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap='Blues', values_format='d')
    plt.title(title)
    plt.tight_layout()
    return fig

st.pyplot(plot_confusion_matrix(y_test, y_pred_best, class_names, f'Confusion Matrix — {best_model_name}'))

# ROC-AUC (cell 34)
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
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
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
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        st.metric("Macro-average AUC", f"{auc_macro:.4f}")
else:
    st.info(f"{best_model_name} does not support predict_proba.")

# Confusion Matrices for ALL models (cell 35)
st.subheader("Confusion Matrices — All Models")
n_models = len(trained_models)
n_cols = min(3, n_models)
n_rows = -(-n_models // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
axes = np.array(axes).ravel()
for i, (name, clf) in enumerate(trained_models.items()):
    y_pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=axes[i], cmap='Blues', values_format='d')
    axes[i].set_title(name)
for j in range(n_models, len(axes)):
    axes[j].axis('off')
plt.tight_layout()
st.pyplot(fig)

# Misclassification Analysis (cell 36)
st.subheader("Misclassification Analysis")
y_pred_all = best_clf.predict(x_test)
test_texts = test_df['processed_text'].values
misclassified_idx = np.where(y_pred_all != y_test)[0]
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

# Feature Importance (cell 37)
st.subheader("Feature Importance")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()
ax_idx = 0
for name in ['Random Forest', 'Decision Tree', 'AdaBoost', 'Logistic Regression']:
    clf = trained_models.get(name)
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
    axes[ax_idx].invert_yaxis()
    ax_idx += 1
for j in range(ax_idx, len(axes)):
    axes[j].axis('off')
plt.tight_layout()
st.pyplot(fig)

# ────────────────────────────────────────────────────────────
#  PHASE 5 : EXPLAINABILITY — SHAP & LIME
# ────────────────────────────────────────────────────────────
st.header("Phase 5 — Explainability (XAI)")

# Pipeline wrapper for LIME (cell 41)
best_pipeline = Pipeline(steps=[('vect', vect), ('clf', best_clf)])
st.write(f"Pipeline ready for: **{best_model_name}**")

tabs = st.tabs(["SHAP (Global)", "SHAP (Individual)", "LIME", "All 5 Models"])

# ── SHAP GLOBAL ──
with tabs[0]:
    st.markdown("**SHAP beeswarm** — global feature contribution ranking.")
    max_samples = st.slider("SHAP sample size", 50, 500, 200, key="shap_global_samples")
    if st.button("Run SHAP (Global)", key="shap_global_btn"):
        with st.spinner("Computing SHAP..."):
            try:
                import shap
                BACKGROUND_SIZE = min(max_samples, x_train.shape[0])
                SHAP_TEST_SAMPLE = min(max_samples, x_test.shape[0])
                x_train_dense_bg = x_train[:BACKGROUND_SIZE].toarray()
                x_test_dense_sample = x_test[:SHAP_TEST_SAMPLE].toarray()

                if hasattr(best_clf, 'coef_'):
                    shap_explainer = shap.LinearExplainer(best_clf, x_train_dense_bg, feature_names=feature_names)
                    shap_values = shap_explainer(x_test_dense_sample)
                elif hasattr(best_clf, 'feature_importances_'):
                    shap_explainer = shap.TreeExplainer(best_clf, feature_names=feature_names)
                    shap_values = shap_explainer(x_test_dense_sample)
                else:
                    st.error("Model type not supported by SHAP.")
                    shap_values = None

                if shap_values is not None:
                    if n_classes == 2:
                        shap.summary_plot(shap_values, x_test_dense_sample, feature_names=feature_names, show=False)
                        st.pyplot(plt.gcf()); plt.clf()
                    else:
                        for ci, cname in enumerate(class_names):
                            st.write(f"**Class: {cname}**")
                            shap.summary_plot(shap_values[:, :, ci], x_test_dense_sample,
                                              feature_names=feature_names, show=False)
                            st.pyplot(plt.gcf()); plt.clf()
            except ImportError:
                st.error("`shap` not installed. Run `pip install shap`.")
            except Exception as e:
                st.error(f"SHAP error: {e}")

# ── SHAP INDIVIDUAL ──
with tabs[1]:
    st.markdown("**SHAP force plot** on a misclassified example.")
    if len(misclassified_idx) == 0:
        st.info("No misclassified examples.")
    else:
        SHAP_TEST_SAMPLE = min(300, x_test.shape[0])
        mis_in_sample = misclassified_idx[misclassified_idx < SHAP_TEST_SAMPLE]
        if len(mis_in_sample) == 0:
            st.info("No misclassified in SHAP sample. Increase sample size above.")
        else:
            idx_choice = st.selectbox("Misclassified index:", mis_in_sample[:10], key="shap_indiv_idx",
                format_func=lambda i: f"Index {i} — True: {class_names[y_test[i]]}, Pred: {class_names[y_pred_all[i]]}")
            if st.button("Explain with SHAP", key="shap_indiv_btn"):
                try:
                    import shap
                    BACKGROUND_SIZE = min(200, x_train.shape[0])
                    x_train_dense_bg = x_train[:BACKGROUND_SIZE].toarray()
                    x_test_dense = x_test[:SHAP_TEST_SAMPLE].toarray()

                    if hasattr(best_clf, 'coef_'):
                        explainer = shap.LinearExplainer(best_clf, x_train_dense_bg, feature_names=feature_names)
                        shap_val = explainer(x_test_dense)
                    else:
                        explainer = shap.TreeExplainer(best_clf, feature_names=feature_names)
                        shap_val = explainer(x_test_dense)

                    st.text(f"True: {class_names[y_test[idx_choice]]}  Pred: {class_names[y_pred_all[idx_choice]]}")

                    if n_classes == 2:
                        shap.force_plot(explainer.expected_value[1], shap_val[idx_choice].values,
                                        x_test_dense[idx_choice], feature_names=feature_names,
                                        matplotlib=True, show=False)
                    else:
                        pred_class_idx = y_pred_all[idx_choice]
                        shap.force_plot(explainer.expected_value[:, pred_class_idx].mean(),
                                        shap_val[idx_choice, :, pred_class_idx].values,
                                        x_test_dense[idx_choice], feature_names=feature_names,
                                        matplotlib=True, show=False)
                    st.pyplot(plt.gcf()); plt.clf()
                except ImportError:
                    st.error("`shap` not installed.")
                except Exception as e:
                    st.error(f"SHAP error: {e}")

# ── LIME ──
with tabs[2]:
    st.markdown("**LIME** — local explanations.")

    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        st.error("`lime` not installed. Run `pip install lime`.")
        st.stop()

    lime_explainer = LimeTextExplainer(class_names=class_names)

    def explain_with_lime(text, num_features=15):
        exp = lime_explainer.explain_instance(
            text, best_pipeline.predict_proba, num_features=num_features, top_labels=n_classes)
        return exp

    def show_lime_explanation(exp):
        fig = exp.as_pyplot_figure()
        st.pyplot(fig)

    # LIME for misclassified
    st.markdown("##### LIME: Misclassified Example")
    if len(misclassified_idx) > 0:
        idx_lime = st.selectbox("Pick misclassified index:", misclassified_idx[:10], key="lime_mis_idx",
            format_func=lambda i: f"Index {i} — True: {class_names[y_test[i]]}, Pred: {class_names[y_pred_all[i]]}")
        if st.button("Explain with LIME", key="lime_mis_btn"):
            exp = explain_with_lime(test_texts[idx_lime])
            probs = best_pipeline.predict_proba([test_texts[idx_lime]])[0]
            for ci, cn in enumerate(class_names):
                st.write(f"P({cn}): {probs[ci]:.2%}")
            show_lime_explanation(exp)

    st.markdown("---")
    st.markdown("##### LIME: Custom Review")
    custom_review = st.text_input("Enter a review to explain with LIME:", "The movie was absolutely fantastic!",
                                  key="lime_custom")
    if st.button("Explain with LIME", key="lime_custom_btn") and custom_review.strip():
        cleaned_review = data_cleaning(custom_review)
        pred = best_pipeline.predict([cleaned_review])[0]
        st.write(f"**Prediction:** {class_names[pred]}")
        probs = best_pipeline.predict_proba([cleaned_review])[0]
        for ci, cn in enumerate(class_names):
            st.write(f"P({cn}): {probs[ci]:.2%}")
        exp = explain_with_lime(cleaned_review)
        show_lime_explanation(exp)

# ── ALL 5 MODELS ──
with tabs[3]:
    st.markdown("**Compare all 5 models** on your own review.")
    user_review = st.text_input("Enter a review to classify:", "This product is amazing and worked perfectly!",
                                key="all_models_input")
    if st.button("Classify with All Models", key="all_models_btn") and user_review.strip():
        cleaned = data_cleaning(user_review)
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

st.success("Pipeline complete! All 5 phases executed successfully.")
