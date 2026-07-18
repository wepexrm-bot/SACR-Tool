import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re
import io
import warnings
from collections import Counter

warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import chi2
from sklearn.preprocessing import label_binarize
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import nltk
for res in ['tokenizers/punkt', 'corpora/stopwords', 'corpora/wordnet', 'taggers/averaged_perceptron_tagger']:
    try:
        nltk.data.find(res)
    except LookupError:
        nltk.download(res.split('/')[1])

stop_words = set(stopwords.words('english'))
stop_words.discard('not')
stop_words.update({'would', 'shall', 'could', 'might',
                   'organization', 'company', 'work', 'worked',
                   'employee', 'employer', 'working', 'firm'})
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r"won\'t", "would not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"don\'t", "do not", text)
    text = re.sub(r"shouldn\'t", "should not", text)
    text = re.sub(r"needn\'t", "need not", text)
    text = re.sub(r"hasn\'t", "has not", text)
    text = re.sub(r"haven\'t", "have not", text)
    text = re.sub(r"weren\'t", "were not", text)
    text = re.sub(r"mightn\'t", "might not", text)
    text = re.sub(r"didn\'t", "did not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    filtered = [w for w in tokens if w not in stop_words]
    lemmatized = [lemmatizer.lemmatize(w) for w in filtered]
    return ' '.join(lemmatized)


st.set_page_config(page_title="Auto Sentiment Pipeline", layout="wide")
st.title("Auto Sentiment Pipeline — One-Click Training")
st.markdown("Upload a CSV, and the full pipeline runs automatically: cleaning, EDA, vectorization, 5 models, evaluation, and explainability. Scroll to the bottom to test predictions.")

# ── File Upload ──
data_file = st.file_uploader("Upload your dataset (CSV)", type=['csv'])
if data_file is None:
    st.stop()

with st.spinner("Loading dataset..."):
    df = pd.read_csv(data_file)
st.success(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")

# ── Phase 0: Auto-detect text & target columns ──
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

rating_col = None
for col in df.columns:
    low = col.lower()
    if 'rating' in low or 'score' in low:
        rating_col = col
        break

sentiment_col = None
for col in df.columns:
    low = col.lower()
    if 'sentiment' in low or 'label' in low:
        sentiment_col = col
        break

st.info(f"Detected → Text: `{text_col}` | Rating: `{rating_col}` | Sentiment: `{sentiment_col}`")

if text_col is None:
    st.error("No text column detected.")
    st.stop()

# ── Phase 1: Preprocessing ──
st.subheader("Phase 1: Preprocessing & Cleaning")
pbar = st.progress(0, text="Cleaning text...")
df = df.dropna(subset=[text_col]).copy()
df['clean_text'] = df[text_col].astype(str).apply(clean_text)
df = df[df['clean_text'].str.strip() != ''].copy()
pbar.progress(100)
st.success(f"Cleaned: {len(df)} rows remaining")

# ── Phase 1b: Label Creation ──
st.subheader("Label Creation")
include_neutral = True
if rating_col is not None and df[rating_col].max() > 1:
    def map_rating(x):
        if x >= 7:
            return 'positive'
        elif x <= 4:
            return 'negative'
        else:
            return 'neutral' if include_neutral else None
    df['label_raw'] = pd.to_numeric(df[rating_col], errors='coerce').map(map_rating)
    source = f"rating column `{rating_col}`"
elif sentiment_col is not None:
    df['label_raw'] = df[sentiment_col].astype(str).str.lower()
    source = f"sentiment column `{sentiment_col}`"
else:
    df['label_raw'] = 'positive'
    source = "default (positive)"
df = df.dropna(subset=['label_raw']).copy()

le = LabelEncoder()
df['label'] = le.fit_transform(df['label_raw'])
class_names = list(le.classes_)
n_classes = len(class_names)
st.write(f"**Source:** {source}  |  **Classes:** {class_names}")
st.write(df['label_raw'].value_counts().to_frame('Count'))

# ── Phase 2: EDA / Visualisation ──
st.subheader("Phase 2: EDA & Visualisation")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6, 4))
    df['text_length'] = df['clean_text'].str.len()
    ax.hist(df['text_length'], bins=50, edgecolor='black')
    ax.set_title('Text Length Distribution')
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    df['label_raw'].value_counts().plot(kind='bar', ax=ax, color=['green', 'blue', 'red'][:n_classes])
    ax.set_title('Class Distribution')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    st.pyplot(fig)

# Word clouds per class
colors_map = {'positive': 'Greens', 'neutral': 'Blues', 'negative': 'Reds'}
ncols = min(3, n_classes)
fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
if ncols == 1:
    axes = [axes]
for i, cls in enumerate(class_names):
    texts = ' '.join(df[df['label_raw'] == cls]['clean_text'].head(500))
    cmap = colors_map.get(cls.lower(), 'viridis')
    wc = WordCloud(width=500, height=400, background_color='white', colormap=cmap, max_words=100).generate(texts)
    axes[i].imshow(wc, interpolation='bilinear')
    axes[i].axis('off')
    axes[i].set_title(f'{cls.title()} Reviews', color=cmap.replace('s', ''))
plt.tight_layout()
st.pyplot(fig)

# ── Phase 2b: Feature Engineering ──
st.subheader("Phase 3: Feature Engineering")

split_opt = st.radio("Train/Test split size:", ["80/20", "70/30", "90/10"], horizontal=True, index=0)
test_size = {'80/20': 0.2, '70/30': 0.3, '90/10': 0.1}[split_opt]

vect_type = st.radio("Vectorizer:", ["TF-IDF", "CountVectorizer"], horizontal=True, index=0)

train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['label'])
st.write(f"Train: {len(train_df)} | Test: {len(test_df)}")

vect_kw = dict(ngram_range=(1, 3), min_df=5, max_features=10000)
vect = TfidfVectorizer(**vect_kw) if vect_type == "TF-IDF" else CountVectorizer(**vect_kw)

x_train = vect.fit_transform(train_df['clean_text'])
x_test = vect.transform(test_df['clean_text'])
y_train = train_df['label'].values
y_test = test_df['label'].values

st.success(f"X_train: {x_train.shape} | X_test: {x_test.shape}")

# Top features
feature_names = vect.get_feature_names_out()
scores = np.asarray(x_train.sum(axis=0)).flatten() if vect_type == "CountVectorizer" else np.asarray(x_train.mean(axis=0)).flatten()
top_df = pd.DataFrame({"Feature": feature_names, "Score": scores}).sort_values("Score", ascending=False).head(20)
st.dataframe(top_df.reset_index(drop=True), use_container_width=True)

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=top_df, x="Score", y="Feature", ax=ax, palette="viridis")
ax.set_title(f"Top 20 Features ({vect_type})")
st.pyplot(fig)

# ── Phase 3: Model Training ──
st.subheader("Phase 4: Model Training & Evaluation")

SEED = 42
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=200, solver='liblinear', random_state=SEED, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=SEED, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=SEED, class_weight='balanced'),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=SEED),
    'Naive Bayes': MultinomialNB(alpha=1.0)
}

results = []
trained_models = {}
prog = st.progress(0, text="Training models...")

for idx, (name, clf) in enumerate(classifiers.items()):
    prog.progress((idx + 1) / len(classifiers), text=f"Training {name}...")
    start = time.time()
    clf.fit(x_train, y_train)
    elapsed = time.time() - start
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_m = f1_score(y_test, y_pred, average='macro', zero_division=0)
    results.append({'Model': name, 'Accuracy': f"{acc:.4f}", 'F1 (weighted)': f"{f1_w:.4f}",
                    'F1 (macro)': f"{f1_m:.4f}", 'Time (s)': f"{elapsed:.2f}"})
    trained_models[name] = {'model': clf, 'accuracy': acc, 'f1_weighted': f1_w}
    if acc >= 0.999:
        st.error(f"⚠️ {name}: Accuracy {acc:.4f} — possible data leakage!")

prog.progress(1.0, text="All models trained!")

st.dataframe(pd.DataFrame(results).set_index('Model'), use_container_width=True)

# ── Phase 4: Evaluation ──
st.subheader("Model Evaluation")

best_name = max(trained_models, key=lambda n: trained_models[n]['f1_weighted'])
best_clf = trained_models[best_name]['model']
st.success(f"**Best model:** {best_name} (F1-weighted: {trained_models[best_name]['f1_weighted']:.4f})")

# Confusion matrices all models
st.markdown("#### Confusion Matrices — All Models")
names = list(trained_models.keys())
n_models = len(names)
n_cols = min(3, n_models)
n_rows = -(-n_models // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
axes_flat = axes.ravel() if n_models > 1 else [axes]
for i, name in enumerate(names):
    y_pred = trained_models[name]['model'].predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=axes_flat[i], cmap='Blues', values_format='d')
    axes_flat[i].set_title(name)
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].axis('off')
plt.tight_layout()
st.pyplot(fig)

# ROC-AUC
st.markdown("#### ROC-AUC Curves")
fig2, ax2 = plt.subplots(figsize=(8, 6))
for name in names:
    clf = trained_models[name]['model']
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(x_test)
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            auc_val = roc_auc_score(y_test, y_prob[:, 1])
            ax2.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
        else:
            y_bin = label_binarize(y_test, classes=range(n_classes))
            auc_macro = roc_auc_score(y_bin, y_prob, average='macro', multi_class='ovr')
            for i, cn in enumerate(class_names):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            ax2.plot(fpr, tpr, label=f"{name} (macro AUC={auc_macro:.3f})", alpha=0.7)
ax2.plot([0, 1], [0, 1], "k--", label="Random")
ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
ax2.set_title("ROC Curves — All Models")
ax2.legend(loc='lower right')
plt.tight_layout()
st.pyplot(fig2)

# Feature Importance
st.markdown("#### Feature Importance (Top 15 per model)")
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
axes3 = axes3.ravel()
ax_idx = 0
for name in ['Random Forest', 'Decision Tree', 'AdaBoost', 'Logistic Regression']:
    if name not in trained_models:
        continue
    clf = trained_models[name]['model']
    if hasattr(clf, 'coef_'):
        coef = clf.coef_
        imp = np.abs(coef).mean(axis=0) if coef.shape[0] > 1 else np.abs(coef[0])
    elif hasattr(clf, 'feature_importances_'):
        imp = clf.feature_importances_
    else:
        continue
    top_n = 15
    top_idx = np.argsort(imp)[-top_n:]
    axes3[ax_idx].barh(range(top_n), imp[top_idx], color='steelblue')
    axes3[ax_idx].set_yticks(range(top_n))
    axes3[ax_idx].set_yticklabels([feature_names[i] for i in top_idx], fontsize=8)
    axes3[ax_idx].set_title(f'Top 15 — {name}')
    axes3[ax_idx].invert_yaxis()
    ax_idx += 1
for j in range(ax_idx, len(axes3)):
    axes3[j].axis('off')
plt.tight_layout()
st.pyplot(fig3)

# Misclassification
st.markdown("#### Misclassification Analysis")
best_y_pred = best_clf.predict(x_test)
mis_idx = np.where(best_y_pred != y_test)[0]
st.write(f"**{best_name}** — Misclassified: {len(mis_idx)} / {len(y_test)} ({len(mis_idx)/len(y_test):.1%})")
x_test_texts = test_df['clean_text'].values
for true_cls in range(n_classes):
    for pred_cls in range(n_classes):
        if true_cls == pred_cls:
            continue
        idxs = np.where((y_test == true_cls) & (best_y_pred == pred_cls))[0]
        if len(idxs) == 0:
            continue
        label = f"True: {class_names[true_cls]} → Pred: {class_names[pred_cls]} ({len(idxs)} cases)"
        if st.checkbox(label, key=f"auto_mis_{true_cls}_{pred_cls}"):
            for idx in idxs[:10]:
                st.text(f"[{idx}] {x_test_texts[idx][:200]}")

# ── Phase 5: Custom Prediction (all models) ──
st.subheader("Test with Custom Review")
user_review = st.text_area("Enter a review to classify with all models:", "", height=100)
if st.button("Classify", type="primary") and user_review.strip():
    cleaned = clean_text(user_review)
    vec = vect.transform([cleaned])
    st.code(f"{'Model':<25} {'Prediction':<15} {'Confidence':<10}")
    st.code("-" * 55)
    for name, entry in trained_models.items():
        clf = entry['model']
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
