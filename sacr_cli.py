#!/usr/bin/env python3
"""
SACR Tool — Complete Sentiment Analysis Pipeline (CLI)
========================================================
Cross-platform command-line interface for the full pipeline:
  Phase 1: Data Preprocessing & EDA
  Phase 2: Feature Engineering
  Phase 3: Model Selection
  Phase 4: Model Evaluation & XAI
  Phase 5: Explainability (SHAP & LIME)

Usage:
  sacr_cli train <csv_path> [options]
  sacr_cli predict --text "..." --model-dir <dir>
  sacr_cli predict --file texts.csv --model-dir <dir>
  sacr_cli explain --model-dir <dir> [--shap] [--lime]
  sacr_cli evaluate --model-dir <dir>
  sacr_cli reset --model-dir <dir>
"""

import argparse, sys, os, re, time, json, warnings, importlib, importlib.metadata, shutil
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# ── NLTK setup ────────────────────────────────────────────
import nltk
# Support both punkt_tab (NLTK 3.9+) and punkt (older NLTK)
for res in ['punkt_tab', 'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger']:
    if res in ('punkt_tab', 'punkt'):
        key = f'tokenizers/{res}'
    elif res in ('stopwords', 'wordnet'):
        key = f'corpora/{res}'
    else:
        key = f'taggers/{res}'
    try:
        nltk.data.find(key)
    except LookupError:
        nltk.download(res, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize

# ── SKLearn ────────────────────────────────────────────────
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, roc_curve, roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB

# ── Optional ───────────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

try:
    import contractions
    HAS_CONTRACTIONS = True
except ImportError:
    HAS_CONTRACTIONS = False

# ═══════════════════════════════════════════════════════════
#  GLOBALS / DEFAULTS
# ═══════════════════════════════════════════════════════════

DEFAULTS = {
    'test_size': 0.2,
    'seed': 42,
    'max_features': 10000,
    'min_df': 10,
    'ngram_min': 1,
    'ngram_max': 3,
    'vectorizer': 'tfidf',
    'include_neutral': True,
    'nb_alpha': 0.5,
    'nrows': 500000,
    'use_smote': True,
    'use_calibration': True,
    'use_voting': True,
    'use_grid_search': True,
}

# Custom stopwords: keep 'not' (critical for sentiment), remove modals
_STOP_WORDS = set(stopwords.words('english'))
_STOP_WORDS.discard('not')
_STOP_WORDS.update(['would', 'shall', 'could', 'might'])
_LEMMATIZER = WordNetLemmatizer()


# ═══════════════════════════════════════════════════════════
#  TEXT CLEANING
# ═══════════════════════════════════════════════════════════

def contraction_expansion(content):
    if HAS_CONTRACTIONS:
        return contractions.fix(content)
    content = re.sub(r"won\'t", "would not", content)
    content = re.sub(r"can\'t", "can not", content)
    content = re.sub(r"don\'t", "do not", content)
    content = re.sub(r"n\'t", " not", content)
    return content


def data_cleaning(content):
    if not isinstance(content, str):
        return ''
    content = contraction_expansion(content)
    content = re.sub(r'http\S+', '', content)
    content = re.sub(r'\W+', ' ', content)

    tokens = []
    negate = False
    for w in content.split():
        wl = w.strip().lower()
        if wl == 'not':
            negate = True
            continue
        if wl.isalpha() and wl not in _STOP_WORDS:
            if negate:
                tokens.append(f'not_{wl}')
                negate = False
            else:
                tokens.append(wl)
    return ' '.join(tokens)


# ═══════════════════════════════════════════════════════════
#  POS-AWARE LEMMATOKENIZER
# ═══════════════════════════════════════════════════════════

class LemmaTokenizer:
    def __init__(self):
        self.wordnetlemma = WordNetLemmatizer()

    def __call__(self, reviews):
        tokens = word_tokenize(reviews)
        try:
            ptags = pos_tag(tokens)
        except LookupError:
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            ptags = pos_tag(tokens)
        lemmas = []
        for word, tag in ptags:
            pos = 'n'
            if tag.startswith('V'): pos = 'v'
            elif tag.startswith('J'): pos = 'a'
            elif tag.startswith('R'): pos = 'r'
            lemmas.append(self.wordnetlemma.lemmatize(word, pos))
        return lemmas


# ═══════════════════════════════════════════════════════════
#  FILE LOADING
# ═══════════════════════════════════════════════════════════

def load_data(path, nrows=None):
    ext = Path(path).suffix.lower()
    if ext == '.csv':
        for enc in ['utf-8', 'latin-1', 'cp1252']:
            try:
                return pd.read_csv(path, encoding=enc, nrows=nrows)
            except (UnicodeDecodeError, UnicodeError):
                continue
        return pd.read_csv(path, encoding='utf-8', errors='replace', nrows=nrows)
    elif ext in ('.xlsx', '.xls'):
        return pd.read_excel(path, nrows=nrows)
    elif ext == '.json':
        return pd.read_json(path)
    else:
        for enc in ['utf-8', 'latin-1', 'cp1252']:
            try:
                return pd.read_csv(path, sep='\t', encoding=enc, nrows=nrows)
            except (UnicodeDecodeError, UnicodeError):
                continue
        return pd.read_csv(path, sep='\t', encoding='utf-8', errors='replace', nrows=nrows)


# ═══════════════════════════════════════════════════════════
#  COLUMN DETECTION
# ═══════════════════════════════════════════════════════════

def detect_columns(df, text_col=None, label_col=None):
    if text_col is None:
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

    potential_sentiment_cols = [
        c for c in df.columns
        if ('sentiment' in c.lower() or 'label' in c.lower())
        and df[c].dtype == 'object'
    ]
    NUMERIC_LABEL_KEYWORDS = ['rating', 'score', 'star', 'target', 'polarity', 'class', 'sentiment', 'label']
    rating_cols = [
        c for c in df.columns
        if any(k in c.lower() for k in NUMERIC_LABEL_KEYWORDS)
        and c != text_col
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not potential_sentiment_cols and not rating_cols:
        for col in df.columns:
            if col == text_col:
                continue
            if pd.api.types.is_numeric_dtype(df[col]) and 2 <= df[col].nunique() <= 10:
                rating_cols = [col]
                break

    if label_col is not None:
        if label_col in df.columns:
            if pd.api.types.is_numeric_dtype(df[label_col]):
                rating_cols = [label_col]
            else:
                potential_sentiment_cols = [label_col]

    return text_col, potential_sentiment_cols, rating_cols


# ═══════════════════════════════════════════════════════════
#  LABEL CREATION
# ═══════════════════════════════════════════════════════════

def create_labels(df, text_col, potential_sentiment_cols, rating_cols, include_neutral=True):
    target_col = None
    label_source = None
    if potential_sentiment_cols:
        target_col = potential_sentiment_cols[0]
        label_source = 'categorical'
    elif rating_cols:
        target_col = rating_cols[0]
        label_source = 'numeric'

    if target_col is None:
        print("  [WARN] No label column found. Using keyword-based labeling (weak fallback).")
        df['label_raw'] = np.where(
            df[text_col].astype(str).str.contains(
                r'\b(good|excellent|positive|amazing|great|wonderful)\b', case=False, na=False),
            'positive', 'negative'
        )
    elif label_source == 'categorical':
        df['label_raw'] = df[target_col].astype(str).str.strip().str.lower()
        print(f"  Labels: categorical column '{target_col}' -> {sorted(df['label_raw'].unique())}")
    else:
        vals = df[target_col].dropna()
        unique_vals = sorted(vals.unique())
        n_unique = len(unique_vals)
        if n_unique == 2:
            lo, hi = unique_vals
            df['label_raw'] = df[target_col].map({lo: 'negative', hi: 'positive'})
            print(f"  Labels: 2-value column '{target_col}' ({lo}/{hi}) -> negative/positive")
        elif n_unique == 3:
            lo, mid, hi = unique_vals
            df['label_raw'] = df[target_col].map({lo: 'negative', mid: 'neutral', hi: 'positive'})
            print(f"  Labels: 3-value column '{target_col}' ({lo}/{mid}/{hi}) -> negative/neutral/positive")
            if not include_neutral:
                df = df[df['label_raw'] != 'neutral']
                print(f"  (neutral rows dropped, INCLUDE_NEUTRAL=False)")
        else:
            vmin, vmax = unique_vals[0], unique_vals[-1]
            rng = vmax - vmin
            pos_cut = vmin + (2/3) * rng
            neg_cut = vmin + (1/3) * rng
            def map_rating(x):
                if x >= pos_cut: return 'positive'
                elif x <= neg_cut: return 'negative'
                else: return 'neutral'
            df['label_raw'] = df[target_col].apply(map_rating)
            print(f"  Labels: scale '{target_col}' ({vmin}-{vmax}, {n_unique} levels) "
                  f"-> pos>={pos_cut:.2f}, neg<={neg_cut:.2f}, else neutral")
            if not include_neutral:
                df = df[df['label_raw'] != 'neutral']

    df = df.dropna(subset=['label_raw']).copy()
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label_raw'])
    class_names = list(le.classes_)
    n_classes = len(class_names)
    print(f"  Classes: {class_names}  (n_classes={n_classes})")
    print(f"  Distribution: {dict(df['label_raw'].value_counts())}")
    return df, le, class_names, n_classes


# ═══════════════════════════════════════════════════════════
#  TRAIN PIPELINE (core function)
# ═══════════════════════════════════════════════════════════

def run_pipeline(df, text_col, label_col=None, config=None):
    cfg = {**DEFAULTS, **(config or {})}

    # Phase 1: Column detection
    print("\n=== Phase 1: Data Preprocessing & EDA ===")
    print(f"  Dataset: {df.shape[0]} rows x {df.shape[1]} cols")
    text_col, potential_sentiment_cols, rating_cols = detect_columns(df, text_col, label_col)
    if text_col is None:
        print("  [ERROR] No text column found.")
        return None
    print(f"  Text column: {text_col}")
    print(f"  Sentiment columns: {potential_sentiment_cols}")
    print(f"  Rating columns: {rating_cols}")

    df = df.dropna(subset=[text_col]).copy()

    # Label creation
    df, le, class_names, n_classes = create_labels(
        df, text_col, potential_sentiment_cols, rating_cols,
        include_neutral=cfg['include_neutral']
    )

    # Phase 2: Feature Engineering
    print("\n=== Phase 2: Feature Engineering ===")
    print("  Cleaning text...")
    df['processed_text'] = df[text_col].astype(str).apply(data_cleaning)
    clean_df = df.dropna(subset=['processed_text', 'label']).copy()
    clean_df = clean_df[clean_df['processed_text'].str.strip() != '']

    train_df, test_df = train_test_split(
        clean_df, test_size=cfg['test_size'],
        random_state=cfg['seed'], stratify=clean_df['label']
    )
    print(f"  Train: {train_df.shape[0]} rows, Test: {test_df.shape[0]} rows")

    vectorizer_cls = TfidfVectorizer if cfg['vectorizer'].lower() == 'tfidf' else CountVectorizer
    vect = vectorizer_cls(
        analyzer='word', tokenizer=LemmaTokenizer(),
        ngram_range=(cfg['ngram_min'], cfg['ngram_max']),
        min_df=cfg['min_df'], max_features=cfg['max_features']
    )
    x_train = vect.fit_transform(train_df['processed_text'])
    x_test = vect.transform(test_df['processed_text'])
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    feature_names = vect.get_feature_names_out()
    print(f"  x_train: {x_train.shape}, x_test: {x_test.shape}")

    # SMOTE
    if cfg['use_smote'] and HAS_SMOTE:
        before = np.bincount(y_train)
        smote = SMOTE(random_state=cfg['seed'], k_neighbors=5)
        x_train, y_train = smote.fit_resample(x_train, y_train)
        after = np.bincount(y_train)
        print(f"  SMOTE: before {before.tolist()} -> after {after.tolist()}")

    # Phase 3: Model Selection
    print("\n=== Phase 3: Model Selection ===")

    SEED = cfg['seed']

    # GridSearch for LR
    if cfg['use_grid_search']:
        print("  GridSearch for LogisticRegression C...")
        param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
        gs = GridSearchCV(
            LogisticRegression(max_iter=200, solver='lbfgs', random_state=SEED, class_weight='balanced'),
            param_grid, cv=3, scoring='f1_weighted', n_jobs=-1
        )
        gs.fit(x_train, y_train)
        best_c = gs.best_params_['C']
        print(f"    Best C = {best_c}")
    else:
        best_c = 10

    classifiers = {
        'Logistic Regression': LogisticRegression(C=best_c, max_iter=200, solver='lbfgs', random_state=SEED, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_split=2, criterion='gini', random_state=SEED, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, random_state=SEED, class_weight='balanced'),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=SEED),
        'Naive Bayes': MultinomialNB(alpha=cfg['nb_alpha']),
    }

    results = []
    trained_models = {}
    raw_models = {}

    for name, clf in classifiers.items():
        print(f"\n  --- {name} ---")
        start = time.time()
        clf.fit(x_train, y_train)
        train_time = time.time() - start
        raw_clf = clf

        if cfg['use_calibration']:
            cal_clf = CalibratedClassifierCV(clf, cv=3)
            cal_clf.fit(x_train, y_train)
            clf = cal_clf

        y_pred = clf.predict(x_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        f1m = f1_score(y_test, y_pred, average='macro', zero_division=0)

        if acc >= 0.999:
            print(f"  [WARN] Accuracy {acc:.4f} is suspiciously perfect (possible data leakage)")

        print(f"    Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1w={f1w:.4f}  F1m={f1m:.4f}  Time={train_time:.2f}s")

        if cfg.get('use_cv_scores', True):
            try:
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
                cv_acc = cross_val_score(clf, x_train, y_train, cv=cv, scoring='accuracy')
                cv_f1 = cross_val_score(clf, x_train, y_train, cv=cv, scoring='f1_weighted')
                print(f"    CV Acc={cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")
                print(f"    CV F1w={cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")
            except Exception:
                pass

        results.append({'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec,
                        'F1_Weighted': f1w, 'F1_Macro': f1m, 'Training_Time': train_time})
        trained_models[name] = clf
        raw_models[name] = raw_clf

    # Voting Ensemble
    if cfg['use_voting'] and len(trained_models) > 1:
        print("\n  --- Voting Ensemble ---")
        voting_clf = VotingClassifier(
            estimators=[(n.replace(' ', '_'), m) for n, m in trained_models.items()],
            voting='soft' if all(hasattr(m, 'predict_proba') for m in trained_models.values()) else 'hard'
        )
        voting_clf.fit(x_train, y_train)
        y_pred_v = voting_clf.predict(x_test)
        acc_v = accuracy_score(y_test, y_pred_v)
        f1w_v = f1_score(y_test, y_pred_v, average='weighted', zero_division=0)
        trained_models['Voting Ensemble'] = voting_clf
        results.append({'Model': 'Voting Ensemble', 'Accuracy': acc_v, 'Precision': 0, 'Recall': 0,
                        'F1_Weighted': f1w_v, 'F1_Macro': 0, 'Training_Time': 0})
        print(f"    Acc={acc_v:.4f}  F1w={f1w_v:.4f}")

    # Best model selection (exclude Voting Ensemble)
    results_df = pd.DataFrame(results).sort_values('F1_Weighted', ascending=False)
    base_results = results_df[results_df['Model'] != 'Voting Ensemble']
    if base_results.empty:
        base_results = results_df
    best_row = base_results.iloc[0]
    best_model_name = best_row['Model']
    best_clf = trained_models[best_model_name]
    best_pipeline = Pipeline(steps=[('vect', vect), ('clf', best_clf)])

    print(f"\n  Best model: {best_model_name} (F1w = {best_row['F1_Weighted']:.4f})")

    # Phase 4: Evaluation
    print("\n=== Phase 4: Model Evaluation ===")
    y_pred_best = best_clf.predict(x_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_best, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_test, y_pred_best)
    print("\nConfusion Matrix:")
    print(cm)

    if hasattr(best_clf, 'predict_proba'):
        y_prob = best_clf.predict_proba(x_test)
        if n_classes == 2:
            auc_val = roc_auc_score(y_test, y_prob[:, 1])
            print(f"AUC: {auc_val:.4f}")
        else:
            y_bin = label_binarize(y_test, classes=range(n_classes))
            auc_macro = roc_auc_score(y_bin, y_prob, average='macro', multi_class='ovr')
            print(f"Macro AUC: {auc_macro:.4f}")

    # Misclassification analysis
    y_pred_all = best_clf.predict(x_test)
    mis_idx = np.where(y_pred_all != y_test)[0]
    print(f"\nMisclassified: {len(mis_idx)} / {len(y_test)} ({len(mis_idx)/len(y_test):.1%})")
    test_texts = test_df['processed_text'].values
    for true_cls in range(n_classes):
        for pred_cls in range(n_classes):
            if true_cls == pred_cls: continue
            idxs = np.where((y_test == true_cls) & (y_pred_all == pred_cls))[0]
            if len(idxs) == 0: continue
            print(f"\n  True={class_names[true_cls]} -> Pred={class_names[pred_cls]} ({len(idxs)} cases)")
            for idx in idxs[:3]:
                print(f"    [{idx}] {test_texts[idx][:150]}")

    return {
        'config': cfg,
        'class_names': class_names,
        'n_classes': n_classes,
        'le': le,
        'vect': vect,
        'feature_names': feature_names,
        'x_train': x_train, 'x_test': x_test,
        'y_train': y_train, 'y_test': y_test,
        'train_df': train_df, 'test_df': test_df,
        'test_texts': test_texts,
        'results': results,
        'results_df': results_df,
        'trained_models': trained_models,
        'raw_models': raw_models,
        'best_model_name': best_model_name,
        'best_clf': best_clf,
        'best_pipeline': best_pipeline,
        'y_pred_best': y_pred_best,
        'y_pred_all': y_pred_all,
        'misclassified_idx': mis_idx,
    }


# ═══════════════════════════════════════════════════════════
#  SAVE / LOAD ARTIFACTS
# ═══════════════════════════════════════════════════════════

def save_artifacts(pipe, out_dir):
    import joblib
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save vectorizer
    joblib.dump(pipe['vect'], out / 'vectorizer.joblib')
    # Save best pipeline (vect + clf)
    joblib.dump(pipe['best_pipeline'], out / 'best_pipeline.joblib')
    # Save label encoder
    joblib.dump(pipe['le'], out / 'label_encoder.joblib')
    # Save trained models
    joblib.dump(pipe['trained_models'], out / 'trained_models.joblib')
    # Save raw models
    joblib.dump(pipe['raw_models'], out / 'raw_models.joblib')
    # Save config + metadata
    meta = {
        'class_names': pipe['class_names'],
        'n_classes': pipe['n_classes'],
        'best_model_name': pipe['best_model_name'],
        'config': pipe['config'],
    }
    with open(out / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    # Save results CSV
    pipe['results_df'].to_csv(out / 'results.csv', index=False)
    print(f"\n  Artifacts saved to: {out.resolve()}")


def load_artifacts(model_dir):
    import joblib
    md = Path(model_dir)
    if not md.exists():
        print(f"[ERROR] Model directory not found: {md.resolve()}")
        return None
    with open(md / 'meta.json') as f:
        meta = json.load(f)
    return {
        'vect': joblib.load(md / 'vectorizer.joblib'),
        'best_pipeline': joblib.load(md / 'best_pipeline.joblib'),
        'le': joblib.load(md / 'label_encoder.joblib'),
        'trained_models': joblib.load(md / 'trained_models.joblib'),
        'raw_models': joblib.load(md / 'raw_models.joblib'),
        'class_names': meta['class_names'],
        'n_classes': meta['n_classes'],
        'best_model_name': meta['best_model_name'],
        'config': meta.get('config', {}),
    }


# ═══════════════════════════════════════════════════════════
#  INFERENCE
# ═══════════════════════════════════════════════════════════

def do_predict(artifacts, texts):
    cleaned = [data_cleaning(t) for t in texts]
    pipe = artifacts['best_pipeline']
    probs = pipe.predict_proba(cleaned)
    preds = pipe.predict(cleaned)
    class_names = artifacts['class_names']
    results = []
    for txt, p, pr in zip(texts, preds, probs):
        label = class_names[int(p)]
        conf = {class_names[i]: float(pr[i]) for i in range(len(class_names))}
        results.append({'text': txt, 'prediction': label, 'confidence': conf})
    return results


# ═══════════════════════════════════════════════════════════
#  SHAP / LIME EXPLAINABILITY
# ═══════════════════════════════════════════════════════════

def do_shap_global(artifacts, x_test, feature_names, max_samples=200):
    import shap
    raw_models = artifacts['raw_models']
    best_name = artifacts['best_model_name']
    raw_best = raw_models.get(best_name, artifacts['best_pipeline'].named_steps['clf'])
    class_names = artifacts['class_names']
    n_classes = artifacts['n_classes']

    rng = np.random.RandomState(42)
    idxs = rng.choice(x_test.shape[0], min(max_samples, x_test.shape[0]), replace=False)
    X_sample = x_test[idxs]

    if hasattr(raw_best, 'coef_'):
        explainer = shap.LinearExplainer(raw_best, X_sample)
    elif hasattr(raw_best, 'feature_importances_'):
        explainer = shap.TreeExplainer(raw_best, feature_names=feature_names)
    else:
        bg = x_test[rng.choice(x_test.shape[0], min(100, x_test.shape[0]), replace=False)]
        explainer = shap.KernelExplainer(artifacts['best_pipeline'].predict_proba, bg)

    shap_values = explainer.shap_values(X_sample)
    print(f"  SHAP computed: {len(idxs)} samples")
    return shap_values, explainer


def do_lime(artifacts, text, num_features=15):
    from lime.lime_text import LimeTextExplainer
    class_names = artifacts['class_names']
    explainer = LimeTextExplainer(class_names=class_names)
    cleaned = data_cleaning(text)
    exp = explainer.explain_instance(cleaned, artifacts['best_pipeline'].predict_proba,
                                     num_features=num_features, top_labels=len(class_names))
    return exp, cleaned


# ═══════════════════════════════════════════════════════════
#  CLI COMMANDS
# ═══════════════════════════════════════════════════════════

def cmd_train(args):
    print(f"SACR Tool — Training Pipeline\n{'='*50}")
    print(f"Data: {args.data}")

    df = load_data(args.data, nrows=args.nrows)
    print(f"Loaded {len(df)} rows")

    config = {
        'test_size': args.test_size,
        'seed': args.seed,
        'max_features': args.max_features,
        'min_df': args.min_df,
        'ngram_min': args.ngram_min,
        'ngram_max': args.ngram_max,
        'vectorizer': args.vectorizer,
        'include_neutral': args.include_neutral,
        'nb_alpha': args.nb_alpha,
        'nrows': args.nrows,
        'use_smote': not args.no_smote,
        'use_calibration': not args.no_calibration,
        'use_voting': not args.no_voting,
        'use_grid_search': not args.no_grid_search,
        'use_cv_scores': True,
    }

    pipe = run_pipeline(df, text_col=args.text_col, label_col=args.label_col, config=config)
    if pipe is None:
        sys.exit(1)

    save_artifacts(pipe, args.out)
    print(f"\n{'='*50}\nTraining complete!")


def cmd_predict(args):
    artifacts = load_artifacts(args.model_dir)
    if artifacts is None:
        sys.exit(1)

    texts = []
    if args.text:
        texts.append(args.text)
    if args.file:
        fpath = Path(args.file)
        if fpath.suffix.lower() == '.csv':
            df = pd.read_csv(fpath)
            col = df.columns[0]
            texts.extend(df[col].astype(str).tolist())
        else:
            with open(fpath) as f:
                texts.extend([line.strip() for line in f if line.strip()])

    if not texts:
        print("[ERROR] No text provided. Use --text or --file.")
        sys.exit(1)

    results = do_predict(artifacts, texts)
    for r in results:
        print(f"\nText: {r['text'][:200]}")
        print(f"Prediction: {r['prediction']}")
        conf_str = " | ".join([f"{k}: {v:.1%}" for k, v in r['confidence'].items()])
        print(f"Confidence: {conf_str}")


def cmd_explain(args):
    artifacts = load_artifacts(args.model_dir)
    if artifacts is None:
        sys.exit(1)

    print(f"SACR Tool — Explainability\n{'='*50}")
    print(f"Best model: {artifacts['best_model_name']}")

    if args.shap:
        print("\n--- SHAP Global ---")
        pipe = artifacts
        x_test_path = Path(args.model_dir) / 'x_test.npz'
        if x_test_path.exists():
            from scipy.sparse import load_npz
            x_test = load_npz(x_test_path)
        else:
            print("  [WARN] x_test not saved, regenerating...")
            x_test = artifacts['best_pipeline'].named_steps['vect'].transform(
                artifacts.get('test_texts', ['dummy'])[:10]
            )

        sv, ex = do_shap_global(artifacts, x_test,
                                artifacts['best_pipeline'].named_steps['vect'].get_feature_names_out(),
                                max_samples=args.shap_samples)
        print("  SHAP values ready. Use --save-plots <dir> to save figures.")

    if args.lime:
        print("\n--- LIME ---")
        if args.text:
            exp, cleaned = do_lime(artifacts, args.text, num_features=args.lime_features)
            print(f"\nExplanation for: {args.text[:200]}")
            print(f"Cleaned: {cleaned[:200]}")
            for cls_idx, cls_name in enumerate(artifacts['class_names']):
                print(f"\nClass: {cls_name}")
                for feat, weight in exp.as_list(label=cls_idx):
                    print(f"  {feat}: {weight:.4f}")


def cmd_evaluate(args):
    artifacts = load_artifacts(args.model_dir)
    if artifacts is None:
        sys.exit(1)

    print(f"SACR Tool — Evaluation\n{'='*50}")
    print(f"Best model: {artifacts['best_model_name']}")
    print(f"Classes: {artifacts['class_names']}")
    print(f"Config: {json.dumps(artifacts.get('config', {}), indent=2)}")

    results_path = Path(args.model_dir) / 'results.csv'
    if results_path.exists():
        df = pd.read_csv(results_path)
        print(f"\nAll Results:\n{df.to_string(index=False)}")


def cmd_reset(args):
    """Erase all trained model artifacts."""
    target = Path(args.model_dir)
    if not target.exists():
        print(f"No artifacts found at '{args.model_dir}' — nothing to reset.")
        return
    if not args.yes:
        ans = input(f"WARNING: This will permanently delete all files in '{target}'. Continue? [y/N] ").strip().lower()
        if ans != 'y':
            print("Reset cancelled.")
            return
    shutil.rmtree(target)
    print(f"All artifacts in '{args.model_dir}' have been deleted.")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='SACR Tool — Complete Sentiment Analysis Pipeline (CLI)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sacr_cli train reviews.csv --text-col review --label-col rating --out my_model
  sacr_cli predict --text "This movie was amazing!" --model-dir my_model
  sacr_cli predict --file test_reviews.txt --model-dir my_model
  sacr_cli explain --shap --model-dir my_model
  sacr_cli explain --lime --text "Terrible film, hated it" --model-dir my_model
  sacr_cli evaluate --model-dir my_model
  sacr_cli reset --model-dir my_model
        """
    )
    parser.add_argument('--version', action='version', version=f"SACR Tool {importlib.metadata.version('sacr-tool')}")

    sub = parser.add_subparsers(dest='command')

    # train
    p_train = sub.add_parser('train', help='Run full training pipeline on a CSV dataset')
    p_train.add_argument('data', help='Path to dataset (CSV/Excel/JSON/TXT)')
    p_train.add_argument('--text-col', help='Text column name (auto-detect if omitted)')
    p_train.add_argument('--label-col', help='Label/rating column name (auto-detect if omitted)')
    p_train.add_argument('--out', '-o', default='sacr_model', help='Output directory for artifacts')
    p_train.add_argument('--test-size', type=float, default=DEFAULTS['test_size'], help='Test split ratio')
    p_train.add_argument('--seed', type=int, default=DEFAULTS['seed'], help='Random seed')
    p_train.add_argument('--max-features', type=int, default=DEFAULTS['max_features'], help='Max vectorizer features')
    p_train.add_argument('--min-df', type=int, default=DEFAULTS['min_df'], help='Min document frequency')
    p_train.add_argument('--ngram-min', type=int, default=DEFAULTS['ngram_min'])
    p_train.add_argument('--ngram-max', type=int, default=DEFAULTS['ngram_max'])
    p_train.add_argument('--vectorizer', choices=['tfidf', 'count'], default=DEFAULTS['vectorizer'])
    p_train.add_argument('--no-neutral', dest='include_neutral', action='store_false', help='Drop neutral class')
    p_train.add_argument('--nb-alpha', type=float, default=DEFAULTS['nb_alpha'], help='Naive Bayes smoothing')
    p_train.add_argument('--nrows', type=int, default=DEFAULTS['nrows'], help='Max rows to load')
    p_train.add_argument('--no-smote', action='store_true', help='Disable SMOTE oversampling')
    p_train.add_argument('--no-calibration', action='store_true', help='Disable probability calibration')
    p_train.add_argument('--no-voting', action='store_true', help='Disable Voting Ensemble')
    p_train.add_argument('--no-grid-search', action='store_true', help='Disable GridSearch for LR')
    p_train.set_defaults(func=cmd_train)

    # predict
    p_pred = sub.add_parser('predict', help='Predict sentiment using a trained model')
    p_pred.add_argument('--model-dir', '-m', default='sacr_model', help='Model artifacts directory')
    p_pred.add_argument('--text', help='Single text to classify')
    p_pred.add_argument('--file', help='File with texts (CSV or one-per-line TXT)')
    p_pred.set_defaults(func=cmd_predict)

    # explain
    p_exp = sub.add_parser('explain', help='Explainability (SHAP / LIME) on trained model')
    p_exp.add_argument('--model-dir', '-m', default='sacr_model', help='Model artifacts directory')
    p_exp.add_argument('--shap', action='store_true', help='Run SHAP global explainability')
    p_exp.add_argument('--shap-samples', type=int, default=200, help='SHAP sample size')
    p_exp.add_argument('--lime', action='store_true', help='Run LIME on a single text')
    p_exp.add_argument('--text', help='Text to explain with LIME')
    p_exp.add_argument('--lime-features', type=int, default=15, help='Number of LIME features')
    p_exp.set_defaults(func=cmd_explain)

    # evaluate
    p_eval = sub.add_parser('evaluate', help='Show evaluation results from a trained model')
    p_eval.add_argument('--model-dir', '-m', default='sacr_model', help='Model artifacts directory')
    p_eval.set_defaults(func=cmd_evaluate)

    # reset
    p_reset = sub.add_parser('reset', help='Erase all trained model artifacts')
    p_reset.add_argument('--model-dir', '-m', default='sacr_model', help='Model artifacts directory to erase')
    p_reset.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompt')
    p_reset.set_defaults(func=cmd_reset)

    args = parser.parse_args()
    if args.command is None:
        print("\n========================================================")
        print("  WELCOME TO SENTIMENT ANALYSIS CUSTOMER REVIEW TOOL")
        print("========================================================")
        print()
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == '__main__':
    main()