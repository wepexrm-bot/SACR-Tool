import streamlit as st
import time
import numpy as np
import pandas as pd
import io
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, roc_auc_score)
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from utils import get_stopwords, clean_text


ALL_CLASSIFIERS = {
    'Logistic Regression': lambda p, s, bw: LogisticRegression(
        C=p.get('C', 1.0), max_iter=p.get('max_iter', 100),
        solver=p.get('solver', 'liblinear'), random_state=s, class_weight=bw),
    'Decision Tree': lambda p, s, bw: DecisionTreeClassifier(
        max_depth=p.get('max_depth', 5), min_samples_split=p.get('min_samples_split', 2),
        criterion=p.get('criterion', 'gini'), random_state=s, class_weight=bw),
    'Random Forest': lambda p, s, bw: RandomForestClassifier(
        n_estimators=p.get('n_estimators', 100), max_depth=p.get('max_depth', 10),
        min_samples_split=p.get('min_samples_split', 2), random_state=s, class_weight=bw),
    'AdaBoost': lambda p, s, bw: AdaBoostClassifier(
        n_estimators=p.get('n_estimators', 50), learning_rate=p.get('learning_rate', 1.0),
        random_state=s),
    'Naive Bayes': lambda p, s, bw: MultinomialNB(alpha=p.get('alpha', 1.0)),
}


def _make_classifier(name, params, seed, use_balanced):
    bw = 'balanced' if use_balanced and name != 'Naive Bayes' else None
    builder = ALL_CLASSIFIERS.get(name)
    if builder is None:
        return None
    return builder(params, seed, bw)


def models_section():
    st.subheader("Machine Learning Models")

    if "feature_engineering_done" not in st.session_state:
        st.session_state.feature_engineering_done = False
    if "models_results" not in st.session_state:
        st.session_state.models_results = []
    if "trained_models_all" not in st.session_state:
        st.session_state.trained_models_all = {}

    if not st.session_state.feature_engineering_done:
        st.warning("Please complete Feature Engineering first.")
        st.stop()

    x_train = st.session_state.x_train
    x_test = st.session_state.x_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    class_names = st.session_state.get("class_names", ["Negative", "Positive"])
    n_classes = len(class_names)
    feature_names = st.session_state.vectorizer.get_feature_names_out()

    # ── Sidebar: Hyperparams ──
    st.sidebar.header("Model Configuration")
    seed = st.sidebar.slider('Random Seed', 1, 200, 42)
    use_balanced = st.sidebar.checkbox("class_weight='balanced'", value=True,
                                       help="Auto-adjust weights for imbalanced data.")

    classifier_name = st.sidebar.selectbox('Classifier:', list(ALL_CLASSIFIERS.keys()))

    def add_parameters(name_of_clf):
        params = {}
        if name_of_clf == 'Logistic Regression':
            params['C'] = st.sidebar.slider('C (inverse reg)', 0.01, 10.0, 1.0)
            params['max_iter'] = st.sidebar.slider('Max iterations', 50, 1000, 100)
            params['solver'] = st.sidebar.selectbox('Solver', ['liblinear', 'lbfgs', 'newton-cg'])
        elif name_of_clf == 'Decision Tree':
            params['max_depth'] = st.sidebar.slider('Max depth', 1, 30, 5)
            params['min_samples_split'] = st.sidebar.slider('Min samples split', 2, 20, 2)
            params['criterion'] = st.sidebar.selectbox('Criterion', ('gini', 'entropy'))
        elif name_of_clf == 'Random Forest':
            params['n_estimators'] = st.sidebar.slider('Trees', 10, 200, 100)
            params['max_depth'] = st.sidebar.slider('Max depth', 1, 30, 10)
            params['min_samples_split'] = st.sidebar.slider('Min samples split', 2, 10, 2)
        elif name_of_clf == 'AdaBoost':
            params['n_estimators'] = st.sidebar.slider('Estimators', 10, 200, 50)
            params['learning_rate'] = st.sidebar.slider('Learning rate', 0.01, 2.0, 1.0)
        elif name_of_clf == 'Naive Bayes':
            params['alpha'] = st.sidebar.slider('Alpha (smoothing)', 0.01, 10.0, 1.0)
        return params

    params = add_parameters(classifier_name)

    # ── Train Single Model ──
    if st.button(f"Train and Test {classifier_name}"):
        _train_and_record(classifier_name, params, seed, use_balanced,
                          x_train, y_train, x_test, y_test, class_names)

    # ── Train All 5 Models ──
    if st.button("Train All 5 Models (for XAI)", type="primary"):
        st.session_state.trained_models_all = {}
        st.session_state.models_results = []
        default_params = {
            'Logistic Regression': {'C': 1.0, 'max_iter': 100, 'solver': 'liblinear'},
            'Decision Tree': {'max_depth': 5, 'min_samples_split': 2, 'criterion': 'gini'},
            'Random Forest': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2},
            'AdaBoost': {'n_estimators': 50, 'learning_rate': 1.0},
            'Naive Bayes': {'alpha': 1.0},
        }
        prog = st.progress(0, text="Training all models...")
        for idx, (name, _) in enumerate(ALL_CLASSIFIERS.items()):
            prog.progress((idx) / len(ALL_CLASSIFIERS), text=f"Training {name}...")
            _train_and_record(name, default_params.get(name, {}), seed, use_balanced,
                              x_train, y_train, x_test, y_test, class_names,
                              store_model=True)
        st.session_state.trained_all_models = True
        prog.progress(1.0, text="All models trained!")
        st.success("All 5 models trained and stored for Explainability tab!")

    # ── Trained models quick status ──
    if st.session_state.trained_models_all:
        names = list(st.session_state.trained_models_all.keys())
        st.info(f"Trained models stored: {', '.join(names)}. See Explainability tab for SHAP/LIME.")

    # ═══════════════════════════════════════════════════
    #  SECTION: Feature Importance (coefficient-based)
    # ═══════════════════════════════════════════════════
    if st.session_state.trained_models_all or st.session_state.get("trained_model") is not None:
        with st.expander("Feature Importance Analysis", expanded=False):
            models_pool = st.session_state.trained_models_all if st.session_state.trained_models_all else {}
            latest = st.session_state.get("trained_model")
            if models_pool:
                pool = {n: models_pool[n]['model'] for n in models_pool}
            else:
                pool = {classifier_name: latest}

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.ravel()
            ax_idx = 0
            eligible = [n for n in ['Random Forest', 'Decision Tree', 'AdaBoost', 'Logistic Regression'] if n in pool]
            for name in eligible:
                clf = pool[name]
                if clf is None:
                    continue
                if hasattr(clf, 'coef_'):
                    coef = clf.coef_
                    imp = np.abs(coef).mean(axis=0) if coef.shape[0] > 1 else np.abs(coef[0])
                elif hasattr(clf, 'feature_importances_'):
                    imp = clf.feature_importances_
                else:
                    continue
                top_n = 15
                top_idx = np.argsort(imp)[-top_n:]
                axes[ax_idx].barh(range(top_n), imp[top_idx], color='steelblue')
                axes[ax_idx].set_yticks(range(top_n))
                axes[ax_idx].set_yticklabels([feature_names[i] for i in top_idx], fontsize=8)
                axes[ax_idx].set_title(f'Top {top_n} Features — {name}')
                axes[ax_idx].invert_yaxis()
                ax_idx += 1
            for j in range(ax_idx, len(axes)):
                axes[j].axis('off')
            plt.tight_layout()
            st.pyplot(fig)

    # ═══════════════════════════════════════════════════
    #  SECTION: Confusion Matrices for ALL Models
    # ═══════════════════════════════════════════════════
    if st.session_state.trained_models_all:
        with st.expander("Confusion Matrices — All Models", expanded=False):
            pool = st.session_state.trained_models_all
            names = list(pool.keys())
            n_models = len(names)
            n_cols = min(3, n_models)
            n_rows = -(-n_models // n_cols)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
            axes_flat = axes.ravel() if n_models > 1 else [axes]
            for idx, name in enumerate(names):
                y_pred_m = pool[name]['model'].predict(x_test)
                cm = confusion_matrix(y_test, y_pred_m)
                ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
                    ax=axes_flat[idx], cmap='Blues', values_format='d')
                axes_flat[idx].set_title(name)
            for j in range(idx + 1, len(axes_flat)):
                axes_flat[j].axis('off')
            plt.tight_layout()
            st.pyplot(fig)

    # ── Misclassification Analysis ──
    if st.session_state.get("y_pred") is not None:
        with st.expander("Misclassification Analysis", expanded=False):
            y_pred_s = st.session_state.y_pred
            y_test_s = y_test
            x_test_texts = st.session_state.get("x_test_texts", None)
            mis_idx = np.where(y_pred_s != y_test_s)[0]
            st.write(f"**Total misclassified:** {len(mis_idx)} / {len(y_test_s)} ({len(mis_idx)/len(y_test_s):.1%})")
            for true_cls in range(n_classes):
                for pred_cls in range(n_classes):
                    if true_cls == pred_cls:
                        continue
                    idxs = np.where((y_test_s == true_cls) & (y_pred_s == pred_cls))[0]
                    if len(idxs) == 0:
                        continue
                    label = f"True: {class_names[true_cls]} → Pred: {class_names[pred_cls]} ({len(idxs)} cases)"
                    if st.checkbox(label, key=f"mis_{true_cls}_{pred_cls}_{classifier_name}"):
                        if x_test_texts is not None:
                            for idx in idxs[:10]:
                                st.text(f"[{idx}] {x_test_texts[idx][:200]}")
                        else:
                            st.info("Test texts not available.")

    # ── Real-time Prediction ──
    if st.session_state.get("trained_model") is not None:
        with st.expander("Test with Custom Text", expanded=True):
            sample_text = st.text_area("Enter your text:", "", height=100,
                                       key=f"sample_text_{classifier_name}")
            if st.button("Predict Sentiment", key=f"predict_btn_{classifier_name}"):
                if sample_text.strip():
                    stop_words = get_stopwords()
                    cleaned = clean_text(sample_text, stop_words)
                    vec = st.session_state.vectorizer.transform([cleaned])
                    model = st.session_state.trained_model
                    pred = model.predict(vec)[0]
                    label = class_names[pred] if pred < len(class_names) else ("Positive" if pred == 1 else "Negative")
                    st.success(f"**Prediction:** {label}")
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(vec)[0]
                        for cname, p in zip(class_names, proba):
                            st.metric(f"P({cname})", f"{p:.2%}")
                    st.caption(f"Cleaned: _{cleaned[:200]}{'...' if len(cleaned) > 200 else ''}_")
                else:
                    st.warning("Please enter some text to analyze.")


def _train_and_record(name, params, seed, use_balanced,
                      x_train, y_train, x_test, y_test, class_names,
                      store_model=False):
    clf = _make_classifier(name, params, seed, use_balanced)
    start = time.time()
    clf.fit(x_train, y_train)
    elapsed = time.time() - start
    y_pred = clf.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_m = f1_score(y_test, y_pred, average='macro', zero_division=0)

    if acc >= 0.999:
        st.error(f"Accuracy of {acc:.4f} is suspiciously perfect — possible data leakage.")

    st.success(f"{name} trained (acc={acc:.4f}) in {elapsed:.2f}s")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("Precision (w)", f"{prec:.4f}")
    col3.metric("Recall (w)", f"{rec:.4f}")
    col4.metric("F1 (weighted)", f"{f1_w:.4f}")
    st.metric("F1 (macro)", f"{f1_m:.4f}")
    st.info(f"Training Time: {elapsed:.2f}s")

    report = classification_report(y_test, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    model_result = {
        'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec,
        'F1_Score': f1_w, 'F1_Macro': f1_m, 'Training_Time': elapsed, 'Parameters': params
    }
    st.session_state.models_results.append(model_result)

    if store_model:
        st.session_state.trained_models_all[name] = {
            'model': clf, 'params': params, 'accuracy': acc,
            'precision': prec, 'recall': rec, 'f1_weighted': f1_w, 'f1_macro': f1_m
        }
    else:
        st.session_state.trained_model = clf
        st.session_state.y_pred = y_pred

    # Download CSV
    export = {k: v for k, v in model_result.items() if k != 'Parameters'}
    export['classes'] = class_names
    export['timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    csv = pd.DataFrame([export]).to_csv(index=False)
    st.download_button("Download Results CSV", csv,
                       f"sacr_{name.lower().replace(' ', '_')}.csv", "text/csv",
                       key=f"csv_{name}_{int(time.time())}")

    # Download PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, text="Sentiment Analysis Model Report", ln=True, align='C')
    for k, v in export.items():
        pdf.cell(200, 10, text=f"{k}: {v}", ln=True, align='L')
    pdf_path = f"sacr_{name.lower().replace(' ', '_')}.pdf"
    pdf.output(pdf_path)
    with open(pdf_path, "rb") as f:
        st.download_button("Download Results PDF", f.read(), pdf_path, "application/pdf",
                           key=f"pdf_{name}_{int(time.time())}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5 + len(class_names), 4 + len(class_names) // 2))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap='Blues', values_format='d')
    plt.tight_layout()
    st.pyplot(fig)

    # ROC-AUC
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(x_test)
        n_cls = len(class_names)
        if n_cls == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            auc_val = roc_auc_score(y_test, y_prob[:, 1])
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            ax2.plot(fpr, tpr, label=f"ROC (AUC={auc_val:.4f})")
            ax2.plot([0, 1], [0, 1], "k--", label="Random")
            ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
            ax2.set_title(f"ROC — {name}"); ax2.legend()
            plt.tight_layout(); st.pyplot(fig2)
            st.metric("AUC", f"{auc_val:.4f}")
        else:
            y_bin = label_binarize(y_test, classes=range(n_cls))
            auc_macro = roc_auc_score(y_bin, y_prob, average='macro', multi_class='ovr')
            fig2, ax2 = plt.subplots(figsize=(7, 6))
            for i, cn in enumerate(class_names):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                ca = roc_auc_score(y_bin[:, i], y_prob[:, i])
                ax2.plot(fpr, tpr, label=f'{cn} (AUC={ca:.3f})')
            ax2.plot([0, 1], [0, 1], "k--", label="Random")
            ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
            ax2.set_title(f"ROC (OvR) — {name}"); ax2.legend()
            plt.tight_layout(); st.pyplot(fig2)
            st.metric("Macro AUC", f"{auc_macro:.4f}")
    else:
        st.info(f"{name} does not support predict_proba.")

    # Download model
    buf = io.BytesIO()
    joblib.dump(clf, buf)
    buf.seek(0)
    st.download_button("Download Model (.joblib)", buf,
                       f"{name.lower().replace(' ', '_')}_model.joblib",
                       "application/octet-stream",
                       key=f"dl_model_{name}_{int(time.time())}")
