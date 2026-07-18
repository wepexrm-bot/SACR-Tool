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
from utils import get_stopwords, clean_text


def models_section():
    st.subheader("🤖 Machine Learning Models")

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
                clf = get_classifier(classifier_name, params)
                start_time = time.time()
                clf.fit(x_train, y_train)
                st.session_state.trained_model = clf
                st.session_state.X_test = x_test
                st.session_state.y_test = y_test
                st.session_state.tfidf_vectorizer = st.session_state.vectorizer

                training_time = time.time() - start_time

                y_pred = clf.predict(x_test)
                st.session_state.y_pred = y_pred

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')

                st.success(f"✅ {classifier_name} Training Completed!")

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

                st.subheader("📋 Detailed Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)

                model_result = {
                    'Model': classifier_name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1_Score': f1,
                    'Training_Time': training_time,
                    'Parameters': params
                }

                st.session_state.models_results.append(model_result)

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

                with st.expander("📊 Confusion Matrix", expanded=False):
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                    ConfusionMatrixDisplay(cm, display_labels=['Negative', 'Positive']).plot(ax=ax_cm)
                    plt.tight_layout()
                    st.pyplot(fig_cm)

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
                    vect_for_dl = st.session_state.vectorizer
                    orig_tokenizer = getattr(vect_for_dl, 'tokenizer', None)
                    if orig_tokenizer is not None:
                        vect_for_dl.tokenizer = None
                    joblib.dump(vect_for_dl, buf_vect)
                    buf_vect.seek(0)
                    if orig_tokenizer is not None:
                        vect_for_dl.tokenizer = orig_tokenizer
                    st.download_button(
                        label="⬇️ Download Vectorizer (.joblib)",
                        data=buf_vect,
                        file_name=f"{classifier_name.lower().replace(' ', '_')}_vectorizer.joblib",
                        mime="application/octet-stream",
                        key=f"download_vect_{classifier_name}"
                    )

        # --- False Positive / False Negative Analysis ---
        if st.session_state.get("y_pred") is not None and st.session_state.get("y_test") is not None:
            with st.expander("🔍 False Positive / False Negative Analysis", expanded=False):
                y_pred_s = st.session_state.y_pred
                y_test_s = st.session_state.y_test
                fp = np.where((y_pred_s == 1) & (y_test_s == 0))[0]
                fn = np.where((y_pred_s == 0) & (y_test_s == 1))[0]
                st.write(f"**False Positives:** {len(fp)}  |  **False Negatives:** {len(fn)}")
                if st.checkbox("Show False Positive examples", key=f"show_fp_outside_{classifier_name}"):
                    x_test_texts = st.session_state.get("x_test_texts", None)
                    if x_test_texts is not None:
                        for idx in fp[:10]:
                            st.text(f"[{idx}] {x_test_texts[idx][:300]}")
                    else:
                        st.info("Test texts not available. Re-run Feature Engineering.")
                if st.checkbox("Show False Negative examples", key=f"show_fn_outside_{classifier_name}"):
                    x_test_texts = st.session_state.get("x_test_texts", None)
                    if x_test_texts is not None:
                        for idx in fn[:10]:
                            st.text(f"[{idx}] {x_test_texts[idx][:300]}")
                    else:
                        st.info("Test texts not available. Re-run Feature Engineering.")

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
