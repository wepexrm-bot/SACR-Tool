import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import compare_models


def model_comparison_section():
    st.subheader("Model Performance Comparison")

    if st.session_state.models_results:
        compare_models(st.session_state.models_results)

        if st.checkbox("📈 Show Performance Charts"):
            results_df = pd.DataFrame(st.session_state.models_results)

            metrics = [m for m in ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'F1_Macro'] if m in results_df.columns]
            n_plots = len(metrics)
            n_cols = 3 if n_plots > 4 else 2
            n_rows = -(-n_plots // n_cols)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
            axes = axes.ravel()
            axes = axes.ravel()

            for i, metric in enumerate(metrics):
                if metric in results_df.columns:
                    sns.barplot(data=results_df, x='Model', y=metric, ax=axes[i])
                    axes[i].set_title(f'{metric} Comparison')
                    axes[i].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            st.pyplot(fig)

            if 'Training_Time' in results_df.columns:
                st.subheader("⏱️ Training Time Comparison")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=results_df, x='Model', y='Training_Time', ax=ax)
                ax.set_title('Training Time Comparison (seconds)')
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)

        if st.button("🗑️ Clear All Results"):
            st.session_state.models_results = []
            st.success("✅ All model results cleared!")
            st.rerun()

    else:
        st.info("🤔 No model results available yet. Train some models first!")

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

                        model_result = {
                            'Model': model_name,
                            'Accuracy': accuracy,
                            'Precision': precision,
                            'Recall': recall,
                            'F1_Score': f1,
                            'Training_Time': None,
                            'Parameters': 'Default'
                        }

                        st.session_state.models_results.append(model_result)

                st.success("✅ All models trained and added to comparison!")
                st.rerun()
        else:
            st.warning("Please complete model training to get customized model comparision !")
