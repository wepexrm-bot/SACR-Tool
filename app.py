import streamlit as st
import io, zipfile, json
from pathlib import Path
from utils import initialize_session_state
from sections.preprocessing import preprocessing_section
from sections.eda import eda_section
from sections.visualisation import visualisation_section
from sections.feature_engineering import feature_engineering_section
from sections.models import models_section
from sections.model_comparison import model_comparison_section
from sections.explainability import explainability_section
from sections.about import about_section
from sentiment_modeltrainer import _clean_text


def test_predictions_section():
    st.subheader("Test Predictions — Use a Trained Model")
    st.markdown("Upload a trained model to make predictions without re-training.")

    col1, col2 = st.columns(2)
    with col1:
        model_zip = st.file_uploader("Upload model package (.zip)", type=["zip"])
    with col2:
        model_joblib = st.file_uploader("Or upload full pipeline (.joblib)", type=["joblib"])

    if model_zip is None and model_joblib is None:
        st.info("Train a model using **Train All 5 Models** then download the full Pipeline, or use `sacr train` and upload here.")
        return

    import joblib

    if model_zip is not None:
        with st.spinner("Loading model package..."):
            with zipfile.ZipFile(io.BytesIO(model_zip.getvalue())) as z:
                z.extractall("_loaded_model")
            pipeline = joblib.load("_loaded_model/best_pipeline.joblib")
            with open("_loaded_model/meta.json") as f:
                meta = json.load(f)
            class_names = meta['class_names']
            best_name = meta['best_model_name']
    else:
        with st.spinner("Loading pipeline..."):
            pipeline = joblib.load(io.BytesIO(model_joblib.getvalue()))
            if hasattr(pipeline, 'named_steps'):
                clf_step = [s for s in pipeline.named_steps if s != 'vect'][0]
                clf = pipeline.named_steps[clf_step]
            else:
                clf = pipeline
            if hasattr(clf, 'classes_'):
                class_names = list(clf.classes_)
            elif hasattr(clf, 'calibrated_classifiers_'):
                class_names = list(clf.calibrated_classifiers_[0].classes_)
            else:
                class_names = ['negative', 'positive']
            best_name = "Uploaded Model"

    st.success(f"Model loaded! **{best_name}** | Classes: {class_names}")

    user_text = st.text_area("Enter text to classify:", "", height=120,
                             placeholder="This movie was amazing!")
    if st.button("Predict", type="primary") and user_text.strip():
        cleaned = _clean_text(user_text)
        if not hasattr(pipeline, 'named_steps'):
            st.error("Cannot predict: this file is a standalone classifier without a vectorizer. "
                     "Upload a full Pipeline (.joblib) or a model package (.zip).")
            st.stop()
        pred = pipeline.predict([cleaned])[0]
        label = class_names[int(pred)]
        probs = pipeline.predict_proba([cleaned])[0]
        st.write(f"### Prediction: **{label}**")
        for i, cn in enumerate(class_names):
            st.metric(f"P({cn})", f"{probs[i]:.2%}")
        st.caption(f"Cleaned: _{cleaned[:200]}{'...' if len(cleaned) > 200 else ''}_")

    import shutil
    if Path("_loaded_model").exists():
        shutil.rmtree("_loaded_model", ignore_errors=True)


st.title("SACR Tool (Sentiment Analysis on Customer Review)")


def web():
    initialize_session_state()
    activities = ['Data Preprocessing', 'EDA', 'Visualisation', 'Feature Engineering',
                  'Models', 'Model Comparison', 'Explainability (XAI)',
                  'Test Predictions', 'About Us']

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
        preprocessing_section()
    elif option == 'EDA':
        eda_section()
    elif option == 'Visualisation':
        visualisation_section()
    elif option == 'Feature Engineering':
        feature_engineering_section()
    elif option == 'Models':
        models_section()
    elif option == 'Model Comparison':
        model_comparison_section()
    elif option == 'Explainability (XAI)':
        explainability_section()
    elif option == 'Test Predictions':
        test_predictions_section()
    elif option == 'About Us':
        about_section()


if __name__ == '__main__':
    web()

with st.sidebar:
    st.markdown("---")
    st.markdown("<div style='height: 250px;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Reset All", help="Clear all session data and start fresh"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()