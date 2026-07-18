import streamlit as st
from utils import initialize_session_state
from sections.preprocessing import preprocessing_section
from sections.eda import eda_section
from sections.visualisation import visualisation_section
from sections.feature_engineering import feature_engineering_section
from sections.models import models_section
from sections.model_comparison import model_comparison_section
from sections.about import about_section


st.title("SACR Tool (Sentiment Analysis on Customer Review)")


def web():
    initialize_session_state()
    activities = ['Data Preprocessing', 'EDA', 'Visualisation', 'Feature Engineering', 'Models', 'Model Comparison', 'About Us']

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
