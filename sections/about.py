import streamlit as st


def about_section():
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
