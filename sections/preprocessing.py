import streamlit as st
import pandas as pd
import numpy as np
from utils import validate_data, get_stopwords, preprocess_with_progress, clean_text


def preprocessing_section():
    st.subheader("Data Preprocessing")

    data = st.file_uploader("Upload dataset:", type=['csv', 'xlsx', 'txt', 'json'])
    if data is not None:
        try:
            if data.name.endswith('.csv'):
                df = pd.read_csv(data)
            elif data.name.endswith('.xlsx'):
                df = pd.read_excel(data)
            elif data.name.endswith('.json'):
                df = pd.read_json(data)
            else:
                df = pd.read_csv(data, sep='\t')

            st.success("✅ Data successfully loaded")

            issues = validate_data(df)
            if issues:
                st.warning("⚠️ Data Quality Issues:")
                for issue in issues:
                    st.write(f"- {issue}")

            st.session_state.current_df = df
            st.session_state.data_loaded = True

            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            with col2:
                st.info(f"💾 Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

            st.dataframe(df.head(50), use_container_width=True)

            st.subheader("Missing Values Overview")
            missing_info = df.isnull().sum()
            missing_info = missing_info[missing_info > 0]
            if not missing_info.empty:
                st.write("Columns with missing values:")
                st.dataframe(missing_info.to_frame('Missing Values'), use_container_width=True)
            else:
                st.info("No missing values detected.")

            potential_sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower() or 'label' in col.lower()]

            if potential_sentiment_cols:
                st.subheader("🧠 Sentiment Conversion (Categorical to Binary)")
                selected_sent_col = st.selectbox("Select sentiment column to convert to binary:", potential_sentiment_cols)

                if df[selected_sent_col].dtype == 'object':
                    df['binary_sentiment_label'] = df[selected_sent_col].astype(str).str.lower().map(
                        lambda x: 1 if any(p in x for p in ['positive', 'pos', 'good', 'excellent'])
                                else 0 if any(n in x for n in ['negative', 'neg', 'bad', 'poor'])
                                else np.nan
                    )

                    st.write("📋 Mapping Preview")
                    st.dataframe(df[[selected_sent_col, 'binary_sentiment_label']].dropna().head(10), use_container_width=True)

                    unmapped = df['binary_sentiment_label'].isna().sum()
                    if unmapped > 0:
                        st.warning(f"⚠️ {unmapped} rows could not be mapped and have NaN in the new label column.")
                    else:
                        st.success("✅ All sentiment values successfully mapped to binary!")

            text_columns = [col for col in df.columns if df[col].dtype == 'object']
            selected_cols = st.multiselect("Select columns to use for text analysis:", text_columns)

            if selected_cols:
                st.subheader("🛠️ Preprocessing Options")
                use_progress = st.checkbox("Show progress bar (recommended for large datasets)", value=True)

                if st.button("🚀 Start Preprocessing"):
                    with st.spinner("Processing text data..."):
                        df['analysis_text'] = df[selected_cols].fillna('').agg(' '.join, axis=1)
                        stop_words = get_stopwords()

                        if use_progress:
                            processed_texts = preprocess_with_progress(df, 'analysis_text', stop_words)
                            df['analysis_text_clean'] = processed_texts
                        else:
                            df['analysis_text_clean'] = df['analysis_text'].apply(lambda x: clean_text(x, stop_words))

                        st.session_state.current_df = df
                        st.session_state.preprocessing_done = True

                    st.success("✅ Text preprocessing completed!")

                    preview_df = df[['analysis_text', 'analysis_text_clean']].head(10)
                    st.subheader("📄 Preprocessing Results Preview")
                    st.dataframe(preview_df, use_container_width=True)

                    avg_original_length = df['analysis_text'].str.len().mean()
                    avg_clean_length = df['analysis_text_clean'].str.len().mean()
                    reduction_pct = ((avg_original_length - avg_clean_length) / avg_original_length * 100)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg Original Length", f"{avg_original_length:.1f} chars")
                    with col2:
                        st.metric("Avg Clean Length", f"{avg_clean_length:.1f} chars")
                    with col3:
                        st.metric("Reduction", f"{reduction_pct:.1f}%")

            else:
                st.warning("⚠️ Please select at least one text column for analysis.")

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
