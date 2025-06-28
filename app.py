import streamlit as st
import pandas as pd
import numpy as np
import re
import openai
from analysis_utils import *
from utils_text import *
from analysis_utils import t_test_analysis

# ✅ App Config
st.set_page_config(page_title="📊 Smart Data Analyzer", layout="wide")
st.title("📊 Smart Data Analyzer")

# ✅ OpenAI API Client
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ✅ AI yorum fonksiyonu
def ai_interpretation(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that analyzes data and provides insights. You can highlight anomalies, interpret correlations between attributes, find and tell similarities or impact from other attributes."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        raw_message = response.choices[0].message.content.strip()
        sentences = re.findall(r'[^.!?]*[.!?]', raw_message)
        return ''.join(sentences).strip()
    except Exception as e:
        return f"**Error during AI interpretation:** {e}"

# ✅ File Upload
uploaded_file = st.file_uploader(
    "Upload your dataset (CSV, Excel, JSON, XML, Feather)",
    type=["csv", "xlsx", "xls", "json", "xml", "feather"]
)

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        elif uploaded_file.name.endswith('.xml'):
            df = pd.read_xml(uploaded_file)
        elif uploaded_file.name.endswith('.feather'):
            df = pd.read_feather(uploaded_file)
        else:
            st.error("Unsupported file format.")
            st.stop()
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    # ✅ Data Preview
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # ✅ Analysis Selection
    option = st.selectbox("Select Analysis Type", [
        "Numeric Summary",
        "Correlation Matrix",
        "Chi-Square Test",
        "T-Test"
    ])

    if option == "Numeric Summary":
        result = analyze_numeric(df)
        st.write(result)

        prompt = f"Analyze the following numeric summary statistics and provide insights:\n{result.to_string()}"
        ai_result = ai_interpretation(prompt)
        st.markdown("### AI Insights")
        st.write(ai_result)

    elif option == "Correlation Matrix":
        fig, corr_df = correlation_plot(df)

        # Geniş grafik görünümü
        fig.update_layout(
            width=1000,
            height=700,
            margin=dict(l=50, r=50, t=50, b=50),
        )

        st.plotly_chart(fig)

        prompt = f"Explain the key points and findings from this correlation matrix:\n{corr_df.to_string()}"
        ai_result = ai_interpretation(prompt)
        st.markdown("### AI Insights")
        st.write(ai_result)

    elif option == "Chi-Square Test":
        categorical_cols = df.select_dtypes(include=['object', 'category'])
