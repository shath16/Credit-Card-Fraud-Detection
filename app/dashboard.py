import streamlit as st
import pandas as pd
import xgboost as xgb
import plotly.express as px

st.set_page_config(page_title="Fraud Intelligence", layout="wide", initial_sidebar_state="collapsed")



st.markdown("""
<style>
    .metric-container {
        background-color: #23232E;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #E57373;
        margin-bottom: 20px;
    }

    .header {
        color: #ef5350;
        font-weight: 600;
        padding-bottom: 5px;
    }

    
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {
        color: #C77DFF !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] * {
        color: #E0E0E0 !important;
    }

    div[data-testid="stMetricDelta"],
    div[data-testid="stMetricDelta"] * {
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


st.markdown("<h2 class='header'>Enterprise Fraud Investigation Radar</h2><hr style='border-color: #455A64;'>",
            unsafe_allow_html=True)


@st.cache_resource
def load_model():
    clf = xgb.XGBClassifier()
    clf.load_model("../models/xgboost_fraud_model.json")
    return clf


@st.cache_data
def load_data():
    return pd.read_csv("../data/dashboard_sample.csv")


try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error("System Error: Could not load the Machine Learning model or Dataset. Please check directory paths.")
    st.stop()

if st.button("🚀 Run Full Security Scan on Pending Transactions", type="primary", use_container_width=True):
    with st.spinner("Analyzing multi-dimensional PCA features and identifying threats..."):

        X = df.drop(columns=['Class'])
        probs = model.predict_proba(X)[:, 1]

        df['AI_Risk_Score'] = (probs * 100).round(2)
        df['AI_Decision'] = df['AI_Risk_Score'].apply(lambda x: "BLOCKED" if x > 50 else "CLEARED")

        frauds = df[df['AI_Risk_Score'] > 50].sort_values(by='AI_Risk_Score', ascending=False)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Transactions Scanned", f"{len(df):,}")
        c2.metric("Critical Threats Intercepted", f"{len(frauds)}", "Action Required", delta_color="inverse")
        c3.metric("Financial Loss Prevented", f"${frauds['Amount'].sum():,.2f}")

        st.write("---")

        st.markdown("<h4 style='color: #90CAF9;'>Transaction Topography (Amount vs. Time)</h4>", unsafe_allow_html=True)
        fig = px.scatter(
            df, x="Time", y="Amount", color="AI_Risk_Score",
            color_continuous_scale=["#4DB6AC", "#FFCA28", "#b71c1c"],
            hover_data=['Amount', 'AI_Risk_Score']
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<h4 style='color: #E57373;'>🚨 Critical Threats Ledger (Detailed View)</h4>",
                    unsafe_allow_html=True)

        display_df = frauds[['Time', 'Amount', 'AI_Risk_Score', 'AI_Decision', 'V1', 'V2', 'V3', 'V4', 'V10', 'V14']]

        # بدون استایل‌دهی پانداس، لود مستقیم و بی‌دردسر
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
else:
    st.info("System Ready. Click the scan button to analyze the transaction database.")