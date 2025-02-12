import streamlit as st
import os
import logging
import boto3
import json
from src.pipeline import WorkLifeBalancePipeline
from src.monitoring import Monitoring

# -----------------------------------------------------------------------------
# Setup Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Initialize Monitoring & AI Pipeline
# -----------------------------------------------------------------------------
monitoring = Monitoring()
bedrock_client = boto3.client("bedrock-runtime", region_name="ap-south-1")

pipeline = WorkLifeBalancePipeline(
    reddit_fetcher=None,  # Initialized in Streamlit UI
    guardrail_checker=None,
    opinion_quantifier=None,
    monitoring=monitoring
)

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.title("💼 Work-Life Balance AI 🔍")

st.sidebar.header("🔍 Query Settings")
subreddit_name = st.sidebar.text_input("Subreddit Name", "all")
query = st.sidebar.text_input("Search Query", "work life balance")
limit = st.sidebar.slider("Number of Posts", 1, 100, 10)

if st.sidebar.button("Run Analysis"):
    with st.spinner("Fetching posts & analyzing... ⏳"):
        result = pipeline.run(subreddit_name, query, reddit_limit=limit, guardrail_enabled=True)
        
        if result.get("guardrail_intervened"):
            st.warning("⚠️ Guardrail Intervened: Unsafe content detected.")
            st.write(result["outputs"])
        else:
            st.success("✅ Analysis Complete!")
            st.json(result)

st.sidebar.markdown("**👀 AI-powered Work-Life Balance Analysis**")
st.sidebar.text("🔹 Fetches Reddit discussions")
st.sidebar.text("🔹 Applies safety guardrails")
st.sidebar.text("🔹 Quantifies work-life balance")

# Display the Datapane report
if os.path.exists("monitoring_report.html"):
    st.sidebar.markdown("📊 **View Report:**")
    st.sidebar.markdown("[Click to Open Report](monitoring_report.html)", unsafe_allow_html=True)
