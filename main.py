import os
import logging
import streamlit as st
import praw
import boto3
from src.monitoring import Monitoring
from src.llm import OpinionQuantifier, GuardrailChecker
from dotenv import load_dotenv


load_dotenv()
# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Load Environment Variables & Initialize Services
# -----------------------------------------------------------------------------
reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")
embedding_model_id = os.getenv("EMBEDDING_MODEL_ID")
llm_model_id = os.getenv("LLM_MODEL_ID")
guardrail_id = os.getenv("GUARDRAIL_ID")
guardrail_version = os.getenv("GUARDRAIL_VERSION")


BEDROCK_CLIENT = boto3.client("bedrock-runtime", 
                          region_name=aws_region,
                          aws_access_key_id=aws_access_key_id,
                          aws_secret_access_key=aws_secret_access_key)

REDDIT_CLIENT = praw.Reddit(
    client_id=reddit_client_id,
    client_secret=reddit_client_secret,
    user_agent="work_life_pipeline_agent"
)

monitoring = Monitoring()
opinion_quantifier = OpinionQuantifier(BEDROCK_CLIENT, monitoring, llm_model_id)
guardrail_checker = GuardrailChecker(BEDROCK_CLIENT, monitoring, guardrail_id, guardrail_version)
vector_store = RAGVectorStore(BEDROCK_CLIENT, AWS_REGION, embedding_model_id)

# Streamlit UI
st.title("Work-Life Balance Analyzer")

subreddit_name = "all"
query = "work life balance"
limit = st.slider("Number of posts", min_value=1, max_value=50, value=10)
if st.button("Run Analysis"):
    with st.spinner("Fetching posts..."):
        subreddit = REDDIT_CLIENT.subreddit(subreddit_name)
        posts = {sub.id: f"{sub.title}\n{sub.selftext}" for sub in subreddit.search(query, limit=limit)}

    retriever = vector_store.build_vector_store(posts)
    question = st.text_input("Query", value="along with work you should try to spend quality time with your parents")
    doc_txt = retriever.invoke(question)[0].page_content

    if guardrail_checker.apply_guardrail(question)["action"] != "GUARDRAIL_INTERVENED":
        st.json(opinion_quantifier.quantify_opinion(question, doc_txt))
