import os
import json
import logging
import praw
import boto3
import unittest
import pandas as pd
import datapane as dp
from datetime import datetime
from unittest.mock import MagicMock, patch
from botocore.exceptions import ClientError
from monitoring import Monitoring
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import BedrockEmbeddings
from chromadb import EmbeddingFunction, Embeddings
# from dotenv import load_dotenv


# load_dotenv()

# reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
# reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
# aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
# aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
# aws_region = os.getenv("AWS_REGION")
# embedding_model_id = os.getenv("EMBEDDING_MODEL_ID")
# llm_model_id = os.getenv("LLM_MODEL_ID")
# guardrail_id = os.getenv("GUARDRAIL_ID")
# guardrail_version = os.getenv("GUARDRAIL_VERSION")



# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# # -----------------------------------------------------------------------------
# # Reddit Fetcher Module
# # -----------------------------------------------------------------------------
# class RedditFetcher:
#     def __init__(self, user_agent: str, monitoring: Monitoring):
#         self.reddit = praw.Reddit(
#             client_id=reddit_client_id,
#             client_secret=reddit_client_secret,
#             user_agent=user_agent
#         )
#         self.monitoring = monitoring

#     def fetch_posts(self, subreddit_name: str, query: str, limit: int = 100) -> dict:
#         """Fetch posts from a subreddit matching the query."""
#         subreddit = self.reddit.subreddit(subreddit_name)
#         posts = {}
#         try:
#             for submission in subreddit.search(query, limit=limit):
#                 content = f"{submission.title}\n{submission.selftext}"
#                 posts[submission.id] = content
#         except Exception as e:
#             logger.error("Error fetching posts from Reddit: %s", e)
#             raise
        
#         self.monitoring.log_metric("Posts Fetched", len(posts))
#         return posts

# # -----------------------------------------------------------------------------
# # Guardrail Checker Module
# # -----------------------------------------------------------------------------
# class GuardrailChecker:
#     def __init__(self, bedrock_client, monitoring: Monitoring):
#         self.bedrock_client = bedrock_client
#         self.monitoring = monitoring
#         self.guardrail_id = guardrail_id
#         self.version = guardrail_version

#     def apply_guardrail(self, question: str) -> dict:
#         """Applies the guardrail check before processing the question."""
#         content = [{"text": {"text": question}}]
#         try:
#             response = self.bedrock_client.apply_guardrail(
#                 guardrailIdentifier=self.guardrail_id,
#                 guardrailVersion=self.version,
#                 source='INPUT',
#                 content=content
#             )
#             action = response.get("action", "UNKNOWN")
#             self.monitoring.log_metric("Guardrail Interventions", 1 if action == "GUARDRAIL_INTERVENED" else 0)
#             return response
#         except Exception as e:
#             logger.error("Guardrail check failed: %s", e)
#             raise

# # -----------------------------------------------------------------------------
# # Opinion Quantifier Module
# # -----------------------------------------------------------------------------
# class OpinionQuantifier:
#     def __init__(self, bedrock_client, monitoring: Monitoring):
#         self.bedrock_client = bedrock_client
#         self.monitoring = monitoring

#     def quantify_opinion(self, text: str) -> dict:
#         """Analyzes discussion sentiment using LLM."""
#         prompt = f"""Analyze work–life balance discussion:
# 1. Support vs. opposition (-10 to 10)
# 2. Work–life balance score (1-10)
# 3. Stress impact (1-10)
# 4. Family time emphasis (1-10)

# Discussion:
# {text}

# Output JSON format with keys: support_level, balance_score, stress_impact, personal_time.
# """
#         formatted_prompt = f"<|begin_of_text|>{prompt}<|end_of_text|>"
#         native_request = {"prompt": formatted_prompt, "max_gen_len": 512, "temperature": 0.5}
#         request_json = json.dumps(native_request)

#         try:
#             response = self.bedrock_client.invoke_model(
#                 ModelId=llm_model_id,
#                 ContentType='application/json',
#                 Body=request_json
#             )
#             model_response = json.loads(response["body"].read())
            
#             # Log opinion quantification metrics
#             self.monitoring.log_metric("Support Level", model_response.get("support_level", 0))
#             self.monitoring.log_metric("Balance Score", model_response.get("balance_score", 0))
#             self.monitoring.log_metric("Stress Impact", model_response.get("stress_impact", 0))
#             self.monitoring.log_metric("Family Time", model_response.get("personal_time", 0))
            
#             return model_response
#         except (ClientError, Exception) as e:
#             logger.error("Failed to invoke model: %s", e)
#             return {"error": "Failed to get opinion"}

# # -----------------------------------------------------------------------------
# # Pipeline Module
# # -----------------------------------------------------------------------------
# class WorkLifeBalancePipeline:
#     def __init__(self, reddit_fetcher, guardrail_checker, opinion_quantifier, monitoring):
#         self.reddit_fetcher = reddit_fetcher
#         self.guardrail_checker = guardrail_checker
#         self.opinion_quantifier = opinion_quantifier
#         self.monitoring = monitoring

#     def run(self, subreddit_name: str, query: str, reddit_limit: int = 10, guardrail_enabled: bool = True) -> dict:
#         """Runs the full pipeline."""
#         posts = self.reddit_fetcher.fetch_posts(subreddit_name, query, limit=reddit_limit)
#         sample_text = next(iter(posts.values()), "No data available.")

#         if guardrail_enabled:
#             guard_response = self.guardrail_checker.apply_guardrail(sample_text)
#             if guard_response.get("action") == "GUARDRAIL_INTERVENED":
#                 logger.info("Guardrail intervened: %s", guard_response.get("outputs"))
#                 return {"guardrail_intervened": True, "outputs": guard_response.get("outputs")}

#         opinion = self.opinion_quantifier.quantify_opinion(sample_text)
#         self.monitoring.generate_report()  # Generate Datapane report at the end
#         return opinion

class GuardrailChecker:
    def __init__(self, bedrock_client, monitoring: Monitoring, guardrail_id, guardrail_version):
        self.bedrock_client = bedrock_client
        self.monitoring = monitoring

    def apply_guardrail(self, question: str) -> dict:
        """Applies guardrail check before proceeding."""
        content = [{"text": {"text": question}}]
        try:
            response = self.bedrock_client.apply_guardrail(
                guardrailIdentifier=guardrail_id,
                guardrailVersion=guardrail_version,
                source='INPUT',
                content=content
            )
            action = response.get("action", "UNKNOWN")
            self.monitoring.log_metric("Guardrail Interventions", 1 if action == "GUARDRAIL_INTERVENED" else 0)
            return response
        except Exception as e:
            logger.error("Guardrail check failed: %s", e)
            raise

class OpinionQuantifier:
    def __init__(self, bedrock_client, monitoring: Monitoring, model_id):
        self.bedrock_client = bedrock_client
        self.monitoring = monitoring
        self.llm = ChatBedrock(
            client=bedrock_client,
            model_id=model_id,
            model_kwargs={"temperature": 0.0}
        )

    def quantify_opinion(self, question: str, doc_txt: str) -> dict:
        """Uses RAG to analyze discussion sentiment."""
        prompt = f"""Analyze the following discussion on work–life balance:
1. Support vs. opposition (-10 to 10)
2. Overall work–life balance score (1-10)
3. Stress impact (1-10)
4. Emphasis on personal/family time (1-10)

Discussion:
{question}

Reference:
{doc_txt}

Output JSON format with keys: support_level, balance_score, stress_impact, personal_time.
"""
        formatted_prompt = f"""
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
        final = PromptTemplate(
            input_variables=["question", "document"],
            template=formatted_prompt
        )

        retrieval_grader = final | self.llm | JsonOutputParser()
        return retrieval_grader.invoke({"question": question, "document": doc_txt})

