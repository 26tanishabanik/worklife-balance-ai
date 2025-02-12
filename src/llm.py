from src.monitoring import Monitoring
from langchain.prompts import PromptTemplate
from langchain_aws import ChatBedrock
from langchain_core.output_parsers import JsonOutputParser
import logging

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GuardrailChecker:
    def __init__(
        self, bedrock_client, monitoring: Monitoring, guardrail_id, guardrail_version
    ):
        self.bedrock_client = bedrock_client
        self.monitoring = monitoring
        self.guardrail_id = guardrail_id
        self.guardrail_version = guardrail_version

    def apply_guardrail(self, question: str) -> dict:
        """Applies guardrail check before proceeding."""
        content = [{"text": {"text": question}}]
        try:
            response = self.bedrock_client.apply_guardrail(
                guardrailIdentifier=self.guardrail_id,
                guardrailVersion=self.guardrail_version,
                source="INPUT",
                content=content,
            )
            action = response.get("action", "UNKNOWN")
            self.monitoring.log_metric(
                "Guardrail Interventions", 1 if action == "GUARDRAIL_INTERVENED" else 0
            )
            return response
        except Exception as e:
            logger.error("Guardrail check failed: %s", e)
            raise


class OpinionQuantifier:
    def __init__(self, bedrock_client, monitoring: Monitoring, model_id):
        self.bedrock_client = bedrock_client
        self.monitoring = monitoring
        self.llm = ChatBedrock(
            client=bedrock_client, model_id=model_id, model_kwargs={"temperature": 0.0}
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

Output JSON format with keys: support_level,
balance_score, stress_impact, personal_time.
"""
        formatted_prompt = f"""
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
        final = PromptTemplate(
            input_variables=["question", "document"], template=formatted_prompt
        )

        retrieval_grader = final | self.llm | JsonOutputParser()
        return retrieval_grader.invoke({"question": question, "document": doc_txt})
