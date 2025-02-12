import unittest
from unittest.mock import MagicMock
from llm import GuardrailChecker, OpinionQuantifier
from monitoring import Monitoring


class TestGuardrailChecker(unittest.TestCase):
    def setUp(self):
        self.mock_bedrock_client = MagicMock()
        self.monitoring = Monitoring()
        self.guardrail_checker = GuardrailChecker(
            self.mock_bedrock_client, self.monitoring
        )

    def test_guardrail_no_intervention(self):
        self.mock_bedrock_client.apply_guardrail.return_value = {
            "action": "NO_INTERVENTION"
        }
        response = self.guardrail_checker.apply_guardrail("Test question")
        self.assertEqual(response["action"], "NO_INTERVENTION")

    def test_guardrail_intervention(self):
        self.mock_bedrock_client.apply_guardrail.return_value = {
            "action": "GUARDRAIL_INTERVENED",
            "outputs": [{"text": "Intervention message."}],
        }
        response = self.guardrail_checker.apply_guardrail("Test question")
        self.assertEqual(response["action"], "GUARDRAIL_INTERVENED")
        self.assertEqual(response["outputs"][0]["text"], "Intervention message.")


class TestOpinionQuantifier(unittest.TestCase):
    def setUp(self):
        self.mock_bedrock_client = MagicMock()
        self.monitoring = Monitoring()
        self.opinion_quantifier = OpinionQuantifier(
            self.mock_bedrock_client, self.monitoring
        )

    def test_quantify_opinion(self):
        fake_response = {
            "body": MagicMock(
                read=MagicMock(
                    return_value=b'{"support_level": 5, "balance_score": 7, "stress_impact": 3, "personal_time": 6}'
                )
            )
        }
        self.mock_bedrock_client.invoke_model.return_value = fake_response

        response = self.opinion_quantifier.quantify_opinion(
            "How is work-life balance?", "Reference document text"
        )
        self.assertEqual(response["support_level"], 5)
        self.assertEqual(response["balance_score"], 7)


if __name__ == "__main__":
    unittest.main()
