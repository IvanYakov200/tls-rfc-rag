import unittest
from unittest.mock import Mock

from tls_rag.local_chat import Chat


class LocalChatTests(unittest.TestCase):
    def test_examples_and_validation_do_not_call_generator(self):
        generate = Mock()
        chat = Chat(generate)
        self.assertEqual({row["status"] for row in chat.examples}, {"answered", "insufficient"})
        for example in chat.examples:
            for claim in example["claims"]:
                self.assertNotRegex(claim["text"], r"(?i)\b(must|should|may|handshake|alert)\b")
        quotes = [ref["quote"] for c in chat.examples[0]["claims"] for ref in c["citations"]]
        self.assertTrue(any("MUST" in quote for quote in quotes))
        self.assertEqual(chat.ask("  ")[0], 400)
        generate.assert_not_called()

    def test_repeated_live_answers_expose_only_public_fields(self):
        generate = Mock(return_value={"question": "q", "claims": [], "status": "insufficient",
                                      "raw_output": "internal"})
        chat = Chat(generate)
        status, result = chat.ask(" q ")
        self.assertEqual(status, 200)
        self.assertEqual(result["status"], "insufficient")
        self.assertNotIn("raw_output", result)
        for _ in range(5):
            self.assertEqual(chat.ask("q")[0], 200)
        self.assertEqual(generate.call_count, 6)

    def test_failed_attempt_unlocks(self):
        chat = Chat(Mock(return_value={"status": "generation_failed"}))
        self.assertEqual(chat.ask("q")[0], 502)
        self.assertEqual(chat.ask("q")[0], 502)
