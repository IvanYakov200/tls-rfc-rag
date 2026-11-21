import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tls_rag.faceted import merge_queries
from tls_rag.pipeline import run


class FacetedTests(unittest.TestCase):
    def test_candidate_budget_and_scope(self):
        passages = [SimpleNamespace(rfc=1 if i < 20 else 2) for i in range(50)]
        search = Mock()
        search.rank.side_effect = lambda q, candidates: {"B": list(reversed(candidates))}
        found = merge_queries(search, passages, list(range(40)), [{"rfc": 2, "text": "rule"}], 40)
        self.assertEqual(found[:20], list(range(20)))
        self.assertEqual(found[20:], list(range(49, 29, -1)))
        self.assertEqual(search.rank.call_args.kwargs["candidates"], list(range(20, 50)))

    def test_failed_rewrite_uses_baseline(self):
        passages = [SimpleNamespace(key=str(i), rfc=1) for i in range(5)]
        search = Mock()
        search.rank.return_value = {"B": list(range(5))}
        with (patch("tls_rag.pipeline.rewrite", side_effect=TimeoutError("timeout")),
              patch("tls_rag.pipeline.select", return_value=(passages, {})),
              patch("tls_rag.pipeline.answer", return_value={"status": "insufficient"})):
            row = run(search, passages, {}, "Question?", "faceted")
        self.assertEqual(row["pool"], row["retrieved"])
        self.assertEqual(row["rewrite_error"], "timeout")

    def test_default_generator_uses_direct_prompt_after_selection(self):
        from tls_rag.generation import DIRECT_PROMPT
        passages = [SimpleNamespace(key=str(i), rfc=1) for i in range(5)]
        search = Mock()
        search.rank.return_value = {"B": list(range(5))}
        with (patch("tls_rag.pipeline.rewrite", return_value=[{"rfc": 1, "text": "rule"}]),
              patch("tls_rag.pipeline.select", return_value=(passages, {})),
              patch("tls_rag.pipeline.answer", return_value={"status": "insufficient"}) as generate):
            row = run(search, passages, {}, "Question?")
        self.assertEqual(row["method"], "direct")
        generate.assert_called_once_with("Question?", passages, {}, multiple=True, system=DIRECT_PROMPT)
