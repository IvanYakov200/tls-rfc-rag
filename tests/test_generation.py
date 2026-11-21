import json
import unittest
from unittest.mock import patch

from tls_rag.corpus import Passage
from tls_rag.generation import answer


class GenerationTest(unittest.TestCase):
    def generate(self, *responses, multiple=False):
        raw = {1: b"One. Two."}
        hits = [Passage(1, "1", "Example", 0, 9, "One. Two.")]
        with patch("tls_rag.generation.structured_completion", side_effect=responses) as api:
            row = answer("Question?", hits, raw, multiple=multiple, transport=api)
        return row, [call.args for call in api.call_args_list]

    def response(self, reason, content):
        return {"done_reason": reason, "message": {"content": content},
                "prompt_eval_count": 80, "eval_count": 15}

    def test_retry_then_success(self):
        reply = json.dumps({"claims": [{"text": "Two", "evidence": 2}]})
        row, calls = self.generate(self.response("length", "{"), self.response("stop", reply))
        self.assertEqual([c[2] for c in calls], [600, 900])
        self.assertEqual(calls[0][0], calls[1][0])
        props = calls[0][1]["properties"]["claims"]["items"]["properties"]
        self.assertEqual(props["evidence"]["enum"], [1, 2])
        self.assertEqual(row["attempts"][0]["prompt_eval_count"], 80)
        self.assertEqual(row["attempts"][1]["eval_count"], 15)
        self.assertEqual(row["claims"][0]["quote"], "Two.")

    def test_two_truncations_are_technical_failure(self):
        cut = self.response("length", '{"claims":[]}')
        row, calls = self.generate(cut, cut)
        self.assertEqual((row["status"], len(calls)), ("generation_failed", 2))

    def test_only_valid_empty_claims_mean_abstention(self):
        for output, status in [('{"claims":[]}', "insufficient"), ("{", "generation_failed"),
                               ('{"claims":[{"text":"x","evidence":0}]}', "generation_failed")]:
            with self.subTest(output=output):
                row, calls = self.generate(self.response("stop", output))
                self.assertEqual((row["status"], len(calls)), (status, 1))

    def test_transport_error(self):
        row, calls = self.generate(TimeoutError("timeout"))
        self.assertEqual((row["status"], len(calls)), ("generation_failed", 1))
        self.assertIsNone(row["attempts"][0]["done_reason"])

    def test_multi_citation_references(self):
        for ids in ([2, 1], [], [2, 2], [True], [3], 1):
            with self.subTest(ids=ids):
                reply = json.dumps({"claims": [{"text": "Two and one", "evidence_ids": ids}]})
                row, _ = self.generate(self.response("stop", reply), multiple=True)
                if ids == [2, 1]:
                    self.assertEqual([c["quote"] for c in row["claims"][0]["citations"]], ["Two.", "One."])
                else:
                    self.assertEqual((row["status"], row["claims"]), ("generation_failed", []))


class SelectionFailureTest(unittest.TestCase):
    def test_batch_records_failure_then_processes_next_question(self):
        import io
        import sys
        from contextlib import redirect_stdout
        from unittest.mock import Mock
        from tls_rag import cli

        passages = [Passage(i, "1", "Example", 0, 4, "One.") for i in range(5)]
        selected = {"done_reason": "stop", "message": {"content": '{"ids":[1,2,3,4,5]}'}}
        invalid = {"done_reason": "stop", "message": {"content": '{"ids":[1,1,2,3,4]}'}}
        batch = [{"id": str(i), "question": "Question?"} for i in range(2)]
        retrieval = Mock()
        retrieval.Search.return_value.rank.return_value = {"B": list(range(5))}
        for method, failure in [("final", TimeoutError("timeout")), ("reranked", invalid), ("multi", invalid)]:
            output = io.StringIO()
            with (self.subTest(method=method), redirect_stdout(output),
                  patch.dict(sys.modules, {"tls_rag.retrieval": retrieval}),
                  patch.object(sys, "argv", ["rag.py", "run", "--method", method]),
                  patch("tls_rag.cli.load", return_value=({}, passages)), patch("tls_rag.cli.questions", return_value=batch),
                  patch("tls_rag.llm_client.api_key"), patch("tls_rag.generation.structured_completion", side_effect=[failure, selected]),
                  patch("tls_rag.pipeline.answer", return_value={"status": "insufficient", "claims": []}) as answer_call,
                  self.assertRaises(SystemExit) as exit_status):
                cli.main()
            first, second = map(json.loads, output.getvalue().splitlines())
            self.assertEqual(exit_status.exception.code, 1)
            self.assertEqual((first["id"], first["status"], first["error_stage"]), ("0", "generation_failed", "selection"))
            self.assertTrue(first["error"])
            self.assertEqual(first["pool"], [p.key for p in passages])
            self.assertEqual((first["retrieved"], first["claims"], first["attempts"]), ([], [], []))
            self.assertEqual((second["id"], second["status"], second["retrieved"]), ("1", "insufficient", first["pool"]))
            answer_call.assert_called_once()
            if method == "multi":
                from tls_rag.generation import generation_prompt
                self.assertEqual(answer_call.call_args.kwargs,
                                 {"multiple": True, "system": generation_prompt(True, False)})
