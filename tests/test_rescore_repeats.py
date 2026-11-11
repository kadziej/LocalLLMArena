"""Re-scoring saved runs (overlays, never edits) and repeat runs (pooled
mean, spread and flaky questions on the leaderboard).

Run with:  python3 -m unittest discover -s tests -v
"""
import copy
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rescore
import server
import storage

BENCH = {
    "id": "toy", "name": "Toy", "category": "Test", "scoring": "exact_match",
    "system_prompt": "Be brief.", "default_params": {}, "_file": "toy.json",
    "cases": [{"id": "a", "prompt": "capital of France?", "expected": "Paris"},
              {"id": "b", "prompt": "2+2?", "expected": "4"},
              {"id": "c", "prompt": "color of sky?", "expected": "blue"}],
}


def fake_completion(answers):
    """answers: prompt -> reply, or a list of replies consumed one per call."""
    def _fake(provider, model, messages, params, should_cancel=None, on_token=None):
        val = answers[messages[-1]["content"]]
        if isinstance(val, list):
            val = val.pop(0)
        if isinstance(val, BaseException):
            raise val
        inline, ans = server.split_reasoning(val)
        return {"content": val, "answer": ans, "reasoning": inline, "reasoning_tokens": None,
                "reasoning_tokens_exact": False, "latency_s": 1.0, "ttft_s": 0.1,
                "decode_s": 0.9, "tokens": 9, "prompt_tokens": 7, "total_tokens": 16,
                "tokens_per_second": 10.0, "cost_usd": None, "finish_reason": "stop",
                "cancelled": False}
    return _fake


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="arena-rescore-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.bench = copy.deepcopy(BENCH)
        for p in (mock.patch.object(server, "RESULTS_DIR", self.dir),
                  mock.patch.object(server, "_REASONING_MODELS", set()),
                  mock.patch.object(server, "get_benchmark", lambda bid: self.bench),
                  mock.patch.object(server, "all_benchmarks", lambda: [self.bench])):
            p.start()
            self.addCleanup(p.stop)

    def run_toy(self, answers, repeats=1, model="m1"):
        targets = [{"provider": "local", "model": model}]
        if repeats > 1:
            targets = [dict(targets[0], group=0, repeat=k + 1, repeats=repeats)
                       for k in range(repeats)]
        with mock.patch.object(server, "chat_completion", fake_completion(answers)):
            run = server.new_run(targets, ["toy"], {})
            server.run_worker(run)

    def files(self):
        return sorted(f for f in os.listdir(self.dir) if f.endswith(".json"))

    def snapshot(self):
        out = {}
        for f in self.files():
            with open(os.path.join(self.dir, f), "rb") as fh:
                out[f] = fh.read()
        return out

    def cell(self):
        return server.build_leaderboard()["rows"][0]["benches"]["toy"]


ANSWERS = {"capital of France?": "Paris", "2+2?": "4", "color of sky?": "Blue."}


class TestRescore(Base):
    def test_fixing_a_wrong_answer_key_repairs_old_runs_without_editing_them(self):
        self.bench["cases"][1]["expected"] = "5"          # buggy key at run time
        self.run_toy(ANSWERS)
        self.assertEqual(self.cell()["score"], 33.3)      # "Blue." also fails exact
        before = self.snapshot()

        self.bench["cases"][1]["expected"] = "4"          # key fixed later
        self.assertTrue(self.cell()["stale"])
        preview = rescore.rescore_all(self.dir, apply=False)
        self.assertEqual([(c["case_id"], c["after"]) for c in preview["changes"]], [("b", True)])
        self.assertFalse(os.path.isdir(os.path.join(self.dir, "rescores")))  # preview writes nothing

        rescore.rescore_all(self.dir, apply=True)
        c = self.cell()
        self.assertEqual((c["score"], c["stale"], c["rescored"]), (66.7, False, True))
        self.assertEqual(self.snapshot(), before)         # run files untouched
        case_b = server.get_result(self.files()[0][:-5])["cases"][1]
        self.assertEqual((case_b["passed"], case_b["passed_recorded"]), (True, False))

        # deleting the overlay restores the original verdicts exactly
        shutil.rmtree(os.path.join(self.dir, "rescores"))
        self.assertEqual(self.cell()["score"], 33.3)

    def test_changed_prompt_is_not_graded_against_the_new_key(self):
        self.run_toy(ANSWERS)
        self.bench["cases"][0] = {"id": "a", "prompt": "capital of Spain?", "expected": "Madrid"}
        rep = rescore.rescore_all(self.dir, apply=True)
        self.assertEqual(rep["changes"], [])
        c = self.cell()
        self.assertEqual(c["score"], 66.7)
        self.assertTrue(c["stale"])                       # still a different version

    def test_scorer_improvement_applies_via_stored_expected(self):
        self.run_toy(ANSWERS)
        real = server.score_case

        def lenient(scoring, expected, output, prompt=None):
            if scoring == "exact_match":
                return output.strip(" .").lower() == str(expected).lower(), "lenient"
            return real(scoring, expected, output, prompt)
        with mock.patch.object(server, "score_case", lenient):
            rep = rescore.rescore_all(self.dir, apply=True)
        self.assertEqual([(c["case_id"], c["after"]) for c in rep["changes"]], [("c", True)])
        self.assertEqual(self.cell()["score"], 100.0)

    def test_think_tags_are_resplit_from_raw_output(self):
        self.run_toy(dict(ANSWERS, **{"2+2?": "<think>maybe 5</think>4"}))
        rep = rescore.rescore_all(self.dir, apply=False)
        self.assertEqual(rep["changes"], [])              # stays correct: only "4" graded

    def test_rescore_is_idempotent(self):
        self.bench["cases"][1]["expected"] = "5"
        self.run_toy(ANSWERS)
        self.bench["cases"][1]["expected"] = "4"
        rescore.rescore_all(self.dir, apply=True)
        again = rescore.rescore_all(self.dir, apply=False)
        self.assertEqual((again["changes"], again["refreshed_benchmarks"]), ([], 0))

    def test_errored_cases_are_left_alone(self):
        self.run_toy(dict(ANSWERS, **{"2+2?": ConnectionError("down")}))
        rep = rescore.rescore_all(self.dir, apply=False)
        self.assertEqual(rep["cases_graded"], 2)

    def test_mismatched_overlay_is_ignored(self):
        self.run_toy(ANSWERS)
        rid = self.files()[0][:-5]
        storage.write_overlay(self.dir, {"run_id": rid, "cases": [
            {"i": 0, "id": "not-a", "passed": False}]})
        self.assertEqual(self.cell()["score"], 66.7)

    def test_backup_includes_overlays(self):
        self.bench["cases"][1]["expected"] = "5"
        self.run_toy(ANSWERS)
        self.bench["cases"][1]["expected"] = "4"
        rescore.rescore_all(self.dir, apply=True)
        import io, zipfile
        names = zipfile.ZipFile(io.BytesIO(storage.export_zip(self.dir))).namelist()
        self.assertIn("rescores/" + self.files()[0], names)


class TestRepeats(Base):
    def test_repeats_are_pooled_with_spread_and_flaky_questions(self):
        answers = {"capital of France?": "Paris",
                   "2+2?": ["4", "5", "4"],                 # flaky
                   "color of sky?": "blue"}
        self.run_toy(answers, repeats=3)
        self.assertEqual(len(self.files()), 3)
        groups = {server.get_result(f[:-5])["repeat_group"] for f in self.files()}
        self.assertEqual(len(groups), 1)
        c = self.cell()
        self.assertEqual((c["runs"], c["score"], c["min"], c["max"]), (3, 88.9, 66.7, 100.0))
        self.assertEqual(c["std"], 19.2)
        self.assertEqual(c["flaky"], ["b"])
        self.assertEqual(len(c["result_ids"]), 3)
        row = server.build_leaderboard()["rows"][0]
        self.assertEqual((row["overall_score"], row["run_count"]), (88.9, 3))

    def test_single_runs_are_not_pooled_with_each_other(self):
        self.run_toy(ANSWERS)                                 # 66.7
        self.run_toy(dict(ANSWERS, **{"color of sky?": "blue"}))  # 100, newer
        c = self.cell()
        self.assertEqual((c["runs"], c["score"], c["std"]), (1, 100.0, None))

    def test_errored_repeat_is_left_out_of_the_pool(self):
        answers = {"capital of France?": "Paris",
                   "2+2?": ["4", ConnectionError("down"), "4"],
                   "color of sky?": "blue"}
        self.run_toy(answers, repeats=3)
        c = self.cell()
        self.assertEqual((c["runs"], c["score"], c["partial"]), (2, 100.0, False))

    def test_api_expands_repeats_per_model(self):
        captured = {}

        def fake_start(targets, bench_ids, params):
            captured.update(targets=targets, params=params)
            raise RuntimeError("stop here")
        h = server.Handler.__new__(server.Handler)
        h._read_body = lambda: {"targets": [{"provider": "local", "model": "x"},
                                            {"provider": "local", "model": "y"}],
                                "benchmark_ids": ["toy"], "params": {"repeats": 2}}
        with mock.patch.object(server, "start_run", fake_start):
            with self.assertRaises(RuntimeError):
                h._handle_run()
        self.assertEqual([(t["model"], t["repeat"], t["group"]) for t in captured["targets"]],
                         [("x", 1, 0), ("x", 2, 0), ("y", 1, 1), ("y", 2, 1)])
        self.assertNotIn("repeats", captured["params"])     # not a generation param


class TestWilson(unittest.TestCase):
    def test_small_benchmarks_have_wide_intervals(self):
        self.assertGreater(server._wilson(0.75, 12), 20)
        self.assertLess(server._wilson(0.75, 1000), 3)
        self.assertIsNone(server._wilson(0.5, 0))


if __name__ == "__main__":
    unittest.main()
