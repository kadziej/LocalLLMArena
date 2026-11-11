"""Token budget for reasoning models, and how generation speed is measured.

Run with:  python3 -m unittest discover -s tests -v
"""
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server
import storage

BENCH = {"id": "b", "name": "B", "scoring": "exact_match", "_file": "b.json",
         "default_params": {"temperature": 0, "max_tokens": 300},
         "cases": [{"id": "1", "prompt": "p1", "expected": "yes"},
                   {"id": "2", "prompt": "p2", "expected": "yes"}]}


class TestEffectiveParams(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(server, "CONFIG", {"default_params": {"max_tokens": 1024},
                                                 "reasoning_budget": 4096})
        p.start()
        self.addCleanup(p.stop)

    def test_benchmark_default_wins_over_config(self):
        self.assertEqual(server.effective_params(BENCH, {})["max_tokens"], 300)

    def test_blank_ui_values_mean_default(self):
        got = server.effective_params(BENCH, {"max_tokens": None, "reasoning_budget": ""})
        self.assertEqual((got["max_tokens"], got["reasoning_budget"]), (300, 4096))

    def test_explicit_run_value_wins(self):
        got = server.effective_params(BENCH, {"max_tokens": 64, "reasoning_budget": 0})
        self.assertEqual((got["max_tokens"], got["reasoning_budget"]), (64, 0))


def responder(script):
    """Fake chat_completion. script(prompt, max_tokens) -> (reasoning, answer,
    finish_reason). Records every call's max_tokens."""
    calls = []

    def _fake(provider, model, messages, params, should_cancel=None, on_token=None):
        prompt = messages[-1]["content"]
        calls.append((prompt, params["max_tokens"]))
        reasoning, answer, fin = script(prompt, params["max_tokens"])
        return {"content": answer, "answer": answer, "reasoning": reasoning,
                "reasoning_tokens": len(reasoning) or None, "reasoning_tokens_exact": True,
                "latency_s": 2.0, "ttft_s": 0.5, "decode_s": 1.5, "tokens": 30,
                "prompt_tokens": 10, "total_tokens": 40, "tokens_per_second": 20.0,
                "cost_usd": 0.0, "finish_reason": fin, "cancelled": False}
    return _fake, calls


def thinker(prompt, max_tokens):
    # needs >1000 tokens of thinking before it can answer
    if max_tokens < 1000:
        return ("hmm " * 50, "", "length")
    return ("hmm " * 50, "yes", "stop")


class TestThinkingBudget(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="arena-budget-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self._patches = [mock.patch.object(server, "RESULTS_DIR", self.dir),
                         mock.patch.object(server, "_REASONING_MODELS", None),
                         mock.patch.object(server, "CONFIG", {"default_params": {"max_tokens": 1024},
                                                              "reasoning_budget": 2000})]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

    def run_bench(self, script, model="m", params=None):
        fake, calls = responder(script)
        events = []
        with mock.patch.object(server, "chat_completion", fake):
            summary, cases, _ = server.run_benchmark_stream(
                "local", model, BENCH, params or {}, events.append)
        return summary, cases, calls, events

    def test_unknown_thinker_is_retried_once_then_gets_the_allowance(self):
        summary, cases, calls, events = self.run_bench(thinker)
        # case 1: 300 -> truncated -> retried at 2300; case 2: straight to 2300
        self.assertEqual(calls, [("p1", 300), ("p1", 2300), ("p2", 2300)])
        self.assertEqual(summary["passed"], 2)
        self.assertTrue(cases[0]["retried_with_thinking_budget"])
        self.assertNotIn("retried_with_thinking_budget", cases[1])
        self.assertEqual([c["max_tokens"] for c in cases], [2300, 2300])
        self.assertEqual(sum(e["type"] == "case_retry" for e in events), 1)

    def test_non_thinking_model_keeps_the_answer_budget(self):
        _, _, calls, _ = self.run_bench(lambda p, m: ("", "yes", "stop"))
        self.assertEqual(calls, [("p1", 300), ("p2", 300)])

    def test_thinker_that_finished_is_not_retried(self):
        _, _, calls, _ = self.run_bench(lambda p, m: ("brief thought", "yes", "stop"))
        self.assertEqual(calls, [("p1", 300), ("p2", 2300)])

    def test_zero_budget_disables_allowance_and_retry(self):
        summary, _, calls, _ = self.run_bench(thinker, params={"reasoning_budget": 0})
        self.assertEqual(calls, [("p1", 300), ("p2", 300)])
        self.assertEqual(summary["passed"], 0)

    def test_thinker_known_from_saved_runs_gets_allowance_from_case_one(self):
        rec = storage.build_record("20260101-000000-abcdef", "2026-01-01T00:00:00+00:00",
                                   "local", "m", {}, [], [{"id": "x", "benchmark_id": "b",
                                   "reasoning": "thinking...", "passed": True}], "complete")
        storage.write_record(self.dir, rec)
        _, _, calls, _ = self.run_bench(thinker)
        self.assertEqual(calls, [("p1", 2300), ("p2", 2300)])


class TestSpeed(unittest.TestCase):
    def test_decode_rate_excludes_time_to_first_token(self):
        self.assertEqual(storage.decode_tps(100, 2.0), 50.0)

    def test_too_short_to_measure_is_none_not_misleading(self):
        # the real case that motivated this: 2 tokens after a 29 s model load
        self.assertIsNone(storage.decode_tps(2, 0.001))
        self.assertIsNone(storage.decode_tps(50, 0.05))
        self.assertIsNone(storage.decode_tps(None, 1.0))

    def test_benchmark_speed_is_token_weighted_and_skips_unmeasurable(self):
        cases = [{"tokens": 2, "decode_s": 0.001},           # ignored
                 {"tokens": 100, "decode_s": 2.0},
                 {"tokens": 300, "decode_s": 6.0}]
        self.assertEqual(storage.speed_of(cases), 50.0)

    def test_old_records_derive_decode_time_from_latency_and_ttft(self):
        old = [{"tokens": 2, "latency_s": 28.95, "ttft_s": 28.949},
               {"tokens": 60, "latency_s": 3.5, "ttft_s": 0.5}]
        self.assertEqual(storage.speed_of(old), 20.0)

    def test_upgrade_recomputes_stored_speed(self):
        rec = {"id": "x", "model": "m", "avg_tokens_per_second": 0.1,
               "benchmarks": [{"benchmark_id": "b", "avg_tokens_per_second": 0.1,
                               "passed": 1, "total": 1}],
               "cases": [{"id": "1", "benchmark_id": "b", "tokens": 60,
                          "latency_s": 3.5, "ttft_s": 0.5}]}
        up = storage.upgrade_record(rec)
        self.assertEqual((up["avg_tokens_per_second"],
                          up["benchmarks"][0]["avg_tokens_per_second"]), (20.0, 20.0))


if __name__ == "__main__":
    unittest.main()
