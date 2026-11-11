"""Live-run event buffer: token events are dropped once their question is
answered, so memory stays flat over long queues, while streams (including a
page re-attaching mid-run) still receive every durable event in order.

Run with:  python3 -m unittest discover -s tests -v
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server

N_CASES, N_TOKENS = 40, 200
BENCH = {"id": "big", "name": "Big", "scoring": "exact_match", "_file": "big.json",
         "default_params": {},
         "cases": [{"id": f"q{i}", "prompt": f"p{i}", "expected": "ok"} for i in range(N_CASES)]}


def streaming_fake(gate=None):
    """chat_completion stand-in that streams N_TOKENS tokens per answer. If
    `gate` is given, it pauses mid-answer on the first case until released."""
    def _fake(provider, model, messages, params, should_cancel=None, on_token=None):
        for n in range(N_TOKENS):
            on_token("x", n + 1, "content")
            if gate and n == N_TOKENS // 2 and not gate["passed"]:
                gate["passed"] = True
                gate["reached"].set()
                gate["release"].wait(5)
        return {"content": "ok", "answer": "ok", "reasoning": "", "reasoning_tokens": None,
                "reasoning_tokens_exact": False, "latency_s": 1.0, "ttft_s": 0.1,
                "decode_s": 0.9, "tokens": N_TOKENS, "prompt_tokens": 5, "total_tokens": 205,
                "tokens_per_second": 222.2, "cost_usd": None, "finish_reason": "stop",
                "cancelled": False}
    return _fake


class TestRunBuffer(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="arena-buf-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        for p in (mock.patch.object(server, "RESULTS_DIR", self.dir),
                  mock.patch.object(server, "_REASONING_MODELS", set()),
                  mock.patch.object(server, "get_benchmark", lambda bid: BENCH),
                  mock.patch.object(server, "all_benchmarks", lambda: [BENCH])):
            p.start()
            self.addCleanup(p.stop)

    def new_run(self, models=1):
        return server.new_run([{"provider": "local", "model": f"m{k}"} for k in range(models)],
                              ["big"], {})

    def test_finished_run_buffers_no_tokens(self):
        run = self.new_run(models=2)
        with mock.patch.object(server, "chat_completion", streaming_fake()):
            server.run_worker(run)
        types = [e["type"] for e in run["events"]]
        self.assertNotIn("token", types)
        self.assertEqual(types.count("case"), 2 * N_CASES)
        emitted = run["next_seq"]
        self.assertGreater(emitted, 2 * N_CASES * N_TOKENS)       # tokens were streamed...
        self.assertLess(len(run["events"]), emitted / 50)         # ...but not kept
        self.assertEqual(run["seqs"], sorted(run["seqs"]))

    def test_buffer_memory_stays_flat(self):
        run = self.new_run(models=2)
        with mock.patch.object(server, "chat_completion", streaming_fake()):
            server.run_worker(run)
        kept = len(json.dumps(run["events"]))
        # every event that was emitted, as the old buffer would have held it
        per_token = len(json.dumps({"type": "token", "benchmark_id": "big", "index": 0,
                                    "piece": "x", "n": 1, "kind": "content"}))
        self.assertLess(kept, 0.1 * (kept + 2 * N_CASES * N_TOKENS * per_token))

    def test_stream_from_start_after_finish_replays_durable_events_in_order(self):
        run = self.new_run()
        with mock.patch.object(server, "chat_completion", streaming_fake()):
            server.run_worker(run)
        got = [e for batch in server.iter_run_events(run, 0) for e in batch]
        self.assertEqual(got, run["events"])
        self.assertEqual(got[0]["type"], "run_started")
        self.assertEqual(got[-1]["type"], "queue_done")

    def test_reattach_mid_question_gets_that_questions_tokens(self):
        run = self.new_run()
        gate = {"passed": False, "reached": threading.Event(), "release": threading.Event()}
        with mock.patch.object(server, "chat_completion", streaming_fake(gate)):
            t = threading.Thread(target=server.run_worker, args=(run,))
            t.start()
            self.assertTrue(gate["reached"].wait(5))
            # a page refresh lands now: it must see the current partial answer
            with run["cond"]:
                snapshot = list(run["events"])
            tokens = [e for e in snapshot if e["type"] == "token"]
            self.assertEqual(len(tokens), N_TOKENS // 2 + 1)
            self.assertTrue(all(e["index"] == 0 for e in tokens))
            gate["release"].set()
            t.join(10)
        self.assertFalse(t.is_alive())

    def test_live_reader_receives_every_case_result_once_in_order(self):
        run = self.new_run()
        received = []

        def reader():
            for batch in server.iter_run_events(run, 0, wait=0.05):
                received.extend(batch)
        r = threading.Thread(target=reader)
        r.start()
        with mock.patch.object(server, "chat_completion", streaming_fake()):
            server.run_worker(run)
        r.join(10)
        self.assertFalse(r.is_alive())
        cases = [e["result"]["id"] for e in received if e["type"] == "case"]
        self.assertEqual(cases, [f"q{i}" for i in range(N_CASES)])
        self.assertEqual(received[-1]["type"], "queue_done")

    def test_resume_from_sequence_number(self):
        run = self.new_run()
        with mock.patch.object(server, "chat_completion", streaming_fake()):
            server.run_worker(run)
        mid = run["seqs"][len(run["seqs"]) // 2]
        got = [e for batch in server.iter_run_events(run, mid) for e in batch]
        self.assertEqual(got, run["events"][len(run["seqs"]) // 2:])


if __name__ == "__main__":
    unittest.main()
