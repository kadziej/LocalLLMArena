"""Durability tests: old run files must stay readable forever, writes must be
atomic, crashes must not lose finished cases, and editing a benchmark must be
visible on the leaderboard instead of silently mixing incomparable scores.

tests/fixtures/result_*.json are real run files in every shape the app has
ever written. NEVER edit or delete them - when the record format changes, add
a new fixture and teach storage.upgrade_record the old shape.

Run with:  python3 -m unittest discover -s tests -v
"""
import glob
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server
import storage

FIXTURES = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "fixtures", "result_*.json")))

BENCH = {
    "id": "toy", "name": "Toy", "category": "Test", "scoring": "exact_match",
    "default_params": {}, "_file": "toy.json",
    "cases": [{"id": "a", "prompt": "say yes", "expected": "yes"},
              {"id": "b", "prompt": "say no", "expected": "no"},
              {"id": "c", "prompt": "say maybe", "expected": "maybe"}],
}


def fake_completion(answers):
    """chat_completion stand-in: replies from `answers` keyed by user prompt;
    an Exception value is raised (simulates the model server being down)."""
    def _fake(provider, model, messages, params, should_cancel=None, on_token=None):
        val = answers[messages[-1]["content"]]
        if isinstance(val, BaseException):
            raise val
        return {"content": val, "answer": val, "reasoning": "", "reasoning_tokens": None,
                "reasoning_tokens_exact": False, "latency_s": 1.0, "ttft_s": 0.1,
                "tokens": 5, "prompt_tokens": 7, "total_tokens": 12,
                "tokens_per_second": 5.0, "cost_usd": None, "finish_reason": "stop",
                "cancelled": False}
    return _fake


class TmpResults(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="arena-test-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        p = mock.patch.object(server, "RESULTS_DIR", self.dir)
        p.start()
        self.addCleanup(p.stop)

    def run_toy(self, answers, bench=BENCH, model="m1"):
        with mock.patch.object(server, "chat_completion", fake_completion(answers)), \
             mock.patch.object(server, "get_benchmark", lambda bid: bench), \
             mock.patch.object(server, "all_benchmarks", lambda: [bench]):
            run = server.new_run([{"provider": "local", "model": model}], [bench["id"]], {})
            server.run_worker(run)
        return run

    def leaderboard(self, bench=BENCH):
        with mock.patch.object(server, "all_benchmarks", lambda: [bench]):
            return server.build_leaderboard()

    def files(self):
        return sorted(f for f in os.listdir(self.dir) if f.endswith(".json"))


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def install_fixtures(dest):
    """Copy fixtures in under their real on-disk name (<id>.json)."""
    for path in FIXTURES:
        shutil.copy(path, os.path.join(dest, read_json(path)["id"] + ".json"))


GOOD = {"say yes": "yes", "say no": "no", "say maybe": "maybe"}


class TestLegacyRecordsStayReadable(TmpResults):
    def test_there_are_fixtures(self):
        self.assertGreaterEqual(len(FIXTURES), 2)

    def test_every_historic_shape_loads_through_the_full_read_path(self):
        install_fixtures(self.dir)
        before = {f: read_bytes(os.path.join(self.dir, f)) for f in self.files()}

        listing = server.list_results()
        self.assertEqual(listing["unreadable"], [])
        self.assertEqual(len(listing["results"]), len(FIXTURES))
        lb = server.build_leaderboard()
        self.assertEqual(lb["storage"]["runs"], len(FIXTURES))
        self.assertTrue(lb["rows"])
        for item in listing["results"]:
            r = server.get_result(item["id"])
            self.assertEqual(r["status"], "complete")
            self.assertIsInstance(r["overall_score"], float)
            for c in r["cases"]:
                for key in ("id", "benchmark_id", "passed", "output", "reasoning", "raw_output"):
                    self.assertIn(key, c)
            json.dumps(r)

        # reading must never rewrite ("migrate") a stored file
        after = {f: read_bytes(os.path.join(self.dir, f)) for f in self.files()}
        self.assertEqual(before, after)

    def test_fixture_scores_are_preserved_exactly(self):
        for path in FIXTURES:
            raw = read_json(path)
            up = storage.upgrade_record(raw)
            self.assertEqual(up["overall_score"], raw["overall_score"], path)
            self.assertEqual(up["passed"], raw["passed"], path)
            self.assertEqual([b["score"] for b in up["benchmarks"]],
                             [b["score"] for b in raw["benchmarks"]], path)

    def test_minimal_record_gets_defaults_instead_of_crashing(self):
        up = storage.upgrade_record({"model": "x", "benchmarks": [
            {"benchmark_name": "Old", "passed": 3, "total": 4}]}, fallback_id="legacy")
        self.assertEqual(up["id"], "legacy")
        self.assertEqual(up["overall_score"], 75.0)
        self.assertEqual(up["benchmarks"][0]["benchmark_id"], "Old")

    def test_unknown_future_fields_are_kept(self):
        up = storage.upgrade_record({"id": "x", "model": "m", "from_the_future": 1,
                                     "benchmarks": [], "cases": []})
        self.assertEqual(up["from_the_future"], 1)


class TestDamagedFiles(TmpResults):
    def test_corrupt_file_is_reported_not_dropped_or_deleted(self):
        shutil.copy(FIXTURES[0], self.dir)
        bad = os.path.join(self.dir, "20260101-000000-broken.json")
        with open(bad, "w") as f:
            f.write('{"id": "trunc')
        lb = server.build_leaderboard()
        self.assertEqual(lb["storage"]["runs"], 1)
        self.assertEqual([u["file"] for u in lb["storage"]["unreadable"]],
                         ["20260101-000000-broken.json"])
        self.assertTrue(os.path.isfile(bad))

    def test_result_id_cannot_escape_the_results_dir(self):
        for rid in ("../config", "..", "a/b", ""):
            self.assertIsNone(storage.result_path(self.dir, rid))
            self.assertIsNone(server.get_result(rid))


class TestAtomicWrite(TmpResults):
    def test_failed_write_keeps_the_previous_file_intact(self):
        path = os.path.join(self.dir, "r.json")
        storage.atomic_write_json(path, {"v": 1})
        with self.assertRaises(TypeError):
            storage.atomic_write_json(path, {"v": object()})   # not serialisable
        self.assertEqual(read_json(path), {"v": 1})
        self.assertEqual(os.listdir(self.dir), ["r.json"])       # no temp litter


class TestRunRecords(TmpResults):
    def test_run_writes_a_self_describing_record(self):
        self.run_toy(GOOD)
        (fn,) = self.files()
        raw = read_json(os.path.join(self.dir, fn))
        self.assertEqual(raw["schema_version"], storage.SCHEMA_VERSION)
        self.assertEqual(raw["status"], "complete")
        self.assertEqual((raw["passed"], raw["total"], raw["overall_score"]), (3, 3, 100.0))
        b = raw["benchmarks"][0]
        self.assertEqual(b["benchmark_hash"], storage.benchmark_fingerprint(BENCH))
        self.assertEqual(b["scoring"], "exact_match")
        self.assertIn("max_tokens", b["params"])
        for c in raw["cases"]:
            # enough to re-score without the benchmark file
            for key in ("prompt", "expected", "scoring", "output", "benchmark_id"):
                self.assertIn(key, c)
            self.assertEqual(server.score_case(c["scoring"], c["expected"], c["output"],
                                               c["prompt"])[0], c["passed"])

    def test_crash_mid_run_keeps_finished_cases(self):
        answers = dict(GOOD)
        answers["say maybe"] = KeyboardInterrupt()      # process dies on case 3
        with self.assertRaises(KeyboardInterrupt):
            self.run_toy(answers)
        (fn,) = self.files()
        r = server.get_result(fn[:-5])
        self.assertEqual(r["status"], "interrupted")
        self.assertEqual([c["id"] for c in r["cases"]], ["a", "b"])
        self.assertEqual((r["passed"], r["answered"], r["total"]), (2, 2, 3))

    def test_summary_can_be_recomputed_from_cases(self):
        self.run_toy({"say yes": "yes", "say no": "wrong", "say maybe": "maybe"})
        r = server.get_result(self.files()[0][:-5])
        again = storage.summarize_cases(r["cases"])
        for k, v in again.items():
            self.assertEqual(r["benchmarks"][0][k], v, k)


class TestLeaderboardIntegrity(TmpResults):
    def cell(self, bench=BENCH):
        return self.leaderboard(bench)["rows"][0]["benches"]["toy"]

    def test_fresh_result_is_not_stale(self):
        self.run_toy(GOOD)
        c = self.cell()
        self.assertEqual((c["score"], c["stale"], c["partial"]), (100.0, False, False))

    def test_editing_a_case_marks_old_results_stale(self):
        self.run_toy(GOOD)
        edited = json.loads(json.dumps(BENCH))
        edited["cases"][0]["expected"] = "yep"
        self.assertTrue(self.cell(edited)["stale"])

    def test_renaming_a_benchmark_does_not(self):
        self.run_toy(GOOD)
        renamed = dict(BENCH, name="Toy v2", description="new words", category="Other")
        self.assertFalse(self.cell(renamed)["stale"])

    def test_deleted_benchmark_keeps_its_column(self):
        self.run_toy(GOOD)
        with mock.patch.object(server, "all_benchmarks", lambda: []):
            lb = server.build_leaderboard()
        self.assertEqual([(c["id"], c["removed"]) for c in lb["columns"]], [("toy", True)])
        self.assertEqual(lb["rows"][0]["benches"]["toy"]["score"], 100.0)

    def test_failed_rerun_does_not_hide_a_good_result(self):
        self.run_toy(GOOD)
        down = {k: ConnectionError("LM Studio not running") for k in GOOD}
        self.run_toy(down)
        c = self.cell()
        self.assertEqual((c["score"], c["partial"]), (100.0, False))
        self.assertEqual(self.leaderboard()["rows"][0]["run_count"], 2)

    def test_newer_clean_run_replaces_older_one(self):
        self.run_toy(GOOD)
        self.run_toy({"say yes": "yes", "say no": "x", "say maybe": "x"})
        self.assertEqual(self.cell()["score"], 33.3)

    def test_errored_run_alone_is_shown_but_flagged(self):
        self.run_toy({k: ConnectionError("down") for k in GOOD})
        c = self.cell()
        self.assertEqual((c["score"], c["partial"], c["errors"]), (0.0, True, 3))


class TestBackup(TmpResults):
    def test_export_contains_every_run_file_byte_for_byte(self):
        install_fixtures(self.dir)
        z = zipfile.ZipFile(io.BytesIO(storage.export_zip(self.dir)))
        self.assertEqual(sorted(z.namelist()), self.files())
        for fn in self.files():
            self.assertEqual(z.read(fn), read_bytes(os.path.join(self.dir, fn)))


class TestResultsDirConfig(unittest.TestCase):
    def test_default_env_and_setting(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ARENA_RESULTS_DIR", None)
            self.assertEqual(server.resolve_results_dir({}),
                             os.path.join(server.ROOT, "results"))
            self.assertEqual(server.resolve_results_dir({"results_dir": "/data/arena"}),
                             "/data/arena")
            os.environ["ARENA_RESULTS_DIR"] = "/elsewhere"
            self.assertEqual(server.resolve_results_dir({"results_dir": "/data/arena"}),
                             "/elsewhere")


if __name__ == "__main__":
    unittest.main()
