"""Re-grade saved runs without calling any model.

Use it after fixing a scorer/extractor bug, or after correcting a wrong
expected answer in a benchmark file. Each stored answer is graded again:

  * If the question's prompt (and system prompt, when known) is unchanged in
    the current benchmark file, it is graded against the *current* expected
    answer and scoring type - so fixing an answer key repairs old runs, and
    their leaderboard cells stop being flagged as edited (*).
  * Otherwise it is graded against the expected answer stored in the run, so
    only scorer-code changes apply. A question whose prompt changed is never
    graded against the new key.
  * Inline <think> tags are split again from the raw model text, so parser
    improvements apply too.

Run files are never modified: verdicts go to RESULTS_DIR/rescores/<id>.json
(see storage.py). Delete that folder to get the original scores back.

  python3 rescore.py           # preview what would change
  python3 rescore.py --apply   # write the overlays
"""
import sys

import server
import storage


def _current_benchmarks():
    return {b["id"]: b for b in server.all_benchmarks() if not b.get("error")}


def rescore_record(rec, benches):
    """Grade one upgraded record (without any overlay applied). Returns
    (overlay, n_graded)."""
    summaries = {b["benchmark_id"]: b for b in rec["benchmarks"]}
    entries, matched = [], {}
    for i, c in enumerate(rec["cases"]):
        if c.get("error"):
            continue                        # request failed: nothing to grade
        bid = c["benchmark_id"]
        bench, summ = benches.get(bid), summaries.get(bid, {})
        cur = None
        if bench:
            cur = next((x for x in bench["cases"] if x.get("id") == c["id"]), None)
        same = cur is not None and cur.get("prompt") == c.get("prompt")
        if same and "system_prompt" in summ:       # v2+: system prompt known
            stored_sys = c.get("system_prompt") or summ.get("system_prompt")
            same = stored_sys == (cur.get("system_prompt") or bench.get("system_prompt"))
        matched.setdefault(bid, []).append((c["id"], same))
        if same:
            expected, scoring = cur.get("expected"), cur.get("scoring") or bench["scoring"]
        else:
            expected = c.get("expected")
            scoring = c.get("scoring") or summ.get("scoring")
            if scoring is None:
                continue                  # v1 record, question gone: can't grade
        raw = c.get("raw_output") or c.get("output") or ""
        inline, answer = server.split_reasoning(raw)
        reasoning = c.get("reasoning") or inline
        if not answer.strip() and reasoning:
            ok = False
            detail = ("no answer: still reasoning when generation stopped "
                      f"(finish_reason={c.get('finish_reason')}) - raise max_tokens")
        else:
            ok, detail = server.score_case(scoring, expected, answer, c.get("prompt"))
        entries.append({"i": i, "id": c["id"], "passed": bool(ok), "detail": detail,
                        "expected": expected, "scoring": scoring})

    # The run now counts as graded against the current benchmark version only
    # if it covers exactly the current question set, all with unchanged prompts.
    hashes = {}
    for bid, pairs in matched.items():
        bench = benches.get(bid)
        if (bench and all(same for _, same in pairs)
                and sorted(cid for cid, _ in pairs)
                == sorted(x.get("id", f"case_{k + 1}") for k, x in enumerate(bench["cases"]))):
            hashes[bid] = storage.benchmark_fingerprint(bench)
    overlay = {"schema_version": 1, "run_id": rec["id"], "rescored_at": storage.now_iso(),
               "app_version": storage.app_version(), "cases": entries,
               "benchmark_hashes": hashes}
    return overlay, len(entries)


def rescore_all(results_dir, apply=False):
    """Re-grade every finished run. Returns a report of what changes compared
    with what is shown now (i.e. with any earlier overlay applied)."""
    benches = _current_benchmarks()
    raw_records, unreadable = storage.scan(results_dir, overlay=False)
    shown = {r["id"]: r for r in storage.scan(results_dir)[0]}
    report = {"runs_checked": 0, "runs_changed": 0, "cases_graded": 0, "changes": [],
              "refreshed_benchmarks": 0, "applied": apply, "unreadable": unreadable}
    for rec in raw_records:
        if rec["status"] == "running":
            continue
        overlay, n = rescore_record(rec, benches)
        report["runs_checked"] += 1
        report["cases_graded"] += n
        before = shown.get(rec["id"], rec)
        changed = False
        for e in overlay["cases"]:
            old = before["cases"][e["i"]]
            if bool(old.get("passed")) != e["passed"]:
                changed = True
                report["changes"].append({
                    "run_id": rec["id"], "model": rec["model"],
                    "benchmark_id": old["benchmark_id"], "case_id": e["id"],
                    "before": bool(old.get("passed")), "after": e["passed"],
                    "detail": e["detail"]})
        before_hash = {b["benchmark_id"]: b["benchmark_hash"] for b in before["benchmarks"]}
        refreshed = [bid for bid, h in overlay["benchmark_hashes"].items()
                     if before_hash.get(bid) != h]
        report["refreshed_benchmarks"] += len(refreshed)
        changed = changed or bool(refreshed)
        if changed:
            report["runs_changed"] += 1
        if apply and overlay["cases"] and (changed or storage.load_overlay(results_dir, rec["id"])):
            storage.write_overlay(results_dir, overlay)
    return report


def main(argv):
    apply = "--apply" in argv
    rep = rescore_all(server.RESULTS_DIR, apply=apply)
    print(f"Results: {server.RESULTS_DIR}")
    print(f"Checked {rep['runs_checked']} runs, graded {rep['cases_graded']} answers.")
    for ch in rep["changes"]:
        arrow = "fail -> PASS" if ch["after"] else "pass -> FAIL"
        print(f"  {ch['run_id']}  {ch['model'][:32]:<32} {ch['benchmark_id']}/{ch['case_id']}: "
              f"{arrow}  ({ch['detail']})")
    if rep["refreshed_benchmarks"]:
        print(f"  {rep['refreshed_benchmarks']} benchmark result(s) now match the current "
              "benchmark files (edited-benchmark * flag cleared)")
    if not rep["changes"] and not rep["refreshed_benchmarks"]:
        print("No changes.")
    elif apply:
        print(f"Applied: overlays written for {rep['runs_changed']} run(s).")
    else:
        print("Preview only - run with --apply to save.")


if __name__ == "__main__":
    main(sys.argv[1:])
