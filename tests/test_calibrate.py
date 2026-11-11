"""scripts/calibrate.py: rule variants for Round 2 and Round 3, scored against labels (no API)."""
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("calibrate", ROOT / "scripts" / "calibrate.py")
calibrate = importlib.util.module_from_spec(_spec)
sys.modules["calibrate"] = calibrate
_spec.loader.exec_module(calibrate)

import screening  # noqa: E402

REQS = {"must_have": ["Python", "Kafka", "AWS"], "nice_to_have": ["Go", "Terraform"]}
VERDICT_KINDS = [
    {"verdict": "met", "evidence_verified": True},
    {"verdict": "met", "evidence_verified": False},
    {"verdict": "partial"},
    {"verdict": "missing"},
]


def verdicts_for(kinds: list[dict]) -> dict:
    texts = REQS["must_have"] + REQS["nice_to_have"]
    return {screening.requirement_key(t): k for t, k in zip(texts, kinds)}


def test_the_current_rules_reproduce_the_agents_fit_score_and_recommendation():
    for kinds in itertools.product(VERDICT_KINDS, repeat=5):
        verdicts = verdicts_for(list(kinds))

        assert calibrate.fit_with(calibrate.CURRENT, REQS, verdicts) == screening.fit_score(REQS, verdicts)
        assert calibrate.recommend_with(calibrate.CURRENT, REQS, verdicts) == screening.recommendation(REQS, verdicts)[0]


def test_a_looser_hire_rule_lets_one_partial_must_have_through():
    met, partial = VERDICT_KINDS[0], VERDICT_KINDS[2]
    verdicts = verdicts_for([met, met, partial, met, met])

    assert calibrate.recommend_with(calibrate.CURRENT, REQS, verdicts) == "borderline"
    assert calibrate.recommend_with(calibrate.Rules(hire_max_partial=1), REQS, verdicts) == "hire"


def test_a_looser_borderline_rule_keeps_one_missing_must_have_off_no_hire():
    met, missing = VERDICT_KINDS[0], VERDICT_KINDS[3]
    verdicts = verdicts_for([met, met, missing, met, met])

    assert calibrate.recommend_with(calibrate.CURRENT, REQS, verdicts) == "no_hire"
    assert calibrate.recommend_with(calibrate.Rules(borderline_max_missing=1), REQS, verdicts) == "borderline"


def test_a_fit_threshold_rule_decides_on_the_fit_score_alone():
    met, missing = VERDICT_KINDS[0], VERDICT_KINDS[3]
    verdicts = verdicts_for([met, met, missing, met, met])  # fit 75: 6 of 8 weighted points
    rules = calibrate.Rules(hire_at=70, borderline_at=50)

    assert calibrate.fit_with(rules, REQS, verdicts) == 75
    assert calibrate.recommend_with(rules, REQS, verdicts) == "hire"
    assert calibrate.recommend_with(calibrate.Rules(hire_at=80, borderline_at=70), REQS, verdicts) == "borderline"


def test_counting_an_unverified_met_as_met_raises_the_fit_score():
    met, unverified = VERDICT_KINDS[0], VERDICT_KINDS[1]
    verdicts = verdicts_for([met, unverified, met, met, met])

    assert calibrate.fit_with(calibrate.Rules(unverified_met_is_partial=False), REQS, verdicts) == 100
    assert calibrate.fit_with(calibrate.CURRENT, REQS, verdicts) < 100


def test_recommendation_metrics_against_the_labellers_mean_grade():
    met, missing = VERDICT_KINDS[0], VERDICT_KINDS[3]
    by_cid = {"hired_ok": verdicts_for([met] * 5), "hired_wrong": verdicts_for([met] * 5),
              "rejected_wanted": verdicts_for([missing] + [met] * 4)}
    gains = {"hired_ok": 2.0, "hired_wrong": 0.0, "rejected_wanted": 2.0}

    m = calibrate.recommendation_metrics(calibrate.CURRENT, REQS, by_cid, gains)

    assert m["agreement"] == pytest.approx(1 / 3)
    assert m["hire_precision"] == pytest.approx(1 / 2)
    assert m["missed"] == 1  # graded "would interview", called no hire


def test_leave_one_posting_out_picks_the_rule_that_wins_on_the_other_postings():
    current, better, lucky = calibrate.CURRENT, calibrate.Rules(hire_max_partial=1), calibrate.Rules(hire_max_partial=2)
    scores = {current: {"p1": 0.5, "p2": 0.5, "p3": 0.5},
              better: {"p1": 0.8, "p2": 0.7, "p3": 0.6},
              lucky: {"p1": 0.4, "p2": 0.4, "p3": 0.8}}

    held_out = calibrate.leave_one_out(scores, current)

    assert held_out["p3"] == (better, 0.6, 0.5)
    assert held_out["p1"] == (better, 0.8, 0.5)


def test_ties_keep_the_current_rules():
    current, other = calibrate.CURRENT, calibrate.Rules(hire_max_partial=1)
    scores = {other: {"p1": 0.5, "p2": 0.5}, current: {"p1": 0.5, "p2": 0.5}}

    assert calibrate.leave_one_out(scores, current)["p1"][0] == current


def test_the_sweep_refuses_to_run_without_a_verdict_for_every_labelled_candidate(tmp_path):
    (tmp_path / "real_x.json").write_text(json.dumps({"round1_top": [], "verdicts": {"a": verdicts_for(
        [VERDICT_KINDS[0]] * 5)}}))

    with pytest.raises(ValueError, match="calibrate.py judge"):
        calibrate.load_verdicts("real_x", REQS, {"a", "b"}, tmp_path)


def test_judging_caches_the_verdicts_and_the_round1_top(tmp_path, monkeypatch):
    calls = []

    def fake_judge(claude, requirements, resumes, cached, today):
        calls.append(sorted(resumes))
        return {cid: {screening.requirement_key(r): {"verdict": "met"} for r in requirements} for cid in resumes}, {}

    monkeypatch.setattr(calibrate.claude_tasks, "judge_candidates", fake_judge)
    top = [{"candidate_id": "a", "rag_score": 0.9}]

    failures = calibrate.judge_posting("real_x", REQS, {"a": ("A", "text"), "b": ("B", "text")}, top,
                                       claude=None, folder=tmp_path)

    assert failures == {}
    assert calls == [["a", "b"]]
    saved = json.loads((tmp_path / "real_x.json").read_text())
    assert saved["round1_top"] == top
    assert set(calibrate.load_verdicts("real_x", REQS, {"a", "b"}, tmp_path)) == {"a", "b"}


def test_the_sweep_reports_both_rounds_held_out_and_labeller_agreement(tmp_path, monkeypatch):
    met, partial, missing = VERDICT_KINDS[0], VERDICT_KINDS[2], VERDICT_KINDS[3]
    people = {"strong": [met] * 5, "close": [met, met, partial, met, met], "weak": [missing, missing, met, met, met]}
    grades = {"A": {"strong": 2, "close": 2, "weak": 0}, "self": {"strong": 2, "close": 1, "weak": 0}}
    (tmp_path / "labels").mkdir()
    for posting in ("p1", "p2"):
        for who, by_cid in grades.items():
            rows = "".join(f"{cid},{cid.title()},resumes/{cid}.md,{g},\n" for cid, g in by_cid.items())
            (tmp_path / "labels" / f"{posting}_{who}.csv").write_text(
                "candidate_id,name,resume_path,human_grade,notes\n" + rows)
        (tmp_path / f"{posting}.json").write_text(json.dumps(REQS))
        (tmp_path / "verdicts").mkdir(exist_ok=True)
        (tmp_path / "verdicts" / f"{posting}.json").write_text(json.dumps({
            "round1_top": [{"candidate_id": cid, "rag_score": 0.5} for cid in people],
            "verdicts": {cid: verdicts_for(kinds) for cid, kinds in people.items()}}))
    monkeypatch.setattr(calibrate, "VERDICTS_DIR", tmp_path / "verdicts")
    monkeypatch.setattr(calibrate.evaluate, "REQUIREMENTS_DIR", tmp_path)
    out = tmp_path / "calibration.md"

    calibrate.main(["sweep", "--labels", str(tmp_path / "labels"), "--out", str(out)])

    report = out.read_text()
    assert "A vs self: weighted kappa" in report
    assert "**(current)**" in report
    assert report.count("Held out, the rules chosen on the other postings") == 2
    assert "p1: A, self (3 candidates)" in report


def test_a_rule_that_gives_everyone_the_same_call_scores_zero_kappa_however_often_it_agrees():
    partial = VERDICT_KINDS[2]
    by_cid = {cid: verdicts_for([partial] * 5) for cid in ("a", "b", "c", "d")}  # all borderline
    gains = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 0.0}

    m = calibrate.recommendation_metrics(calibrate.CURRENT, REQS, by_cid, gains)

    assert m["agreement"] == pytest.approx(0.75)
    assert m["kappa"] == pytest.approx(0.0)


def test_the_report_warns_when_nobody_was_graded_would_interview():
    lines = calibrate.grade_summary({"p1": {"a": 1.0, "b": 0.0}, "p2": {"c": 0.5}})

    assert "no candidate was graded 2" in " ".join(lines).lower()
    assert "always borderline" in " ".join(lines).lower()
