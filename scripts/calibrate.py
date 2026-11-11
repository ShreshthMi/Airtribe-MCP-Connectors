"""
Calibrate Round 2's fit score and Round 3's hire rule against human labels, as the
Milestone 3 review asked: set them from independent labels rather than by hand.

    python scripts/calibrate.py judge                  # once; calls Claude for each labelled candidate
    python scripts/calibrate.py sweep                  # offline; every labeller with a graded sheet
    python scripts/calibrate.py sweep --labellers A B  # just these labellers

judge  For each posting with a graded sheet in evaluation/labels/, judges every candidate on
       its sheets, and Round 1's top 10, against the posting's frozen requirements
       (evaluation/private/requirements/). Caches the verdicts and that top 10 in
       evaluation/private/verdicts/<posting>.json.

sweep  Scores the agent's current rules and a small grid of alternatives against the
       labellers' mean grade, without calling Claude:
         Round 2  how the fit score orders Round 1's top 10: P@5 and nDCG@10, as in evaluate.py
         Round 3  hire / borderline / no hire against the rounded mean grade: weighted kappa
                  (chance-corrected, used to choose), agreement, hire precision, and
                  interviews missed (mean grade 1.5 or more, called no hire)
       With so few labelled candidates, the rule that scores best on all of them is partly
       luck. So each posting is also held out in turn: the rule that does best on the other
       postings is scored on it. Writes evaluation/calibration.md; changes nothing in the agent.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402

import claude_tasks  # noqa: E402
import evaluate  # noqa: E402
import human_eval  # noqa: E402
from screening import CREDIT, MUST_WEIGHT, NICE_WEIGHT, requirement_key  # noqa: E402

VERDICTS_DIR = ROOT / "evaluation" / "private" / "verdicts"
WOULD_INTERVIEW = 1.5  # a mean grade at least this high: the labellers would interview
SHOWN_ROWS = 6


@dataclass(frozen=True)
class Rules:
    """One setting of the scoring rules. The defaults are the agent's own (screening.py)."""
    must_weight: int = MUST_WEIGHT
    partial_credit: float = CREDIT["partial"]
    unverified_met_is_partial: bool = True
    hire_max_partial: int = 0        # must-haves only partly met that a Hire may still have
    borderline_max_missing: int = 0  # must-haves missing that a Borderline may still have
    hire_at: int | None = None       # with borderline_at: decide on the fit score instead
    borderline_at: int | None = None

    def ranking(self) -> str:
        unverified = "partial" if self.unverified_met_is_partial else "met"
        return f"must-haves x{self.must_weight}, partial = {self.partial_credit:g}, unverified met = {unverified}"

    def decision(self) -> str:
        if self.hire_at is not None:
            return f"hire at fit >= {self.hire_at}, borderline at fit >= {self.borderline_at}"
        unverified = "" if self.unverified_met_is_partial else "; unverified met counts as met"
        return (f"hire: none missing, <= {self.hire_max_partial} partial; "
                f"borderline: <= {self.borderline_max_missing} missing{unverified}")


CURRENT = Rules()
RANKING_GRID = [CURRENT] + [r for r in (Rules(must_weight=w, partial_credit=c, unverified_met_is_partial=s)
                                        for w in (1, 2, 3) for c in (0.25, 0.5, 0.75) for s in (True, False))
                            if r != CURRENT]
DECISION_GRID = [CURRENT] + [r for r in (Rules(unverified_met_is_partial=s, hire_max_partial=h, borderline_max_missing=b)
                                         for s in (True, False) for h in (0, 1) for b in (0, 1)) if r != CURRENT] + [
    Rules(hire_at=h, borderline_at=b) for h in (70, 80, 90) for b in (40, 50, 60)]


# --------------------------------------------------------------------------- #
# The rules, parameterised
# --------------------------------------------------------------------------- #
def _verdict(verdict: dict, rules: Rules) -> str:
    if rules.unverified_met_is_partial and verdict["verdict"] == "met" and not verdict.get("evidence_verified", False):
        return "partial"
    return verdict["verdict"]


def _texts(requirements: dict) -> list[str]:
    return requirements.get("must_have", []) + requirements.get("nice_to_have", [])


def fit_with(rules: Rules, requirements: dict, verdicts: dict[str, dict]) -> int:
    """screening.fit_score with these weights: weighted share of requirements met, 0-100."""
    credit = {**CREDIT, "partial": rules.partial_credit}
    weighted = ([(r, rules.must_weight) for r in requirements.get("must_have", [])]
                + [(r, NICE_WEIGHT) for r in requirements.get("nice_to_have", [])])
    total = sum(weight for _, weight in weighted)
    if not total:
        return 0
    earned = sum(weight * credit[_verdict(verdicts[requirement_key(r)], rules)] for r, weight in weighted)
    return round(100 * earned / total)


def recommend_with(rules: Rules, requirements: dict, verdicts: dict[str, dict]) -> str:
    """screening.recommendation's call (hire | borderline | no_hire) under these rules."""
    if rules.hire_at is not None:
        fit = fit_with(rules, requirements, verdicts)
        return "hire" if fit >= rules.hire_at else "borderline" if fit >= rules.borderline_at else "no_hire"
    musts = requirements.get("must_have", [])
    if not musts:
        return "borderline"
    status = [_verdict(verdicts[requirement_key(r)], rules) for r in musts]
    missing, partial = status.count("missing"), status.count("partial")
    if missing == 0 and partial <= rules.hire_max_partial:
        return "hire"
    return "borderline" if missing <= rules.borderline_max_missing else "no_hire"


# --------------------------------------------------------------------------- #
# Metrics, and choosing on held-out postings
# --------------------------------------------------------------------------- #
def ranking_metrics(rules: Rules, requirements: dict, top: list[dict], verdicts_by_cid: dict,
                    gains: dict[str, float]) -> dict:
    """Round 2: Round 1's top 10 re-ordered by this fit score (Round 1's score breaks ties)."""
    order = sorted(top, key=lambda c: (-fit_with(rules, requirements, verdicts_by_cid[c["candidate_id"]]),
                                       -c["rag_score"], c["candidate_id"]))
    ids = [c["candidate_id"] for c in order]
    return {"p5": evaluate.precision_at(ids, gains, 5), "ndcg10": evaluate.ndcg_at(ids, gains, evaluate.TOP)}


def recommendation_metrics(rules: Rules, requirements: dict, verdicts_by_cid: dict, gains: dict[str, float]) -> dict:
    """Round 3 against the labellers' mean grade, over every labelled candidate that was judged."""
    calls = {cid: recommend_with(rules, requirements, verdicts_by_cid[cid]) for cid in gains if cid in verdicts_by_cid}
    hires = [cid for cid, call in calls.items() if call == "hire"]
    agent = [human_eval.AGENT_GRADE[call] for call in calls.values()]
    people = [round(gains[cid]) for cid in calls]
    return {"kappa": human_eval.cohens_kappa(agent, people, weighted=True) if calls else None,
            "agreement": sum(a == p for a, p in zip(agent, people)) / len(calls) if calls else None,
            "hire_precision": sum(gains[c] >= evaluate.RELEVANT_GRADE for c in hires) / len(hires) if hires else None,
            "missed": sum(call == "no_hire" and gains[cid] >= WOULD_INTERVIEW for cid, call in calls.items()),
            "judged": len(calls)}


def leave_one_out(scores: dict[Rules, dict[str, float]], current: Rules) -> dict[str, tuple[Rules, float, float]]:
    """posting -> (rule with the best mean on the other postings, its score here, the current rule's here).

    Ties go to the current rules, so a rule has to do strictly better to be chosen.
    """
    in_order = [current] + [r for r in scores if r != current]
    postings = sorted({p for by_posting in scores.values() for p in by_posting})
    held_out = {}
    for posting in postings:
        def on_the_others(rules: Rules) -> float:
            values = [v for p, v in scores[rules].items() if p != posting and v is not None]
            return sum(values) / len(values) if values else float("-inf")
        chosen = max(in_order, key=on_the_others)  # max keeps the first of equals
        held_out[posting] = (chosen, scores[chosen].get(posting), scores[current].get(posting))
    return held_out


# --------------------------------------------------------------------------- #
# The verdict cache
# --------------------------------------------------------------------------- #
def _cache(posting: str, folder: Path) -> dict:
    path = folder / f"{posting}.json"
    return json.loads(path.read_text()) if path.is_file() else {"round1_top": [], "verdicts": {}}


def load_verdicts(posting: str, requirements: dict, candidate_ids: set[str], folder: Path = VERDICTS_DIR) -> dict:
    """The cached verdicts for these candidates; refuses if any candidate lacks one for a requirement."""
    cached = _cache(posting, folder)["verdicts"]
    keys = {requirement_key(r) for r in _texts(requirements)}
    missing = sorted(cid for cid in candidate_ids if not keys <= set(cached.get(cid, {})))
    if missing:
        raise ValueError(f"{posting}: no verdicts yet for {', '.join(missing)}. Run: python scripts/calibrate.py judge")
    return {cid: cached[cid] for cid in candidate_ids}


def judge_posting(posting: str, requirements: dict, resumes: dict[str, tuple[str, str]], top: list[dict], claude,
                  folder: Path = VERDICTS_DIR, today: date | None = None) -> dict[str, str]:
    """Judge these candidates (id -> (name, text)), cache their verdicts with Round 1's top 10; return failures."""
    cached = _cache(posting, folder)["verdicts"]
    verdicts, failures = claude_tasks.judge_candidates(claude, _texts(requirements), resumes, cached,
                                                       today or date.today())
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{posting}.json").write_text(
        json.dumps({"round1_top": top, "verdicts": verdicts}, indent=2, ensure_ascii=False) + "\n")
    return failures


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def sheet_candidates(labels_dir: Path, postings: list[str]) -> dict[str, dict[str, str]]:
    """posting -> {candidate_id: name} over every sheet for it, graded or not."""
    people: dict[str, dict[str, str]] = {}
    for path in sorted(labels_dir.glob("*.csv")):
        posting = path.stem.rpartition("_")[0]
        if posting in postings:
            people.setdefault(posting, {}).update({r["candidate_id"]: r["name"] for r in human_eval.read_sheet(path)})
    return people


def frozen(posting: str) -> dict:
    path = evaluate.REQUIREMENTS_DIR / f"{posting}.json"
    if not path.is_file():
        raise ValueError(f"No frozen requirements for {posting} in {evaluate._shown(path.parent)}.")
    return json.loads(path.read_text())


def run_judge(args) -> None:
    from llm import Claude
    from matching_agent import search_query
    postings = args.postings or sorted(evaluate.load_labels(args.labels))
    corpus, claude = evaluate.open_corpora([args.provider])[args.provider], Claude()
    try:
        for posting, people in sheet_candidates(args.labels, postings).items():
            reqs, jd = frozen(posting), evaluate.find_posting(posting, args.jobs).read_text(encoding="utf-8")
            found = corpus.search(search_query(jd, reqs), k=evaluate.TOP, must_haves=reqs["must_have"])
            top = [{"candidate_id": c["candidate_id"], "rag_score": c["rag_score"]} for c in found]
            names = {**people, **{c["candidate_id"]: c["name"] for c in found}}
            resumes = {cid: (name, corpus.resume_text(cid)) for cid, name in names.items()}
            failures = judge_posting(posting, reqs, resumes, top, claude)
            print(f"{posting}: {len(resumes) - len(failures)} judged" + (f", failed: {failures}" if failures else ""),
                  flush=True)
    finally:
        corpus.close()
        print(f"(Claude usage: {claude.usage['calls']} calls)")


def score_all(labels: dict) -> tuple[dict, dict, dict]:
    """(ranking scores, decision scores, mean grades per posting) for every rule in both grids."""
    ranking: dict = {r: {} for r in RANKING_GRID}
    decision: dict = {r: {} for r in DECISION_GRID}
    gains_by_posting = {}
    for posting, by_labeller in labels.items():
        reqs, gains = frozen(posting), evaluate.mean_grades(by_labeller)
        top = _cache(posting, VERDICTS_DIR)["round1_top"]
        verdicts = load_verdicts(posting, reqs, set(gains) | {c["candidate_id"] for c in top}, VERDICTS_DIR)
        gains_by_posting[posting] = gains
        for rules in RANKING_GRID:
            ranking[rules][posting] = ranking_metrics(rules, reqs, top, verdicts, gains)
        for rules in DECISION_GRID:
            decision[rules][posting] = recommendation_metrics(rules, reqs, verdicts, gains)
    return ranking, decision, gains_by_posting


def grade_summary(gains_by_posting: dict[str, dict[str, float]]) -> list[str]:
    """How the labels are spread, and the score of always saying borderline: the bar a rule has to clear."""
    grades = Counter(round(g) for gains in gains_by_posting.values() for g in gains.values())
    total = sum(grades.values())
    lines = [f"Rounded mean grades: {grades[2]} x 2 (would interview), {grades[1]} x 1 (borderline), "
             f"{grades[0]} x 0, over {total} candidates. An always borderline rule would agree on "
             f"{grades[1] / total:.2f} of them with kappa 0, so agreement alone rewards saying borderline. "
             "Rules are therefore chosen on quadratic-weighted kappa, which corrects for chance."]
    if not grades[2]:
        lines.append("**No candidate was graded 2, so these labels cannot set the hire cut-off; only the "
                     "borderline / no hire boundary and the ranking can be calibrated.**")
    return lines


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _kappas(labels: dict) -> list[str]:
    """Weighted kappa for each pair of labellers, pooled over the postings both graded."""
    names = sorted({who for by_labeller in labels.values() for who in by_labeller})
    lines = []
    for first, second in combinations(names, 2):
        pairs = [(by[first][cid], by[second][cid]) for by in labels.values() if first in by and second in by
                 for cid in by[first] if cid in by[second]]
        if pairs:
            kappa = human_eval.cohens_kappa([a for a, _ in pairs], [b for _, b in pairs], weighted=True)
            lines.append(f"- {first} vs {second}: weighted kappa {kappa:.2f} ({human_eval.kappa_reading(kappa)}), "
                         f"{len(pairs)} candidates")
    return lines or ["- Only one labeller: agreement between labellers cannot be measured."]


def _table(scores: dict, key: str, describe, extra: list[tuple[str, str]]) -> list[str]:
    postings = sorted({p for by in scores.values() for p in by})
    ranked = sorted(scores, key=lambda r: -(_mean(m[key] for m in scores[r].values()) or 0))
    shown = ranked[:SHOWN_ROWS] + ([CURRENT] if CURRENT not in ranked[:SHOWN_ROWS] else [])
    head = ["Rules", f"mean {key}"] + [label for _, label in extra] + postings
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for rules in shown:
        by = scores[rules]
        cells = [describe(rules) + (" **(current)**" if rules == CURRENT else ""),
                 evaluate._cell(_mean(m[key] for m in by.values()))]
        cells += [evaluate._cell(_mean(m[k] for m in by.values())) if k != "missed"
                  else str(sum(m[k] for m in by.values())) for k, _ in extra]
        lines.append("| " + " | ".join(cells + [evaluate._cell(by[p][key]) for p in postings]) + " |")
    return lines


def _held_out(scores: dict, key: str, describe) -> list[str]:
    held = leave_one_out({r: {p: m[key] for p, m in by.items()} for r, by in scores.items()}, CURRENT)
    lines = ["| Held out | Chosen on the other postings | Its score there | Current rules there |", "|---|---|---|---|"]
    better = worse = 0
    for posting, (chosen, its, current) in held.items():
        lines.append(f"| {posting} | {describe(chosen)} | {evaluate._cell(its)} | {evaluate._cell(current)} |")
        if its is not None and current is not None:
            better, worse = better + (its > current + 1e-9), worse + (its < current - 1e-9)
    tied = len(held) - better - worse
    return lines + ["", f"Held out, the rules chosen on the other postings did better than the current ones on "
                        f"{better} of {len(held)} postings, worse on {worse}, the same on {tied}."]


def render(labels: dict, ranking: dict, decision: dict, gains: dict, today: date) -> str:
    who = "; ".join(f"{p}: {', '.join(sorted(by))} ({len(gains[p])} candidates)" for p, by in labels.items())
    lines = [f"# Calibrating Round 2 and Round 3 against human labels ({today.isoformat()})", "",
             f"Labels: {who}. A candidate's grade is the mean of these labellers' grades.", "",
             "Agreement between labellers:", *_kappas(labels), "", *grade_summary(gains), "",
             "## Round 3: the hire / borderline / no hire rule", "",
             "Kappa: quadratic-weighted, the call (2 / 1 / 0) against the rounded mean grade. Agreement: share "
             "where they are equal. Hire precision: share of Hire calls graded 1 or more. Missed: graded "
             f"{WOULD_INTERVIEW:g} or more, called no hire.", "",
             *_table(decision, "kappa", Rules.decision, [("agreement", "agreement"),
                                                          ("hire_precision", "hire precision"),
                                                          ("missed", "interviews missed")]), "",
             *_held_out(decision, "kappa", Rules.decision), "",
             "## Round 2: the fit score's ranking of Round 1's top 10", "",
             *_table(ranking, "ndcg10", Rules.ranking, [("p5", "P@5")]), "",
             *_held_out(ranking, "ndcg10", Rules.ranking), "",
             "How to read this: a rule earns a change only if it beats the current rules on postings it was not "
             "chosen on. A handful of postings and labellers show a direction, not a significant result."]
    return "\n".join(lines) + "\n"


def run_sweep(args) -> str:
    labels = evaluate.load_labels(args.labels)
    if args.labellers:
        labels = {p: {w: g for w, g in by.items() if w in args.labellers} for p, by in labels.items()}
        labels = {p: by for p, by in labels.items() if by}
    if not labels:
        raise ValueError("No graded sheets from those labellers.")
    ranking, decision, gains = score_all(labels)
    report = render(labels, ranking, decision, gains, date.today())
    args.out.write_text(report)
    return report


def main(argv: list[str] | None = None) -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Calibrate the screening rules against human labels.")
    parser.add_argument("command", choices=["judge", "sweep"])
    parser.add_argument("--labels", type=Path, default=ROOT / "evaluation" / "labels")
    parser.add_argument("--jobs", type=Path, default=ROOT / "job_descriptions")
    parser.add_argument("--provider", default="huggingface")
    parser.add_argument("--postings", nargs="+", default=[], help="judge only these postings")
    parser.add_argument("--labellers", nargs="+", default=[], help="sweep with only these labellers' sheets")
    parser.add_argument("--out", type=Path, default=ROOT / "evaluation" / "calibration.md")
    args = parser.parse_args(argv)
    try:
        if args.command == "judge":
            run_judge(args)
        else:
            print(run_sweep(args))
            print(f"wrote {evaluate._shown(args.out)}")
    except ValueError as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
