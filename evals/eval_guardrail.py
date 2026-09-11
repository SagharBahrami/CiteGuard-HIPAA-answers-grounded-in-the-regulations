"""Evaluate the faithfulness guardrail's accuracy against a curated set of
faithful and deliberately-unfaithful answers (evals/cases.py).

Reports two things from one run, since they answer different questions:
- "Direct check": check_faithfulness in isolation -- is the guardrail model
  itself accurate? Standard precision/recall, positive class = UNFAITHFUL
  (what the guardrail exists to detect). This is what a guardrail-prompt or
  guardrail_model change should be judged against.
- "Full pipeline": mirrors qa.answer_question's actual behavior -- if the
  direct check flags an answer, it's regenerated once with the flagged
  claims as feedback and re-checked, exactly like production.

The full-pipeline report deliberately does NOT reuse precision/recall: the
retry can rewrite a fabricated answer into a genuinely accurate one, and
when that happens the case's ground-truth label (written for the *original*
text) no longer describes what the user actually sees. Scoring that as a
"false negative" -- the same label used for a real hallucination slipping
through unflagged -- would conflate two opposite outcomes under one number.
Instead each case is classified into one of five outcomes (see _classify)
so a successful self-correction is counted separately from a dangerous miss.

The retry only ever runs on cases the direct check flagged, same gate as
production (qa.py only retries when check_faithfulness returns
is_faithful=False) -- so this doesn't spend extra API calls on cases that
didn't need it.

Not a pytest test: makes real OpenAI API calls (guardrail + retry
generation) and costs money each run, unlike the mocked unit test suite, and
results can vary slightly run to run since the underlying models aren't
fully deterministic. Run manually:

    python -m evals.eval_guardrail
"""

from evals.cases import CASES
from citedguard.generate import regenerate_answer
from citedguard.guardrails import check_faithfulness
from citedguard.retriever import RetrievedChunk

OUTCOME_LABELS = {
    "passed_clean": "faithful, never flagged -- correct immediately",
    "corrected": "flagged, retry fixed it -- user sees an accurate answer, no warning",
    "false_alarm": "faithful, still flagged after retry -- unnecessary warning shown",
    "caught": "fabricated, still flagged after retry -- warning correctly shown",
    "missed": "fabricated, never flagged at all -- DANGEROUS, nothing caught this",
}


def _classify(expected_faithful: bool, direct_faithful: bool, final_faithful: bool) -> str:
    if expected_faithful:
        if direct_faithful:
            return "passed_clean"
        return "corrected" if final_faithful else "false_alarm"
    else:
        if direct_faithful:
            return "missed"  # never flagged in the first place -- retry never even ran
        return "corrected" if final_faithful else "caught"


def _confusion(pairs: list[tuple[bool, bool]]) -> dict:
    """pairs of (actually_unfaithful, predicted_unfaithful) -> confusion matrix + metrics."""
    tp = fn = fp = tn = 0
    for actually_unfaithful, predicted_unfaithful in pairs:
        if actually_unfaithful and predicted_unfaithful:
            tp += 1
        elif actually_unfaithful and not predicted_unfaithful:
            fn += 1
        elif not actually_unfaithful and predicted_unfaithful:
            fp += 1
        else:
            tn += 1
    total = len(pairs)
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn, "total": total,
        "accuracy": (tp + tn) / total if total else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
    }


def _print_direct_report(m: dict) -> None:
    print("--- Direct check (no retry) -- tests check_faithfulness in isolation ---")
    print(f"Accuracy:  {m['accuracy']:.1%} ({m['tp'] + m['tn']}/{m['total']})")
    print(f"Precision: {m['precision']:.1%} (of answers flagged unfaithful, how many really were)")
    print(f"Recall:    {m['recall']:.1%} (of answers that were really unfaithful, how many got caught)")
    print(f"Confusion: TP={m['tp']} FN={m['fn']} FP={m['fp']} TN={m['tn']}")
    print()


def _print_pipeline_report(outcomes: list[str]) -> None:
    print("--- Full pipeline (matches qa.answer_question: retry once if flagged) ---")
    counts = {label: outcomes.count(label) for label in OUTCOME_LABELS}
    for label, desc in OUTCOME_LABELS.items():
        print(f"  {label:14s} {counts[label]:2d}   ({desc})")

    n_expected_unfaithful = sum(1 for c in CASES if not c.expected_faithful)
    n_expected_faithful = sum(1 for c in CASES if c.expected_faithful)

    print()
    print(f"Safety (a fabrication reaching the user with no warning and no fix):")
    print(f"  missed: {counts['missed']}/{n_expected_unfaithful} fabricated cases -- this is the number that matters most")
    print(f"User friction (an accurate answer getting an unnecessary warning):")
    print(f"  false_alarm: {counts['false_alarm']}/{n_expected_faithful} faithful cases")
    print()


def run() -> None:
    direct_pairs: list[tuple[bool, bool]] = []
    outcomes: list[str] = []

    for case in CASES:
        chunks = [RetrievedChunk(**d) for d in case.chunks]
        result, _ = check_faithfulness(case.answer, chunks)

        actually_unfaithful = not case.expected_faithful
        direct_predicted_unfaithful = not result.is_faithful
        direct_pairs.append((actually_unfaithful, direct_predicted_unfaithful))

        retry_result = None
        final_faithful = result.is_faithful
        if direct_predicted_unfaithful:
            corrected, _ = regenerate_answer(case.query, chunks, case.answer, result.unsupported_claims)
            retry_result, _ = check_faithfulness(corrected, chunks)
            final_faithful = retry_result.is_faithful

        outcome = _classify(case.expected_faithful, result.is_faithful, final_faithful)
        outcomes.append(outcome)

        status = "OK  " if direct_predicted_unfaithful == actually_unfaithful else "MISS"
        line = (
            f"[{status}] {case.name:32s} expected_faithful={case.expected_faithful!s:5s} "
            f"direct_faithful={result.is_faithful} outcome={outcome}"
        )
        if retry_result is not None:
            line += f" (after_retry_faithful={retry_result.is_faithful})"
        print(line)
        if direct_predicted_unfaithful:
            print(f"         direct_unsupported_claims={result.unsupported_claims}")
            if retry_result is not None:
                print(f"         after_retry_unsupported_claims={retry_result.unsupported_claims}")

    print()
    _print_direct_report(_confusion(direct_pairs))
    _print_pipeline_report(outcomes)


if __name__ == "__main__":
    run()
