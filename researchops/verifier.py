"""Verifier: an independent model pass that reviews every finding."""

import time

from researchops.audit import verify_report
from researchops.grounding import lacks_independent_source
from researchops.report import findings_review_report
from researchops.state import State, record

PASSING = ("supported", "research_limitation")


def verifier(state: State):
    """
    Review the findings extracted in this round.

    A finding is approved only if every claim extracted from it passed
    review against its cited source excerpt. Approved findings, their
    reviews and the rejected findings accumulate across rounds.
    """
    start = time.perf_counter()
    findings = state.get("findings", [])

    approved = list(state.get("approved", []))
    evidence = list(state.get("evidence", []))
    rejected_items = list(state.get("rejected", []))
    reviews = list(state.get("reviews", []))

    if findings:
        draft, line_map = findings_review_report(findings)
        review = verify_report(draft, state["sources"])
        claims = review["claims"]
    else:
        line_map, claims = {}, []

    line_ok = {}
    claim_by_line = {}

    for claim in claims:
        line = claim["line"]
        line_ok[line] = line_ok.get(line, True) and claim["status"] in PASSING
        claim_by_line.setdefault(line, claim)

    source_types = {
        source["id"]: source.get("source_type")
        for source in state["sources"]
    }

    passed_now = 0

    for line, index in sorted(line_map.items()):
        finding = findings[index]
        result = claim_by_line.get(line, {})

        entry = {
            "statement": finding["statement"],
            "source_ids": finding["source_ids"],
            "category": finding["category"],
            "quote": finding.get("quote", ""),
            "source_type": finding.get("source_type"),
            "review_status": result.get("status", ""),
            "review_explanation": result.get("explanation", ""),
        }

        if not line_ok.get(line, False):
            rejected_items.append(entry)
            continue

        passed_now += 1
        approved.append(finding)

        evidence.append({
            **entry,
            "unverified_only": lacks_independent_source(
                finding["source_ids"],
                source_types,
            ),
        })

        reviews.append({
            "statement": finding["statement"],
            "source_ids": finding["source_ids"],
            "status": result.get("status", "supported"),
            "explanation": result.get("explanation", ""),
            "quote": finding.get("quote", ""),
        })

    round_label = (
        f"round {state['followup_rounds'] + 1}: "
        if state.get("followup_rounds")
        else ""
    )

    return {
        "approved": approved,
        "evidence": evidence,
        "rejected": rejected_items,
        "reviews": reviews,
        "events": record(
            state,
            "verifier",
            start,
            f"{round_label}{passed_now} of {len(findings)} findings "
            f"supported; {len(approved)} verified in total",
        ),
    }