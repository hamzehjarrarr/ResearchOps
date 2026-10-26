"""Shared research state and helpers used by every stage."""

import time
from typing import TypedDict

from researchops.llm import invoke_model


class State(TypedDict, total=False):
    question: str
    # The specific thing the question is about, e.g. "AI agents".
    subject: str
    queries: list[str]
    themes: list[str]

    # Every source retrieved so far, each with a stable id.
    sources: list[dict]
    # Sources the extractor has already read.
    read_source_ids: list[int]

    # Findings extracted in this round, waiting for verification.
    findings: list[dict]
    # Findings that passed verification, across all rounds.
    approved: list[dict]
    evidence: list[dict]
    rejected: list[dict]
    # Verification results for every approved finding's statement.
    reviews: list[dict]

    followup_rounds: int
    followup_queries: list[str]

    report: str
    verification: dict
    feedback: str
    passed: bool
    events: list[dict]


def record(state, agent, start, detail):
    return (state.get("events") or []) + [{
        "agent": agent,
        "duration_ms": round(
            (time.perf_counter() - start) * 1000
        ),
        "detail": detail,
    }]


def structured_model(schema, system, user, tokens=1400):
    return invoke_model(
        system,
        user,
        schema=schema,
        output_tokens=tokens,
    )