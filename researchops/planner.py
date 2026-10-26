"""Planner: turns the question into search queries and report themes."""

import re
import time

from pydantic import BaseModel, ConfigDict

from researchops.config import MAX_SEARCHES
from researchops.state import State, record, structured_model
from researchops.themes import DEFAULT_THEMES, clean_themes


class SearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    queries: list[str]
    themes: list[str]


PLANNER_SYSTEM = """
You are a rigorous research planning agent.

First name the subject: the specific technology, policy or thing the
question is about, in 1 to 4 words, taken from the question where
possible. Examples: "AI agents", "stablecoins", "large language models".

Produce 2 to 4 distinct search queries covering the
user's question.

Include at least one query aimed at regulators, central
banks, standard setters or academic research.

Prioritize:
- Primary sources and official institutions
- Real-world deployments and documented case studies
- Independent research
- Specific operational and financial risks
- Regulatory and governance considerations

Important:
Distinguish the exact technology asked about from
broader or adjacent technologies.

For example, autonomous AI agents are not necessarily
the same as chatbots, predictive machine learning,
or generative AI tools.

Do not assume evidence for one proves adoption of another.

Also propose 3 to 5 report themes: short section names
(2 to 5 words each) that together organise a report
answering THIS question. Choose themes that fit the
question. Include one about risks, challenges or open
questions when that is relevant. Never use "Executive
Summary", "Key Findings", "References" or "Source links"
as a theme.

Return structured output only.
"""


def clean_subject(value):
    """A short, plain subject phrase, or "" if the model gave none."""
    text = re.sub(r"[^\w\s\-/]", "", " ".join(str(value).split()))
    return text.strip()[:60] if len(text.split()) <= 6 else ""


def planner(state: State):
    start = time.perf_counter()

    themes = []
    subject = ""

    try:
        response = structured_model(
            SearchPlan,
            PLANNER_SYSTEM,
            state["question"],
            tokens=500,
        )

        queries = []
        seen = set()

        for value in response.queries:
            query = str(value).strip()[:180]

            if not query:
                continue

            key = query.lower()

            if key in seen:
                continue

            seen.add(key)
            queries.append(query)

            if len(queries) >= min(MAX_SEARCHES, 4):
                break

        themes = clean_themes(response.themes)
        subject = clean_subject(response.subject)

    except Exception as exc:
        print(
            "Planner fallback:",
            type(exc).__name__,
            str(exc)[:150],
        )
        queries = []

    if not queries:
        queries = [state["question"][:180]]

    if len(themes) < 2:
        themes = list(DEFAULT_THEMES)

    return {
        "subject": subject,
        "queries": queries,
        "themes": themes,
        "events": record(
            state,
            "planner",
            start,
            f"{len(queries)} search queries; "
            f"{len(themes)} report themes; "
            f"subject: {subject or 'not identified'}",
        ),
    }