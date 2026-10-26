"""Writer: turns verified findings into a report, sentence by verified sentence.

The writer may only combine and compare verified findings. Each sentence
it writes must cite the findings it uses, pass deterministic checks
(numbers and topics must come from those findings), and then pass the
same independent review as the findings themselves. Anything that fails
is dropped, and the findings it would have covered are listed instead.
"""

import re
import time

from pydantic import BaseModel, ConfigDict

from researchops.audit import remove_citations, split_sentences, verify_report
from researchops.config import ENABLE_WRITER
from researchops.grounding import added_question_terms, numbers_grounded
from researchops.report import (
    assemble_report,
    evidence_gaps,
    finding_rank,
    id_of,
    verify_assembled_report,
)
from researchops.state import State, record, structured_model
from researchops.themes import group_findings

# Findings shown to the model when writing the direct answer.
MAX_ANSWER_FINDINGS = 20

# Verbs that only frame the question ("How are X changing Y?"). The
# writer may answer in the same words ("X is changing Y by ..."); that is
# not a new topic. Whole words only, so "transactions" is unaffected by
# "transform". The verifier still reviews every sentence.
FRAMING_WORDS_RE = re.compile(
    r"\b(?:chang|approach|affect|impact|evolv|transform|reshap|shap|"
    r"influenc|alter)(?:e|es|ed|ing|s)?\b",
    re.IGNORECASE,
)


class WrittenSentence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    findings: list[int]


class WriterOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentences: list[WrittenSentence]


RULES = """
Rules:
- Use ONLY the numbered findings. Add no facts, numbers, names, dates
  or opinions that are not in them.
- Every sentence lists the finding numbers it uses in "findings".
- Combine and compare findings where they relate, but keep each
  finding's subject and certainty: a finding about machine learning
  stays about machine learning, and "may" stays "may".
- Do not put citations or finding numbers in the text itself.

Example
Findings:
1. Federal banking regulators plan a request for information on model
   risk management that covers agentic AI.
2. Existing model risk rules from banking agencies cover AI deployment.
Good: "Banking regulators apply existing model risk rules to AI and
plan to gather information on agentic AI specifically." findings [1, 2]
Bad: "Regulators are worried that AI agents are unsafe." (not in the
findings)

Return structured output only.
"""

ANSWER_SYSTEM = (
    """
You write the opening answer of a research report.

Write 2 to 4 sentences that directly answer the research question.
If the findings only partly answer it, say what they do show and keep
every claim as narrow as the findings.
"""
    + RULES
)

SECTION_SYSTEM = (
    """
You write one section of a research report.

Write 2 to 4 sentences for the section named below that help answer
the research question.
"""
    + RULES
)


def numbered(findings):
    return "\n".join(
        f"{number}. {item['statement']}"
        for number, item in enumerate(findings, 1)
    )


def draft_sentences(system, question, findings, heading=None):
    """Ask the model for sentences; return (text, finding list) pairs."""
    user = f"QUESTION:\n{question}\n\n"

    if heading:
        user += f"SECTION:\n{heading}\n\n"

    user += f"FINDINGS:\n{numbered(findings)}"

    try:
        response = structured_model(WriterOutput, system, user, tokens=600)

    except Exception as exc:
        print("Writer error:", type(exc).__name__, str(exc)[:150])
        return []

    drafted = []

    for sentence in response.sentences:
        cited = [
            findings[number - 1]
            for number in dict.fromkeys(sentence.findings)
            if 1 <= number <= len(findings)
        ]

        drafted.append((sentence.text, cited))

    return drafted


def clean_sentence(text):
    """One tidy sentence, or None if it is not exactly one sentence."""
    text = " ".join(remove_citations(str(text)).split())
    text = text.strip(" -•")

    if not text:
        return None

    text = text[0].upper() + text[1:]

    if text[-1] not in ".!?":
        text += "."

    return text if len(split_sentences(text)) == 1 else None


def topic_words(question):
    """The question without its framing verbs, for the drift check."""
    return FRAMING_WORDS_RE.sub(" ", str(question))


def check_failure(text, cited, question):
    """Deterministic checks before the model review.

    Returns why the sentence fails, or None if it passes. The sentence
    must cite findings ("uncited"), every number in it must appear in
    those findings ("number"), and it must not bring in a question
    topic that none of them mention ("drift").
    """
    if not cited:
        return "uncited"

    evidence = " ".join(
        f"{item['statement']} {item.get('quote', '')}" for item in cited
    )

    if not numbers_grounded(text, evidence):
        return "number"

    if added_question_terms(text, evidence, topic_words(question)):
        return "drift"

    return None


def passes_checks(text, cited, question):
    return check_failure(text, cited, question) is None


def source_ids_for(cited):
    return sorted({i for item in cited for i in item["source_ids"]})


def verify_sentences(candidates, findings):
    """
    Review writer sentences against the findings they cite.

    Each finding becomes a pseudo-source whose excerpt is the finding
    and its verbatim quote, so the reviewer judges the sentence only
    against verified evidence.
    """
    if not candidates:
        return []

    index = {id_of(item): number for number, item in enumerate(findings, 1)}

    pseudo_sources = [
        {
            "id": number,
            "title": "Verified finding",
            "snippet": f"{item['statement']} Source quote: \"{item.get('quote', '')}\"",
        }
        for number, item in enumerate(findings, 1)
    ]

    lines = ["## Sentences", ""]
    line_of = {}

    for position, (text, cited) in enumerate(candidates):
        marks = "".join(f"【{index[id_of(item)]}】" for item in cited)
        lines.append(f"- {text} {marks}")
        line_of[len(lines)] = position

    review = verify_report("\n".join(lines), pseudo_sources)

    status = {}

    for claim in review["claims"]:
        position = line_of.get(claim["line"])

        if position is None:
            continue

        ok = claim["status"] == "supported"
        previous = status.get(position, (True, ""))
        status[position] = (previous[0] and ok, claim["explanation"])

    return [
        (text, cited, status.get(position, (False, ""))[1])
        for position, (text, cited) in enumerate(candidates)
        if status.get(position, (False, ""))[0]
    ]


def write(question, findings, themes):
    """Draft, check and verify the answer and the theme paragraphs."""
    ranked = sorted(findings, key=finding_rank)[:MAX_ANSWER_FINDINGS]

    plan = [("answer", draft_sentences(ANSWER_SYSTEM, question, ranked))]

    for theme in themes:
        items = [item for item in findings if item["category"] == theme]

        if len(items) >= 2:
            plan.append(
                (theme, draft_sentences(SECTION_SYSTEM, question, items, theme))
            )

    candidates = []
    placement = []
    drafted = 0
    dropped = {"format": 0, "uncited": 0, "number": 0, "drift": 0}

    for target, sentences in plan:
        for text, cited in sentences:
            drafted += 1
            cleaned = clean_sentence(text)
            reason = (
                check_failure(cleaned, cited, question)
                if cleaned
                else "format"
            )

            if reason:
                dropped[reason] += 1
                print(f"Writer dropped ({reason}): {str(text)[:160]}")
                continue

            candidates.append((cleaned, cited))
            placement.append(target)

    verified = verify_sentences(candidates, findings)
    kept = {(text, tuple(id_of(item) for item in cited)) for text, cited, _ in verified}
    explanations = {text: explanation for text, _, explanation in verified}

    written = {"answer": [], "sections": {}, "used_findings": set()}
    reviews = []

    for (text, cited), target in zip(candidates, placement, strict=True):
        if (text, tuple(id_of(item) for item in cited)) not in kept:
            print(f"Writer dropped (verifier): {text[:160]}")
            continue

        sentence = {"text": text, "source_ids": source_ids_for(cited)}

        if target == "answer":
            written["answer"].append(sentence)
        else:
            written["sections"].setdefault(target, []).append(sentence)
            written["used_findings"].update(id_of(item) for item in cited)

        reviews.append({
            "statement": text,
            "source_ids": sentence["source_ids"],
            "status": "supported",
            "explanation": explanations.get(text)
            or "Verified against the findings it cites.",
            "quote": "",
        })

    stats = {
        "drafted": drafted,
        "checked": len(candidates),
        "verified": len(reviews),
        "dropped": dropped,
    }

    return written, reviews, stats


def writer(state: State):
    start = time.perf_counter()
    findings = state.get("approved", [])

    if not findings:
        raise RuntimeError(
            "No findings passed independent evidence verification. "
            "An unsupported report will not be produced."
        )

    # Gaps are measured against the planner's themes, before regrouping.
    gaps = evidence_gaps(
        state.get("themes"),
        findings,
        state["sources"],
        len(state.get("rejected", [])),
        state.get("followup_rounds", 0),
        subject=state.get("subject"),
    )

    # Group the verified findings into themes that fit what was found.
    findings, grouped = group_findings(state["question"], findings)
    themes = grouped or state.get("themes")

    if grouped:
        categories = {id_of(item): item["category"] for item in findings}

        for item in state.get("evidence", []):
            item["category"] = categories.get(id_of(item), item["category"])

    written, writer_reviews, stats = (
        write(state["question"], findings, themes)
        if ENABLE_WRITER
        else ({}, [], None)
    )

    report = assemble_report(
        state["question"],
        findings,
        state["sources"],
        themes,
        written=written,
        gaps=gaps,
    )

    verification = verify_assembled_report(
        report,
        state.get("reviews", []) + writer_reviews,
    )

    if stats:
        dropped = stats["dropped"]
        detail = (
            f"{stats['drafted']} sentences drafted; {stats['checked']} passed "
            f"deterministic checks (dropped: {dropped['drift']} drift, "
            f"{dropped['number']} number, {dropped['format']} format, "
            f"{dropped['uncited']} uncited); "
            f"{stats['verified']} verified and kept"
        )
    else:
        detail = "writer off: report lists the verified findings"

    return {
        "report": report,
        "themes": themes,
        "verification": verification,
        "passed": verification["passed"],
        "feedback": (
            f"{len(findings)} findings passed independent verification; "
            f"{len(state.get('rejected', []))} rejected. "
            + (
                f"{stats['verified']} synthesised sentences were verified "
                "against the findings they cite."
                if stats
                else "The report reuses verified statements."
            )
        ),
        "events": record(state, "writer", start, detail),
    }