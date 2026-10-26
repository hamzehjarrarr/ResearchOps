"""Report assembly and final checks.

Every factual sentence in the report is either a verified finding or a
writer sentence that passed its own verification, and keeps its source
citations. Sections generated from run statistics (evidence gaps,
evidence quality) are clearly separated and never treated as claims.
"""

import re
from collections import Counter, defaultdict

from researchops.audit import extract_claims
from researchops.grounding import (
    INDEPENDENT_TYPES,
    SOURCE_LABELS,
    STOPWORDS,
    lacks_independent_source,
)
from researchops.themes import theme_order


def cite(finding):
    citations = "".join(
        f"【{source_id}】"
        for source_id in finding["source_ids"]
    )

    return (
        finding["statement"].strip()
        + " "
        + citations
    )


def paragraph(items):
    """
    Combine individually attributed sentences.

    Each sentence retains its own citations, so
    verification can extract it independently.
    """
    return " ".join(
        cite(item)
        for item in items
    )


def paragraphs(items, max_per_paragraph=3):
    if not items:
        return ""

    blocks = []

    for start in range(
        0,
        len(items),
        max_per_paragraph,
    ):
        group = items[
            start:start + max_per_paragraph
        ]

        blocks.append(
            paragraph(group)
        )

    return "\n\n".join(blocks)


def is_vendor_finding(item):
    return item.get("source_type") == "commercial"


def is_independent_finding(item):
    return item.get("source_type") in INDEPENDENT_TYPES


def finding_rank(item):
    """Lower sorts first: independent, relevant, well-scored sources."""
    return (
        not is_independent_finding(item),
        is_vendor_finding(item),
        -int(item.get("relevance") or 3),
        -float(item.get("source_score") or 0),
    )


def grouped_findings(findings):
    groups = defaultdict(list)

    for item in findings:
        groups[item["category"]].append(
            item
        )

    for items in groups.values():
        items.sort(key=finding_rank)

    return groups


def render_theme_sections(groups, order):
    sections = []

    for theme in order:
        items = groups.get(theme, [])

        if not items:
            continue

        sections.append(
            f"### {theme}\n\n"
            + paragraphs(items)
        )

    return "\n\n".join(sections)


def source_links(sources, cited_ids=None):
    lines = []

    for source in sources:
        notes = []

        label = SOURCE_LABELS.get(
            source.get("source_type")
        )

        if label:
            notes.append(label)

        if (
            cited_ids is not None
            and source["id"] not in cited_ids
        ):
            notes.append("not cited")

        suffix = (
            " — " + " · ".join(notes)
            if notes
            else ""
        )

        lines.append(
            f"- [{source['id']}] "
            f"[{source['title']}]({source['url']})"
            f"{suffix}"
        )

    return "\n".join(lines)


def evidence_quality_notes(findings, sources):
    """
    Plain-language notes on how strong the evidence base is.

    These sit under the source links, so they are never
    extracted as factual claims.
    """

    types = {
        source["id"]: source.get("source_type")
        for source in sources
    }

    cited = {
        source_id
        for finding in findings
        for source_id in finding["source_ids"]
    }

    by_type = Counter(types.get(source_id) for source_id in cited)

    breakdown = ", ".join(
        f"{label} {by_type[key]}"
        for key, label in SOURCE_LABELS.items()
        if by_type.get(key)
    )

    unverified = sum(
        lacks_independent_source(finding["source_ids"], types)
        for finding in findings
    )

    return "\n".join([
        "### Evidence quality",
        "",
        f"- Sources retrieved: {len(sources)}; "
        f"cited in this report: {len(cited)}.",
        f"- Cited sources by type: {breakdown or 'not classified'}.",
        f"- Findings without a recognised independent source "
        f"(vendor or unrecognised sites only): {unverified} of "
        f"{len(findings)} (marked ⚠ in the dashboard).",
        "- Every finding was grounded in a verbatim quote from its "
        "source, then reviewed independently in a second model pass. "
        "Synthesised sentences were verified against the findings "
        "they cite.",
        "- Verification covers retrieved passages, not full "
        "documents. Treat vendor-sourced adoption claims as "
        "indicative, not established.",
    ])


def select_summary(findings, limit=4):
    """
    The best finding from each theme, best first.

    "Best" means independent sources before vendor ones, then the
    extractor's relevance rating, then the search relevance score.
    No new factual claims are introduced.
    """

    best = {}

    for item in findings:
        key = item["category"]

        if (
            key not in best
            or finding_rank(item) < finding_rank(best[key])
        ):
            best[key] = item

    return sorted(
        best.values(),
        key=finding_rank,
    )[:limit]


def word_stems(text):
    """First five letters of each word of four or more letters."""
    return {
        word[:5]
        for word in re.findall(r"[a-z]{4,}", str(text).lower())
    }


def subject_coverage(subject, findings):
    """How many findings mention the question's subject.

    Returns (count, total), or None when the subject has no word long
    enough to match on (for example just "AI").
    """
    stems = {
        word[:5]
        for word in re.findall(r"[a-z]{4,}", str(subject).lower())
        if word not in STOPWORDS
    }

    if not stems or not findings:
        return None

    about = sum(
        stems <= word_stems(item["statement"])
        for item in findings
    )

    return about, len(findings)


def evidence_gaps(
    themes,
    findings,
    sources,
    rejected_count,
    followup_rounds,
    subject=None,
):
    """
    What the evidence does not show, worked out from the run itself.

    Deterministic: no model writes this section, so it cannot overstate
    or invent anything. It is never extracted as a claim.
    """
    counts = Counter(item["category"] for item in findings)
    missing = [theme for theme in themes or [] if not counts.get(theme)]
    thin = [theme for theme in themes or [] if counts.get(theme) == 1]

    types = {source["id"]: source.get("source_type") for source in sources}
    cited = {i for item in findings for i in item["source_ids"]}
    cited_types = {types.get(i) for i in cited}

    background = sum((item.get("relevance") or 3) <= 3 for item in findings)

    lines = []

    # The most important gap: evidence about related topics rather
    # than the thing the question asks about.
    coverage = subject_coverage(subject, findings)

    if coverage and coverage[0] < coverage[1] / 2:
        about, total = coverage
        lines.append(
            f"- Only {about} of {total} verified findings are specifically "
            f"about {subject}; the rest describe related topics more "
            f"broadly, so direct evidence on {subject} is limited."
        )

    if missing:
        lines.append(
            "- No verified evidence was found for: "
            + "; ".join(missing)
            + (
                f", even after {followup_rounds} follow-up search round"
                f"{'' if followup_rounds == 1 else 's'}."
                if followup_rounds
                else "."
            )
        )

    if thin:
        lines.append(
            "- Only one verified finding supports: " + "; ".join(thin) + "."
        )

    if findings and background > len(findings) / 2:
        lines.append(
            f"- {background} of {len(findings)} findings are background "
            "rather than direct answers to the question: the sources "
            "mostly discuss related topics."
        )

    if cited and not cited_types & {"official", "academic"}:
        lines.append(
            "- None of the cited sources is a regulator, central bank "
            "or academic publication."
        )

    if rejected_count:
        lines.append(
            f"- {rejected_count} extracted finding"
            f"{' was' if rejected_count == 1 else 's were'} rejected by "
            "the verifier and left out (see the Evidence tab)."
        )

    return lines


def render_sentences(sentences):
    """Writer sentences, each followed by its source citations."""
    return " ".join(
        sentence["text"].strip()
        + " "
        + "".join(f"【{source_id}】" for source_id in sentence["source_ids"])
        for sentence in sentences
    )


def assemble_report(
    question,
    findings,
    sources,
    themes=None,
    written=None,
    gaps=None,
):
    """
    Build the Markdown report.

    With ``written`` (verified writer output), the report opens with a
    direct answer and each theme starts with a synthesised paragraph.
    Findings the writer did not use are listed after it, so no verified
    evidence is lost. Without it, the report lists the verified findings
    under each theme, as before.
    """

    written = written or {}
    answer = written.get("answer") or []
    sections = written.get("sections") or {}
    used = set(written.get("used_findings") or [])

    groups = grouped_findings(findings)
    one_line_question = " ".join(str(question).split())

    parts = [
        "# Research Report",
        f"**Question:** {one_line_question}",
    ]

    if answer:
        parts.append(f"## Answer\n\n{render_sentences(answer)}")
    else:
        summary = paragraphs(
            select_summary(findings, limit=4),
            max_per_paragraph=2,
        )

        if summary:
            parts.append(f"## Executive Summary\n\n{summary}")

    blocks = []

    for theme in theme_order(findings, themes):
        items = groups.get(theme, [])

        if not items:
            continue

        text = []

        if sections.get(theme):
            text.append(render_sentences(sections[theme]))

        remaining = [
            item
            for item in items
            if id_of(item) not in used
        ]

        if remaining:
            text.append(paragraphs(remaining))

        blocks.append(f"### {theme}\n\n" + "\n\n".join(text))

    if blocks:
        parts.append("## Key Findings\n\n" + "\n\n".join(blocks))

    if gaps:
        parts.append("## Gaps in the evidence\n\n" + "\n".join(gaps))

    cited_ids = {
        source_id
        for finding in findings
        for source_id in finding["source_ids"]
    }

    parts.append(
        "## Source links\n\n"
        f"{source_links(sources, cited_ids)}\n\n"
        f"{evidence_quality_notes(findings, sources)}"
    )

    return "\n\n".join(parts)


def id_of(finding):
    """A stable key for a finding: its statement and sources."""
    return (
        re.sub(r"\s+", " ", finding["statement"]).strip(),
        tuple(sorted(finding["source_ids"])),
    )


def findings_review_report(findings):
    """
    One bullet per finding.

    Returns (markdown, line_map) where line_map maps each bullet's
    1-based line number to the index of its finding. This keeps the
    mapping exact even if the claim splitter behaves unexpectedly.
    """

    lines = ["## Findings", ""]
    line_map = {}

    for index, item in enumerate(findings):
        lines.append(f"- {cite(item)}")
        line_map[len(lines)] = index

    return "\n".join(lines), line_map


def verify_assembled_report(report, reviews):
    """
    Match every factual sentence of the final report to a statement
    that already passed independent review.

    ``reviews`` holds the passing reviews of verified findings and of
    verified writer sentences. Exact statement AND exact source IDs
    must match; anything else is reported as a review_error, never
    silently approved.
    """

    passing = {}

    for review in reviews:
        if review["status"] not in ("supported", "research_limitation"):
            continue

        key = (
            re.sub(r"\s+", " ", review["statement"]).strip(),
            tuple(sorted(review["source_ids"])),
        )

        passing[key] = review

    results = []

    for claim in extract_claims(report):
        key = (
            re.sub(r"\s+", " ", claim["statement"]).strip(),
            tuple(sorted(claim["source_ids"])),
        )

        review = passing.get(key)

        if review is not None:
            status = review["status"]
            explanation = (
                review.get("explanation")
                or "Passed independent evidence review."
            )
        else:
            status = "review_error"
            explanation = (
                "This statement or its citations do not match an "
                "independently verified finding or sentence."
            )

        results.append({
            "claim_id": claim["id"],
            "statement": claim["statement"],
            "status": status,
            "source_ids": claim["source_ids"],
            "explanation": explanation,
            "quote": review.get("quote", "") if review else "",
            "line": claim["line"],
            "section": claim["section"],
            "section_type": claim["section_type"],
        })

    supported = sum(item["status"] == "supported" for item in results)
    limitation_claims = sum(
        item["status"] == "research_limitation" for item in results
    )
    errors = len(results) - supported - limitation_claims
    total = len(results)

    units = []

    for item in results:
        passed = item["status"] != "review_error"

        units.append({
            "unit_id": item["claim_id"],
            "line": item["line"],
            "text": item["statement"],
            "status": "passed" if passed else item["status"],
            "passed": passed,
            "claims": [item],
            "issues": [] if passed else [item["explanation"]],
        })

    cited = sum(bool(item["source_ids"]) for item in results)

    return {
        "passed": total > 0 and errors == 0,
        "coverage_complete": total > 0 and errors == 0,
        "coverage_percent": (
            round(100 * (supported + limitation_claims) / total, 2)
            if total
            else 0.0
        ),
        "total_units": total,
        "reviewed_units": supported + limitation_claims,
        "total_claims": total,
        "flagged_claims": errors,
        "claims": results,
        "issues": [
            item["explanation"]
            for item in results
            if item["status"] == "review_error"
        ],
        "units": units,
        "status_counts": {
            "supported": supported,
            "research_limitation": limitation_claims,
            "insufficient_evidence": 0,
            "conflicting": 0,
            "review_error": errors,
        },
        "cited_claims": cited,
        "uncited_claims": total - cited,
        "citation_coverage_percent": (
            round(100 * cited / total, 2) if total else 0.0
        ),
        "limitations": [
            "Every report sentence is a verified finding or a writer "
            "sentence verified against the findings it cites.",
            "Sentences the verifier did not support were removed.",
            "Verification uses retrieved passages rather than full "
            "documents.",
            "The evidence gaps section is generated from run "
            "statistics, not written by the model.",
            "Automated verification does not guarantee factual "
            "accuracy.",
        ],
    }