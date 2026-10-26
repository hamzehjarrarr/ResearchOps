"""Researcher and follow-up: web search, filtering and source selection."""

import os
import time
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict

from researchops.config import (
    EXCERPT_CHARS,
    MAX_FOLLOWUP_ROUNDS,
    MAX_SOURCES,
    MIN_AUTHORITATIVE_SCORE,
    MIN_FINDINGS_PER_THEME,
    MIN_SOURCE_SCORE,
    SEARCH_DEPTH,
    USE_PAGE_TEXT,
)
from researchops.grounding import (
    AUTHORITATIVE_DOMAINS,
    INDEPENDENT_TYPES,
    TYPE_RANK,
    classify_source,
    is_on_topic,
    normalize_title,
    select_passages,
)
from researchops.state import State, record, structured_model
from researchops.themes import OTHER_THEME

# Sources added by one follow-up round.
FOLLOWUP_SOURCES = 6


# ============================================================
# Search
# ============================================================

def canonical_url(url):
    parsed = urlparse(url)

    return (
        parsed.netloc.lower().removeprefix("www."),
        parsed.path.rstrip("/"),
    )


def source_rank(source):
    """Prefer official and academic sources over vendor content."""
    source_type = (
        source.get("source_type")
        or classify_source(source["url"])
    )

    return TYPE_RANK.get(source_type, 3)


def evidence_text(item, question):
    """The evidence kept for one search result.

    With page text available, the passages most related to the question
    are kept; otherwise Tavily's own extract of the page is used.
    """
    content = str(item.get("content", "")).strip()
    raw = str(item.get("raw_content") or "").strip()

    if raw:
        passages = select_passages(raw, question, EXCERPT_CHARS)

        # Keep Tavily's extract too when the page passages are thin.
        if len(passages) >= min(600, EXCERPT_CHARS // 2):
            return passages

        return f"{content}\n\n{passages}".strip()[:EXCERPT_CHARS]

    return content[:EXCERPT_CHARS]


def search_group(client, query, question, domains=None):
    """
    Run one Tavily search and return cleaned, classified results.

    With domains, the search is limited to those domains. Raises on
    search errors so the caller can count them.
    """

    options = {
        "query": query,
        "search_depth": SEARCH_DEPTH,
        "max_results": 5 if domains else 7,
        "include_raw_content": USE_PAGE_TEXT,
    }

    if domains:
        options["include_domains"] = list(domains)

    response = client.search(**options)

    group = []

    for item in response.get("results", []):
        url = str(item.get("url", "")).strip()
        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            continue

        snippet = evidence_text(item, question)

        if not snippet:
            continue

        title = str(item.get("title", "Untitled"))[:200]

        group.append({
            "title": title,
            "url": url,
            "snippet": snippet,
            "query": query,
            "source_type": classify_source(url, title),
            "score": item.get("score"),
            "authoritative_search": bool(domains),
        })

    return group


def filter_relevant(group, question):
    """
    Drop results that are clearly off-topic.

    Searches limited to authoritative domains return their best match
    even when nothing on those domains is relevant (a 2007 supervisory
    framework for a question about AI agents, for example), so their
    results must clear a higher relevance score and keyword overlap.
    """

    kept = []
    dropped = 0

    for source in group:
        authoritative = source.get("authoritative_search", False)
        score = source.get("score")
        minimum = MIN_AUTHORITATIVE_SCORE if authoritative else MIN_SOURCE_SCORE

        low_score = isinstance(score, (int, float)) and score < minimum

        text = f"{source['title']} {source['snippet']}"
        share = 0.6 if authoritative else 0.5

        if low_score or not is_on_topic(question, text, share=share):
            dropped += 1
            continue

        kept.append(source)

    return kept, dropped


def balanced_sources(groups, limit, seen_urls=None, seen_titles=None):
    """
    Take results from each search in turn, best-ranked first.

    Pages already selected, or with the same title as a selected page
    (the same document under another URL), are skipped.
    """
    groups = [
        sorted(
            group,
            key=lambda item: (
                source_rank(item),
                -len(item["snippet"]),
            ),
        )
        for group in groups
    ]

    selected = []
    seen = set(seen_urls or ())
    titles = set(seen_titles or ())

    depth = max((len(group) for group in groups), default=0)

    for position in range(depth):
        for group in groups:
            if position >= len(group):
                continue

            source = group[position]
            key = canonical_url(source["url"])
            title = normalize_title(source["title"])

            if key in seen or (title and title in titles):
                continue

            seen.add(key)

            if title:
                titles.add(title)

            selected.append(source)

            if len(selected) >= limit:
                return selected

    return selected


def tavily_client():
    from tavily import TavilyClient

    key = os.getenv("TAVILY_API_KEY")

    if not key:
        raise RuntimeError("TAVILY_API_KEY is missing.")

    return TavilyClient(api_key=key)


def run_searches(client, question, queries, authoritative_queries):
    """Run the searches and return (kept, unfiltered, dropped, errors).

    Authoritative searches come first, so their results get first
    pick when sources are selected.
    """
    groups = []
    unfiltered = []
    dropped = 0
    errors = 0

    plans = [(query, AUTHORITATIVE_DOMAINS) for query in authoritative_queries]
    plans += [(query, None) for query in queries]

    for query, domains in plans:
        try:
            group = search_group(client, query, question, domains)

        except Exception as exc:
            errors += 1

            print(
                "Search failed:",
                type(exc).__name__,
                str(exc)[:150],
            )
            continue

        kept, count = filter_relevant(group, question)
        groups.append(kept)
        unfiltered.append(group)
        dropped += count

    return groups, unfiltered, dropped, errors


def describe_sources(sources):
    independent = sum(
        source.get("source_type") in INDEPENDENT_TYPES
        for source in sources
    )

    return f"{len(sources)} sources ({independent} from recognised independent publishers)"


def researcher(state: State):
    start = time.perf_counter()
    client = tavily_client()

    # Extra searches limited to regulators, central banks,
    # international bodies and research publishers. Vendor blogs
    # otherwise dominate general web search results.
    groups, unfiltered, dropped, errors = run_searches(
        client,
        state["question"],
        state["queries"],
        state["queries"][:2],
    )

    selected = balanced_sources(groups, MAX_SOURCES)

    # Never fail a run just because the relevance filter was strict.
    if not selected:
        selected = balanced_sources(unfiltered, MAX_SOURCES)

    if not selected:
        raise RuntimeError("No usable source evidence was retrieved.")

    sources = [
        {"id": index, **source}
        for index, source in enumerate(selected, 1)
    ]

    return {
        "sources": sources,
        "read_source_ids": [],
        "approved": [],
        "evidence": [],
        "rejected": [],
        "reviews": [],
        "followup_rounds": 0,
        "events": record(
            state,
            "researcher",
            start,
            f"{describe_sources(sources)}; "
            f"{dropped} off-topic dropped; "
            f"{errors} search errors",
        ),
    }


# ============================================================
# Follow-up: the agent decides whether to search again
# ============================================================

class FollowupQueries(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queries: list[str]


FOLLOWUP_SYSTEM = """
You write web search queries to fill gaps in a research report.

You are given a research question and report themes that still have
too little evidence. Write one specific search query per theme that
would find evidence for that theme of the question.

Example
Question: How are central banks regulating stablecoins?
Themes: Reserve requirements; Cross-border coordination
Queries:
- stablecoin reserve requirements central bank rules
- cross-border coordination stablecoin regulation FSB

Return structured output only.
"""


def thin_themes(state):
    """Planned themes with fewer verified findings than the minimum."""
    counts = {}

    for finding in state.get("approved", []):
        counts[finding["category"]] = counts.get(finding["category"], 0) + 1

    return [
        theme
        for theme in state.get("themes", [])
        if theme != OTHER_THEME
        and counts.get(theme, 0) < MIN_FINDINGS_PER_THEME
    ]


def followup_queries(question, themes):
    try:
        response = structured_model(
            FollowupQueries,
            FOLLOWUP_SYSTEM,
            f"QUESTION:\n{question}\n\nTHEMES:\n" + "\n".join(themes),
            tokens=300,
        )

        queries = [
            " ".join(str(query).split())[:180]
            for query in response.queries
            if str(query).strip()
        ]

    except Exception as exc:
        print(
            "Follow-up query fallback:",
            type(exc).__name__,
            str(exc)[:150],
        )
        queries = []

    if not queries:
        queries = [f"{question[:120]} {theme}" for theme in themes]

    return list(dict.fromkeys(queries))[: len(themes)]


def followup(state: State):
    """
    Decide whether the evidence is good enough to write the report.

    If some planned themes have too few verified findings, and the
    follow-up budget is not spent, search again for those themes and
    add the new sources for the extractor. Otherwise do nothing; the
    graph then moves on to the writer.
    """
    start = time.perf_counter()
    rounds = state.get("followup_rounds", 0)
    gaps = thin_themes(state)

    if not gaps or rounds >= MAX_FOLLOWUP_ROUNDS:
        reason = (
            "every theme has enough verified findings"
            if not gaps
            else f"follow-up budget used ({rounds} round"
            f"{'' if rounds == 1 else 's'})"
        )

        return {
            "events": record(state, "followup", start, f"skipped: {reason}"),
        }

    queries = followup_queries(state["question"], gaps)
    client = tavily_client()

    groups, _unfiltered, dropped, errors = run_searches(
        client,
        state["question"],
        queries,
        [],
    )

    existing = state["sources"]

    new = balanced_sources(
        groups,
        FOLLOWUP_SOURCES,
        seen_urls={canonical_url(source["url"]) for source in existing},
        seen_titles={normalize_title(source["title"]) for source in existing},
    )

    next_id = max((source["id"] for source in existing), default=0) + 1

    added = [
        {"id": next_id + offset, **source}
        for offset, source in enumerate(new)
    ]

    return {
        "sources": existing + added,
        "followup_rounds": rounds + 1,
        "followup_queries": state.get("followup_queries", []) + queries,
        "events": record(
            state,
            "followup",
            start,
            f"round {rounds + 1}: {len(gaps)} thin theme"
            f"{'' if len(gaps) == 1 else 's'} "
            f"({', '.join(gaps)}); {len(queries)} searches; "
            f"{len(added)} new sources; {dropped} off-topic dropped; "
            f"{errors} search errors",
        ),
    }


def route_after_followup(state: State):
    """Back to the extractor if the follow-up found new sources."""
    unread = [
        source
        for source in state.get("sources", [])
        if source["id"] not in set(state.get("read_source_ids", []))
    ]

    return "extractor" if unread else "writer"