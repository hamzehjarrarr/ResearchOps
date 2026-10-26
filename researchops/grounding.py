"""Deterministic grounding checks and source classification.

Nothing in this module calls a model. These checks are cheap, repeatable
and independent of the model's own judgement:

- classify_source: labels a URL as official, academic, institution,
  news, commercial (vendor) or other content.
- quote_in_excerpt: confirms a quoted passage really appears in the
  source excerpt the model was given.
- numbers_grounded: confirms every number in a statement appears in
  that excerpt.
- is_on_topic: drops search results unrelated to the question.
- added_question_terms: catches a statement that brings in a topic
  from the question that its own excerpt never mentions.
- select_passages: keeps the parts of a long page that relate to the
  question.
"""

import re
import unicodedata
from urllib.parse import urlparse

SOURCE_LABELS = {
    "official": "Official / regulator",
    "academic": "Academic / research",
    "institution": "Think tank / industry body",
    "news": "News / trade press",
    "commercial": "Vendor / commercial",
    "other": "Unrecognised site",
}

# Source types that count as independent evidence. Unrecognised sites
# are often company blogs, so they do not count.
INDEPENDENT_TYPES = frozenset({"official", "academic", "institution", "news"})

# Lower rank is preferred when selecting sources.
TYPE_RANK = {
    "official": 0,
    "academic": 1,
    "institution": 1,
    "news": 2,
    "other": 3,
    "commercial": 4,
}

OFFICIAL_DOMAINS = (
    "gov.uk",
    "canada.ca",
    "gc.ca",
    "who.int",
    "europa.eu",
    "bankofengland.co.uk",
    "fca.org.uk",
    "bis.org",
    "imf.org",
    "worldbank.org",
    "oecd.org",
    "fsb.org",
    "un.org",
)

INSTITUTION_DOMAINS = (
    "weforum.org",
    "cloudsecurityalliance.org",
    "rooseveltinstitute.org",
    "finos.org",
    "rand.org",
    "brookings.edu",
    "cfr.org",
    "piie.com",
    "bruegel.org",
    "chathamhouse.org",
    "csis.org",
    "intereconomics.eu",
    "cepr.org",
    "atlanticcouncil.org",
    "carnegieendowment.org",
)

ACADEMIC_DOMAINS = (
    "arxiv.org",
    "sciencedirect.com",
    "nature.com",
    "science.org",
    "ieee.org",
    "acm.org",
    "springer.com",
    "ssrn.com",
    "mdpi.com",
    "wiley.com",
    "tandfonline.com",
    "jstor.org",
    "researchgate.net",
    "nber.org",
    "thelancet.com",
    "nejm.org",
    "bmj.com",
    "cell.com",
    "plos.org",
    "biorxiv.org",
    "medrxiv.org",
)

NEWS_DOMAINS = (
    "reuters.com",
    "bloomberg.com",
    "ft.com",
    "wsj.com",
    "nytimes.com",
    "cnbc.com",
    "economist.com",
    "apnews.com",
    "bbc.com",
    "theguardian.com",
    "americanbanker.com",
    "finextra.com",
    "risk.net",
    "yahoo.com",
)

# Words in a URL path that signal marketing or vendor content.
VENDOR_PATH_WORDS = frozenset({
    "blog", "blogs", "resources", "resource", "product", "products",
    "solutions", "platform", "pricing", "guides", "learn", "use-cases",
    "case-studies", "customers", "press-release", "agentic-insights",
})

VENDOR_TLDS = (".ai", ".io")

# Pages on official domains that are staff research, not policy.
# Working papers and staff notes carry the authors' views, not the
# institution's, so they are classed as research.
RESEARCH_PATH_HINTS = (
    "working-paper",
    "workingpaper",
    "working_paper",
    "/scpwps/",
    "/wps/",
    "/ifdp/",
    "feds-notes",
    "/econres/",
    "discussion-paper",
    "staff-report",
    "staff-working",
)

# Domains used for an extra search that targets authoritative sources.
AUTHORITATIVE_DOMAINS = (
    "bis.org", "imf.org", "oecd.org", "worldbank.org", "fsb.org",
    "bankofengland.co.uk", "fca.org.uk", "federalreserve.gov",
    "treasury.gov", "sec.gov", "consumerfinance.gov", "ftc.gov",
    "nist.gov", "europa.eu", "gov.uk", "canada.ca", "osfi-bsif.gc.ca",
    "mas.gov.sg", "hkma.gov.hk", "un.org", "who.int", "nih.gov",
    "cdc.gov", "fda.gov", "arxiv.org", "ssrn.com", "nber.org",
    "weforum.org", "brookings.edu", "nature.com", "thelancet.com",
    "nejm.org", "sciencedirect.com", "ieee.org", "acm.org",
    "springer.com", "jstor.org",
)

STOPWORDS = frozenset({
    "about", "above", "after", "also", "been", "being", "between",
    "both", "could", "does", "each", "from", "have", "into", "like",
    "made", "main", "make", "many", "more", "most", "much", "only",
    "other", "over", "same", "should", "some", "such", "than", "that",
    "their", "them", "then", "there", "these", "they", "this", "those",
    "through", "used", "using", "very", "were", "what", "when",
    "where", "which", "while", "will", "with", "would", "your",
})

MIN_QUOTE_CHARS = 25
MIN_QUOTE_PART_CHARS = 8


def _host(url):
    host = urlparse(str(url)).netloc.lower().split(":")[0]
    return host[4:] if host.startswith("www.") else host


def _matches(host, domains):
    return any(
        host == domain or host.endswith("." + domain)
        for domain in domains
    )


def _looks_vendor(url):
    """True for pages that look like vendor marketing content."""
    parsed = urlparse(str(url))
    host = _host(url)

    if host.endswith(VENDOR_TLDS):
        return True

    if host.startswith(("blog.", "resources.", "learn.")):
        return True

    segments = {
        segment.lower()
        for segment in parsed.path.split("/")
        if segment
    }

    return bool(segments & VENDOR_PATH_WORDS)


def classify_source(url, title=""):
    """Classify a source from its domain and URL.

    Only pages that look like vendor marketing are "commercial".
    Anything else not recognised is "other", so unfamiliar but
    legitimate sites are not mislabelled as vendors.
    """
    host = _host(url)

    if (
        host.endswith(".gov")
        or ".gov." in host
        or host.endswith(".mil")
        or _matches(host, OFFICIAL_DOMAINS)
    ):
        path = urlparse(str(url)).path.lower()

        if any(hint in path for hint in RESEARCH_PATH_HINTS):
            return "academic"

        return "official"

    if _matches(host, INSTITUTION_DOMAINS):
        return "institution"

    if (
        host.endswith(".edu")
        or ".edu." in host
        or host.endswith(".ac.uk")
        or _matches(host, ACADEMIC_DOMAINS)
    ):
        return "academic"

    if _matches(host, NEWS_DOMAINS):
        return "news"

    if _looks_vendor(url):
        return "commercial"

    return "other"


def is_vendor_only(source_ids, source_types):
    """True when every cited source is vendor or commercial content."""
    types = [source_types.get(i) for i in source_ids]

    return bool(types) and all(t == "commercial" for t in types)


def lacks_independent_source(source_ids, source_types):
    """True when no cited source is a recognised independent publisher.

    Vendor pages and unrecognised sites (often company blogs) do not
    count as independent evidence.
    """
    types = [source_types.get(i) for i in source_ids]

    return bool(types) and not any(t in INDEPENDENT_TYPES for t in types)


def normalize_title(title):
    """A comparable form of a page title, for spotting duplicate pages.

    The same document is often published under several URLs (for
    example one per year), with the same title.
    """
    return _squash(re.sub(r"\b(?:19|20)\d\d\b", "", str(title)))


REFERENCE_PATTERNS = (
    r"\[cited\b",
    r"\bcited\s+(?:20|19)\d\d",
    r"\baccessed\s+(?:20|19)?\d",
    r"\bdoi\b",
    r"https?://",
    r"\bavailable\s+(?:from|at|online)\b",
    r"\bretrieved\s+from\b",
)

REFERENCE_RE = re.compile(
    "|".join(REFERENCE_PATTERNS),
    re.IGNORECASE,
)


def looks_like_reference(text):
    """True for bibliography or citation entries.

    A reference-list line such as "Title; 2024 Nov [cited 2025 Sep 18]"
    only proves the work exists. It is not a claim the source makes.
    """
    return bool(REFERENCE_RE.search(str(text)))


def question_keywords(question):
    """First five letters of each content word in the question."""
    stems = []

    for word in re.findall(r"[a-z]{4,}", str(question).lower()):
        if word in STOPWORDS:
            continue

        stem = word[:5]

        if stem not in stems:
            stems.append(stem)

    return stems


def is_on_topic(question, text, share=0.5):
    """True when enough of the question's keywords appear in the text.

    Stems are matched on the first five letters, so "adopted" also
    matches "adoption" and "agents" matches "agentic".
    """
    stems = question_keywords(question)

    if not stems:
        return True

    needed = max(1, round(len(stems) * share))

    words = {
        word[:5]
        for word in re.findall(r"[a-z]{4,}", str(text).lower())
    }

    return sum(stem in words for stem in stems) >= needed


def _squash(text):
    """Lowercase and drop everything except letters and digits.

    Makes the comparison immune to spacing, hyphen and quote differences,
    including words the model ran together such as "areshort".
    """
    text = unicodedata.normalize("NFKC", str(text)).casefold()

    return re.sub(r"[\W_]+", "", text)


def quote_in_excerpt(quote, excerpt):
    """True if the quote (or every part of an elided quote) is in the excerpt."""
    haystack = _squash(excerpt)

    parts = [
        _squash(part)
        for part in re.split(r"\.{3}|…", str(quote))
    ]

    parts = [part for part in parts if part]

    if not parts:
        return False

    if sum(len(part) for part in parts) < MIN_QUOTE_CHARS:
        return False

    return all(
        len(part) >= MIN_QUOTE_PART_CHARS and part in haystack
        for part in parts
    )


NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


def numbers_in(text):
    return {
        match.replace(",", "")
        for match in NUMBER_RE.findall(str(text))
    }


def numbers_grounded(statement, excerpt):
    """True if every number in the statement also appears in the excerpt."""
    return numbers_in(statement) <= numbers_in(excerpt)


def added_question_terms(statement, excerpt, question):
    """Question topics a statement mentions that its excerpt does not.

    A small model asked to answer a question tends to rewrite a finding
    so it fits, for example turning "a machine-learning engine predicts
    loan defaults" into "AI agents improve credit risk analysis". Any
    question keyword in the statement must also appear in the excerpt.
    """
    excerpt_words = {
        word[:5]
        for word in re.findall(r"[a-z]{4,}", str(excerpt).lower())
    }

    statement_words = {
        word[:5]
        for word in re.findall(r"[a-z]{4,}", str(statement).lower())
    }

    return [
        stem
        for stem in question_keywords(question)
        if stem in statement_words and stem not in excerpt_words
    ]


def select_passages(text, question, limit):
    """Keep the passages of a page that best match the question.

    The page is split into paragraphs; paragraphs are scored by how
    many of the question's keywords they contain, and the best are
    kept in their original order until ``limit`` characters.
    """
    stems = question_keywords(question)

    paragraphs = [
        " ".join(part.split())
        for part in re.split(r"\n\s*\n|\n(?=[#*-]\s)", str(text))
    ]

    paragraphs = [
        part
        for part in paragraphs
        if len(part) >= 80 and not looks_like_reference(part)
    ]

    def score(part):
        words = {
            word[:5]
            for word in re.findall(r"[a-z]{4,}", part.lower())
        }

        return sum(stem in words for stem in stems)

    ranked = sorted(
        range(len(paragraphs)),
        key=lambda index: (-score(paragraphs[index]), index),
    )

    chosen = []
    used = 0

    for index in ranked:
        if score(paragraphs[index]) == 0:
            break

        part = paragraphs[index]

        if used + len(part) > limit:
            if used == 0:
                chosen.append(index)
                used = limit
            continue

        chosen.append(index)
        used += len(part) + 2

        if used >= limit:
            break

    text = "\n\n".join(paragraphs[index] for index in sorted(chosen))

    return text[:limit]