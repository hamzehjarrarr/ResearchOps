"""Streamlit dashboard for ResearchOps AI."""

import html
import os
import re
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API = os.getenv(
    "API_URL",
    "http://localhost:8000",
).rstrip("/")

POLL_INTERVAL = "3s"

st.set_page_config(
    page_title="ResearchOps AI",
    page_icon="🔎",
    layout="wide",
)


# --------------------------------------------------
# Constants
# --------------------------------------------------

EXAMPLES = {
    "AI agents in finance": (
        "How are AI agents being adopted by financial "
        "institutions, and what are the main risks?"
    ),
    "LLMs in clinical care": (
        "What are the main benefits and risks of using large "
        "language models for clinical decision support?"
    ),
    "Stablecoin regulation": (
        "How are central banks and regulators approaching "
        "the regulation of stablecoins?"
    ),
}

STAGES = [
    ("planner", "Plan", "Drafts search queries and report themes"),
    ("researcher", "Search", "Retrieves, filters and ranks sources"),
    ("extractor", "Extract", "Quote-grounded findings, one source at a time"),
    ("verifier", "Verify", "Independent evidence review"),
    ("followup", "Follow up", "Searches again for thin themes"),
    ("writer", "Write", "Synthesis, every sentence verified"),
]

# Source types that count as independent evidence.
INDEPENDENT_TYPES = ("official", "academic", "institution", "news")

SOURCE_LABELS = {
    "official": "Official / regulator",
    "academic": "Academic / research",
    "institution": "Think tank / industry body",
    "news": "News / trade press",
    "commercial": "Vendor / commercial",
    "other": "Unrecognised site",
}

SOURCE_COLORS = {
    "official": "#2dd4bf",
    "academic": "#60a5fa",
    "institution": "#a78bfa",
    "news": "#fbbf24",
    "commercial": "#fb923c",
    "other": "#94a3b8",
}

STATUS_COLORS = {
    "supported": "#34d399",
    "research_limitation": "#60a5fa",
    "insufficient_evidence": "#fbbf24",
    "conflicting": "#f87171",
    "review_error": "#f87171",
}

STATUS_ICONS = {
    "supported": "✅",
    "research_limitation": "ℹ️",
    "insufficient_evidence": "⚠️",
    "conflicting": "❌",
    "review_error": "🛑",
}

CATEGORY_LABELS = {
    "adoption": "Adoption",
    "applications": "Applications",
    "benefits": "Benefits",
    "governance": "Governance",
    "operational_risk": "Operational risk",
    "security_risk": "Security risk",
    "financial_risk": "Financial risk",
    "limitations": "Limitations",
}

RUN_ICONS = {
    "running": "⏳",
    "completed": "✅",
    "failed": "❌",
}

ACTIVE_STATES = ("running", "queued", "pending")


# --------------------------------------------------
# Styling
# --------------------------------------------------

CSS = """
<style>
.block-container { padding-top: 1.5rem; max-width: 1250px; }

.hero {
    padding: 28px 32px;
    border-radius: 18px;
    margin-bottom: 1.2rem;
    background: linear-gradient(120deg, #1b2347 0%, #2b2166 45%, #0f3d63 100%);
    border: 1px solid rgba(124, 131, 255, 0.35);
}
.hero-title { font-size: 2.1rem; font-weight: 800; color: #fff; letter-spacing: -0.5px; }
.hero-sub { color: #c9d1f5; font-size: 1.02rem; margin: 6px 0 14px 0; }
.pill {
    display: inline-block; margin: 0 6px 6px 0; padding: 3px 12px;
    border-radius: 999px; font-size: 0.78rem; font-weight: 600;
    color: #dfe4ff; background: rgba(255, 255, 255, 0.10);
    border: 1px solid rgba(255, 255, 255, 0.18);
}

.badge {
    display: inline-block; margin: 0 6px 4px 0; padding: 2px 10px;
    border-radius: 999px; font-size: 0.74rem; font-weight: 600;
    color: var(--c); border: 1px solid var(--c);
    background: color-mix(in srgb, var(--c) 14%, transparent);
}

.mcard {
    padding: 14px 16px; border-radius: 14px; min-width: 0;
    background: rgba(124, 131, 255, 0.07);
    border: 1px solid rgba(124, 131, 255, 0.22);
}
.mlabel {
    font-size: 0.72rem; opacity: 0.7; text-transform: uppercase;
    letter-spacing: 0.4px; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
}
.mvalue {
    font-size: clamp(1.25rem, 2.2vw, 1.8rem); font-weight: 800;
    line-height: 1.3; white-space: nowrap;
}
.msub {
    font-size: 0.75rem; opacity: 0.65; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
}

.durations { margin: 6px 0 12px 0; }
.drow { display: flex; align-items: center; gap: 12px; margin: 7px 0; }
.dlabel { width: 92px; font-size: 0.85rem; font-weight: 600; }
.dtrack {
    flex: 1; height: 14px; border-radius: 8px;
    background: rgba(255, 255, 255, 0.07); overflow: hidden;
}
.dbar {
    height: 100%; border-radius: 8px;
    background: linear-gradient(90deg, #7c83ff, #38bdf8);
}
.dvalue { width: 64px; text-align: right; font-size: 0.85rem; opacity: 0.85; }

.stages { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 10px 0 14px 0; }
.stage {
    padding: 14px 16px; border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.10);
    background: rgba(255, 255, 255, 0.03);
}
.stage-title { font-weight: 700; font-size: 1rem; }
.stage-desc { font-size: 0.78rem; opacity: 0.65; margin: 2px 0 6px 0; }
.stage-meta { font-size: 0.78rem; font-weight: 600; }
.stage.done { border-color: rgba(52, 211, 153, 0.55); background: rgba(52, 211, 153, 0.08); }
.stage.done .stage-meta { color: #34d399; }
.stage.active {
    border-color: rgba(124, 131, 255, 0.9);
    background: rgba(124, 131, 255, 0.12);
    animation: pulse 1.6s ease-in-out infinite;
}
.stage.active .stage-meta { color: #9aa1ff; }
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(124, 131, 255, 0.45); }
    50% { box-shadow: 0 0 0 7px rgba(124, 131, 255, 0); }
}

[data-testid="stSidebar"] .stButton button {
    width: 100%; justify-content: flex-start; text-align: left;
}
</style>
"""


def run_time(run):
    """Short local time for a run, so similar questions can be told apart."""
    try:
        moment = datetime.fromisoformat(run["created_at"])
        return moment.astimezone().strftime("%b %d %H:%M")
    except (KeyError, ValueError, TypeError):
        return ""


def esc(value):
    return html.escape(str(value))


def badge(text, color):
    return (
        f'<span class="badge" style="--c:{color}">'
        f"{esc(text)}</span>"
    )


def metric_card(label, value, sub=""):
    return (
        '<div class="mcard">'
        f'<div class="mlabel">{esc(label)}</div>'
        f'<div class="mvalue">{esc(value)}</div>'
        f'<div class="msub">{esc(sub)}</div>'
        "</div>"
    )


def source_badge(source_type):
    label = SOURCE_LABELS.get(source_type)

    if not label:
        return ""

    return badge(label, SOURCE_COLORS[source_type])


def status_badge(status):
    return badge(
        f"{STATUS_ICONS.get(status, '')} "
        f"{str(status).replace('_', ' ')}",
        STATUS_COLORS.get(status, "#94a3b8"),
    )


# --------------------------------------------------
# Backend requests
# --------------------------------------------------

def get(path):
    response = requests.get(
        f"{API}{path}",
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


# --------------------------------------------------
# Pipeline progress
# --------------------------------------------------

def render_stages(events, running):
    finished = {
        event.get("agent"): event
        for event in events
    }

    active_assigned = False
    cells = []

    for key, title, description in STAGES:
        if key in finished:
            css = "done"
            seconds = finished[key].get("duration_ms", 0) / 1000
            meta = f"✓ {seconds:.1f}s"

        elif running and not active_assigned:
            css = "active"
            meta = "running…"
            active_assigned = True

        else:
            css = "pending"
            meta = ""

        cells.append(
            f'<div class="stage {css}">'
            f'<div class="stage-title">{esc(title)}</div>'
            f'<div class="stage-desc">{esc(description)}</div>'
            f'<div class="stage-meta">{esc(meta)}</div>'
            "</div>"
        )

    st.markdown(
        f'<div class="stages">{"".join(cells)}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# Report display
# --------------------------------------------------

CITATION_RUN_RE = re.compile(
    r"(?:【\d+(?:†L\d+(?:-L?\d+)?)?】)+"
)

CITATION_ONE_RE = re.compile(
    r"【(\d+)(?:†L\d+(?:-L?\d+)?)?】"
)


def render_report(report, sources):
    source_map = {
        str(source["id"]): source["url"]
        for source in sources
    }

    unverified_ids = {
        str(source["id"])
        for source in sources
        if source.get("source_type") not in INDEPENDENT_TYPES
    }

    def replace_run(match):
        ids = CITATION_ONE_RE.findall(match.group(0))

        links = "".join(
            f"[[{source_id}]]({source_map[source_id]})"
            if source_id in source_map
            else f"[Missing source {source_id}]"
            for source_id in ids
        )

        # Statement with no recognised independent source.
        if ids and all(i in unverified_ids for i in ids):
            links += " ⚠"

        return links

    st.caption(
        "⚠ marks statements supported only by "
        "vendor or unrecognised sites (no recognised independent source)."
    )

    st.markdown(
        CITATION_RUN_RE.sub(replace_run, report)
    )


# --------------------------------------------------
# Verification display
# --------------------------------------------------

def show_claim(claim, index):
    status = claim.get("status", "unknown")
    statement = claim.get("statement", "")

    title = (
        f"{index}. {STATUS_ICONS.get(status, '')} "
        f"{statement[:110]}"
    )

    with st.expander(title):
        st.markdown(
            status_badge(status),
            unsafe_allow_html=True,
        )

        st.markdown(f"**Claim:** {statement}")

        quote = claim.get("quote")

        if quote:
            st.markdown(f"> {quote}")
            st.caption("Verbatim quote from the cited source.")

        st.write(
            "**Source IDs:**",
            claim.get("source_ids", []),
        )

        st.write(
            "**Reviewer's reasoning:**",
            claim.get("explanation", ""),
        )


def show_verification(verification):
    coverage = verification.get("coverage_percent", 0)
    claims = verification.get("claims", [])

    supported = [
        claim for claim in claims
        if claim.get("status")
        in ("supported", "research_limitation")
    ]

    insufficient = [
        claim for claim in claims
        if claim.get("status") == "insufficient_evidence"
    ]

    conflicting = [
        claim for claim in claims
        if claim.get("status") == "conflicting"
    ]

    errors = [
        claim for claim in claims
        if claim.get("status") == "review_error"
    ]

    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(
        metric_card("Review coverage", f"{coverage}%"),
        unsafe_allow_html=True,
    )
    col2.markdown(
        metric_card("Supported", len(supported)),
        unsafe_allow_html=True,
    )
    col3.markdown(
        metric_card("Insufficient", len(insufficient)),
        unsafe_allow_html=True,
    )
    col4.markdown(
        metric_card("Conflicting", len(conflicting)),
        unsafe_allow_html=True,
    )

    st.write("")

    if errors:
        st.error(
            f"{len(errors)} claims could not be assessed "
            "due to technical errors."
        )

    if verification.get("passed"):
        st.success(
            "Automated evidence review passed. "
            "This does not guarantee factual accuracy."
        )
    else:
        st.warning(
            "Automated review found unresolved issues."
        )

    st.caption(
        f"{verification.get('reviewed_units', 0)} of "
        f"{verification.get('total_units', 0)} extracted "
        "claims reviewed."
    )

    filter_option = st.selectbox(
        "Show claims",
        [
            "All claims",
            "Flagged only",
            "Supported",
            "Insufficient evidence",
            "Conflicting",
            "Review errors",
        ],
    )

    if filter_option == "Flagged only":
        displayed = insufficient + conflicting + errors
    elif filter_option == "Supported":
        displayed = supported
    elif filter_option == "Insufficient evidence":
        displayed = insufficient
    elif filter_option == "Conflicting":
        displayed = conflicting
    elif filter_option == "Review errors":
        displayed = errors
    else:
        displayed = claims

    if not displayed:
        st.info("No claims match this filter.")

    for index, claim in enumerate(displayed, 1):
        show_claim(claim, index)

    issues = verification.get("issues", [])

    if issues:
        with st.expander("Unresolved issues"):
            for issue in dict.fromkeys(issues):
                st.write(f"- {issue}")

    limitations = verification.get("limitations", [])

    if limitations:
        with st.expander("Verification limitations"):
            for limitation in limitations:
                st.write(f"- {limitation}")


# --------------------------------------------------
# Findings the verifier caught
# --------------------------------------------------

def show_rejected(rejected):
    st.subheader(f"Caught by the verifier ({len(rejected)})")

    if not rejected:
        st.caption(
            "Every extracted finding passed review in this run."
        )
        return

    st.caption(
        "These findings passed the quote check but were not "
        "established by their cited excerpt, so they were left "
        "out of the report."
    )

    for index, item in enumerate(rejected, 1):
        title = (
            f"{index}. {STATUS_ICONS.get(item.get('review_status'), '🛑')} "
            f"{item.get('statement', '')[:100]}"
        )

        with st.expander(title):
            st.markdown(
                status_badge(item.get("review_status", ""))
                + source_badge(item.get("source_type")),
                unsafe_allow_html=True,
            )

            st.markdown(f"**Statement:** {item.get('statement', '')}")

            if item.get("quote"):
                st.markdown(f"> {item['quote']}")

            st.write(
                "**Why it was rejected:**",
                item.get("review_explanation", ""),
            )


# --------------------------------------------------
# Evidence display
# --------------------------------------------------

def show_evidence(evidence, sources):
    if not evidence:
        st.info("No evidence records were saved for this run.")
        return

    type_by_id = {
        source["id"]: source.get("source_type")
        for source in sources
    }

    categories = sorted({
        item.get("category", "") for item in evidence
    })

    col1, col2 = st.columns([3, 1])

    chosen = col1.multiselect(
        "Category",
        categories,
        default=categories,
        format_func=lambda c: CATEGORY_LABELS.get(c, c),
    )

    only_unverified = col2.toggle("No independent source")

    shown = 0

    for item in evidence:
        if item.get("category") not in chosen:
            continue

        if only_unverified and not item.get("unverified_only"):
            continue

        shown += 1

        with st.container(border=True):
            badges = badge(
                CATEGORY_LABELS.get(
                    item.get("category"),
                    item.get("category", ""),
                ),
                "#7c83ff",
            )

            for source_id in item.get("source_ids", []):
                badges += source_badge(
                    type_by_id.get(source_id)
                )
                badges += badge(
                    f"source {source_id}",
                    "#94a3b8",
                )

            if item.get("unverified_only"):
                badges += badge(
                    "⚠ no independent source",
                    "#fb923c",
                )

            badges += status_badge(
                item.get("review_status", "")
            )

            st.markdown(badges, unsafe_allow_html=True)
            st.markdown(f"**{item.get('statement', '')}**")

            if item.get("quote"):
                st.markdown(f"> {item['quote']}")

            if item.get("review_explanation"):
                st.caption(
                    "Reviewer: "
                    + item["review_explanation"]
                )

    if not shown:
        st.info("No findings match these filters.")


# --------------------------------------------------
# Sources display
# --------------------------------------------------

def show_sources(sources, evidence):
    cited = {
        source_id
        for item in evidence
        for source_id in item.get("source_ids", [])
    }

    for source in sources:
        is_cited = source["id"] in cited

        marker = "📌 " if is_cited else ""

        with st.expander(
            f"{marker}[{source['id']}] {source['title']}"
        ):
            badges = source_badge(
                source.get("source_type")
            )

            badges += badge(
                "cited in report" if is_cited else "not cited",
                "#34d399" if is_cited else "#94a3b8",
            )

            score = source.get("score")

            if isinstance(score, (int, float)):
                badges += badge(
                    f"relevance {score:.2f}",
                    "#94a3b8",
                )

            st.markdown(badges, unsafe_allow_html=True)

            st.markdown(
                "[Open original source]"
                f"({source['url']})"
            )

            st.write(source.get("snippet", ""))


# --------------------------------------------------
# Pipeline display
# --------------------------------------------------

def show_pipeline(result):
    events = result.get("events", [])

    render_stages(events, running=False)

    st.write("**Research plan**")

    for index, query in enumerate(
        result.get("plan", []),
        start=1,
    ):
        st.write(f"{index}. {query}")

    followups = result.get("followup_queries", [])

    if followups:
        st.write("**Follow-up searches** (themes with too little evidence)")

        for index, query in enumerate(followups, start=1):
            st.write(f"{index}. {query}")

    if not events:
        return

    longest = max(
        (event.get("duration_ms", 0) for event in events),
        default=1,
    ) or 1

    rows = []

    for event in events:
        seconds = event.get("duration_ms", 0) / 1000
        width = max(
            2,
            round(100 * event.get("duration_ms", 0) / longest),
        )

        rows.append(
            '<div class="drow">'
            f'<div class="dlabel">{esc(event.get("agent", ""))}</div>'
            '<div class="dtrack">'
            f'<div class="dbar" style="width:{width}%"></div>'
            "</div>"
            f'<div class="dvalue">{seconds:.1f}s</div>'
            "</div>"
        )

    st.write("**Time per agent**")

    st.markdown(
        f'<div class="durations">{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )

    st.write("**What each agent did**")

    for event in events:
        st.markdown(
            f"- **{event.get('agent', '')}**: "
            f"{event.get('detail', '')}"
        )


# --------------------------------------------------
# Completed research display
# --------------------------------------------------

@st.fragment
def show_completed_run(run_id, result):
    verification = result.get("verification", {})
    sources = result.get("sources", [])
    evidence = result.get("evidence", [])
    rejected = result.get("rejected", [])
    events = result.get("events", [])

    cited = {
        source_id
        for item in evidence
        for source_id in item.get("source_ids", [])
    }

    independent = sum(
        1
        for source in sources
        if source["id"] in cited
        and source.get("source_type") in INDEPENDENT_TYPES
    )

    unverified_only = sum(
        1 for item in evidence if item.get("unverified_only")
    )

    total_seconds = sum(
        event.get("duration_ms", 0) for event in events
    ) / 1000

    cards = st.columns(5)

    cards[0].markdown(
        metric_card(
            "Findings",
            len(evidence),
            f"{unverified_only} without an independent source",
        ),
        unsafe_allow_html=True,
    )
    cards[1].markdown(
        metric_card(
            "Sources cited",
            f"{len(cited)}/{len(sources)}",
            "of those retrieved",
        ),
        unsafe_allow_html=True,
    )
    cards[2].markdown(
        metric_card(
            "Independent",
            f"{independent}/{len(cited)}",
            "recognised publishers, cited",
        ),
        unsafe_allow_html=True,
    )
    cards[3].markdown(
        metric_card(
            "Run time",
            f"{total_seconds:.0f}s",
            "local model",
        ),
        unsafe_allow_html=True,
    )
    cards[4].markdown(
        metric_card(
            "Caught",
            len(rejected),
            "rejected by verifier",
        ),
        unsafe_allow_html=True,
    )

    st.write("")

    tabs = st.tabs([
        "📄 Report",
        "🧾 Evidence",
        "✅ Verification",
        "📚 Sources",
        "⚙️ Pipeline",
    ])

    with tabs[0]:
        render_report(result.get("report", ""), sources)

        st.download_button(
            "Download Markdown report",
            data=result.get("report", ""),
            file_name=f"research-{run_id[:8]}.md",
            mime="text/markdown",
        )

        if not result.get("review_passed", False):
            st.warning(
                "This report contains unresolved "
                "automated verification findings."
            )

    with tabs[1]:
        st.caption(
            "Each finding is tied to a verbatim quote from its "
            "source and the reviewer's reasoning."
        )
        show_evidence(evidence, sources)

    with tabs[2]:
        show_verification(verification)

        st.divider()
        show_rejected(rejected)

        feedback = result.get("review_feedback", "")

        if feedback:
            st.caption(feedback)

    with tabs[3]:
        show_sources(sources, evidence)

    with tabs[4]:
        show_pipeline(result)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    CSS
    + """
<div class="hero">
  <div class="hero-title">🔎 ResearchOps AI</div>
  <div class="hero-sub">
    Multi-agent research where every claim is tied to a verbatim
    quote and independently verified.
  </div>
  <span class="pill">LangGraph</span>
  <span class="pill">Ollama · local LLM</span>
  <span class="pill">Tavily search</span>
  <span class="pill">FastAPI</span>
  <span class="pill">Quote-grounded citations</span>
</div>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Backend health
# --------------------------------------------------

try:
    status = get("/health")

except requests.RequestException:
    st.error(
        "Cannot connect to FastAPI. "
        "Start the backend first."
    )
    st.stop()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("System status")

    if not status.get("ollama_reachable"):
        ollama_badge = badge("Ollama not reachable", "#f87171")
    elif not status.get("ollama_model_pulled"):
        ollama_badge = badge(
            f"{status.get('ollama_model')} not pulled",
            "#fbbf24",
        )
    else:
        ollama_badge = badge(
            f"Ollama · {status.get('ollama_model')}",
            "#34d399",
        )

    tavily_badge = (
        badge("Tavily ready", "#34d399")
        if status.get("tavily_configured")
        else badge("Tavily key missing", "#f87171")
    )

    st.markdown(
        ollama_badge + tavily_badge,
        unsafe_allow_html=True,
    )

    st.divider()
    st.header("Recent research")

    try:
        runs = get("/runs")

        for run in runs:
            icon = RUN_ICONS.get(run["status"], "•")

            label = (
                f"{icon} {run_time(run)} · "
                f"{run['question'][:30]}"
            )

            if st.button(label, key=run["id"]):
                st.session_state["run_id"] = run["id"]
                st.rerun()

    except requests.RequestException:
        st.warning("Unable to load research history.")

    st.divider()

    st.caption(
        "Do not submit confidential data. "
        "This local demo has no authentication."
    )


# --------------------------------------------------
# Research form
# --------------------------------------------------

st.session_state.setdefault(
    "question",
    next(iter(EXAMPLES.values())),
)

st.markdown("##### Try an example")

example_columns = st.columns(len(EXAMPLES))

for column, (name, text) in zip(
    example_columns,
    EXAMPLES.items(),
    strict=True,
):
    if column.button(name, key=f"example-{name}"):
        st.session_state["question"] = text

with st.form("research"):
    question = st.text_area(
        "Research question",
        key="question",
        height=110,
        max_chars=2000,
    )

    submitted = st.form_submit_button(
        "Run research",
        type="primary",
    )


if submitted:
    if len(question.strip()) < 12:
        st.warning("Enter at least 12 characters.")
    else:
        try:
            response = requests.post(
                f"{API}/research",
                json={"question": question.strip()},
                timeout=20,
            )

            if response.status_code == 422:
                st.error(
                    "Research question failed "
                    "backend validation."
                )
                st.json(response.json())
                st.stop()

            response.raise_for_status()

            st.session_state["run_id"] = (
                response.json()["id"]
            )

            st.rerun()

        except requests.RequestException as exc:
            st.error(
                f"Could not start research: {exc}"
            )


# --------------------------------------------------
# Automatic research status updates
# --------------------------------------------------

def run_events(run):
    result = run.get("result")

    if isinstance(result, dict):
        return result.get("events", [])

    return []


@st.fragment(run_every=POLL_INTERVAL)
def poll_running_run(run_id):
    """
    Poll only while a run is in progress. As soon as the run
    leaves the running state, rerun the whole app so the result
    is rendered by the normal (non-polling) code path below.
    """
    try:
        run = get(f"/runs/{run_id}")
    except requests.RequestException as exc:
        st.error(f"Unable to load research run: {exc}")
        return

    if run.get("status") not in ACTIVE_STATES:
        st.rerun()
        return

    render_stages(run_events(run), running=True)

    st.caption(
        "Running on a local model, so a full run takes a few "
        "minutes. This page updates automatically."
    )


def show_active_run():
    run_id = st.session_state.get("run_id")

    if not run_id:
        return

    try:
        run = get(f"/runs/{run_id}")
    except requests.RequestException as exc:
        st.error(f"Unable to load research run: {exc}")
        return

    st.divider()

    st.markdown(f"#### {run.get('question', '')}")
    st.caption(f"Run {run_id[:8]}")

    run_status = run.get("status")

    if run_status in ACTIVE_STATES:
        poll_running_run(run_id)
        return

    if run_status == "failed":
        render_stages(run_events(run), running=False)
        st.error(run.get("error") or "Research failed.")
        return

    if run_status == "completed":
        result = run.get("result")

        if not isinstance(result, dict):
            st.error(
                "The run is marked completed, "
                "but its result is missing."
            )
            return

        show_completed_run(run_id, result)
        return

    st.warning(f"Unknown research status: {run_status}")


show_active_run()