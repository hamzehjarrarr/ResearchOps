"""Tests for the agent loop, the faithful extractor checks and the writer."""

import pytest
from pydantic import ValidationError

import researchops.researcher as researcher
import researchops.themes as theme_mod
import researchops.writer as writer
from researchops.api import ResearchRequest
from researchops.audit import extract_claims
from researchops.graph import build_graph
from researchops.grounding import (
    added_question_terms,
    normalize_title,
    select_passages,
)
from researchops.report import assemble_report, evidence_gaps, verify_assembled_report
from researchops.writer import WriterOutput, WrittenSentence

QUESTION = "How are AI agents changing credit risk analysis?"

SOURCES = [
    {"id": 1, "title": "A", "url": "https://www.bis.org/a", "snippet": "x", "source_type": "official"},
    {"id": 2, "title": "B", "url": "https://arxiv.org/abs/2", "snippet": "x", "source_type": "academic"},
    {"id": 3, "title": "C", "url": "https://vendor.com/c", "snippet": "x", "source_type": "commercial"},
]

FINDINGS = [
    {"statement": "Banks use AI agents to draft credit memos.", "source_ids": [1],
     "category": "Applications", "quote": "agents draft credit memos for analysts", "relevance": 5,
     "source_type": "official"},
    {"statement": "AI agents summarise borrower filings for analysts.", "source_ids": [2],
     "category": "Applications", "quote": "agents summarise borrower filings", "relevance": 4,
     "source_type": "academic"},
    {"statement": "Supervisors expect human review of agent outputs.", "source_ids": [1],
     "category": "Risks", "quote": "human review of agent outputs", "relevance": 4,
     "source_type": "official"},
    {"statement": "Agent errors can propagate into credit decisions.", "source_ids": [3],
     "category": "Risks", "quote": "errors can propagate into credit decisions", "relevance": 4,
     "source_type": "commercial"},
]


# ---------- faithful extraction ----------

def test_drift_check_flags_question_topics_missing_from_the_excerpt():
    excerpt = "A machine-learning engine predicts loan defaults from repayment history."
    drifted = "AI agents improve credit risk analysis by predicting loan defaults."
    faithful = "A machine-learning engine predicts loan defaults."

    assert added_question_terms(drifted, excerpt, QUESTION)
    assert not added_question_terms(faithful, excerpt, QUESTION)


def test_select_passages_keeps_relevant_paragraphs_in_order():
    page = "\n\n".join([
        "Cookie policy: this site uses cookies to improve your browsing experience on every page.",
        "AI agents are changing credit risk analysis by drafting memos and monitoring borrowers.",
        "Our office is closed on public holidays and reopens the following working day as usual.",
        "Credit analysts now review agent output before any risk decision is taken by the bank.",
    ])

    text = select_passages(page, QUESTION, 400)

    assert "Cookie" not in text and "holidays" not in text
    assert text.index("drafting memos") < text.index("Credit analysts")
    assert len(select_passages(page, QUESTION, 100)) <= 100


def test_evidence_text_prefers_page_passages():
    page = "\n\n".join(
        ["AI agents are changing credit risk analysis at large banks and lenders. " * 3] * 6
    )
    item = {"content": "Short search snippet.", "raw_content": page}
    text = researcher.evidence_text(item, QUESTION)
    assert "credit risk" in text
    assert "Short search snippet" not in text

    assert researcher.evidence_text({"content": "Only a snippet."}, QUESTION) == "Only a snippet."


# ---------- source selection ----------

def test_same_document_under_another_url_is_dropped():
    assert normalize_title("OSFI Guideline E-23 (2025)") == normalize_title("OSFI guideline E-23 2027")

    groups = [[
        {"title": "OSFI Guideline E-23 (2025)", "url": "https://osfi.ca/e23-2025", "snippet": "x" * 50, "source_type": "official"},
        {"title": "OSFI Guideline E-23 (2027)", "url": "https://osfi.ca/e23-2027", "snippet": "x" * 40, "source_type": "official"},
        {"title": "Another page", "url": "https://bis.org/p", "snippet": "x" * 30, "source_type": "official"},
    ]]

    chosen = researcher.balanced_sources(groups, 10)
    assert [s["title"] for s in chosen] == ["OSFI Guideline E-23 (2025)", "Another page"]

    later = researcher.balanced_sources(groups, 10, seen_titles={normalize_title("Another page")})
    assert "Another page" not in [s["title"] for s in later]


# ---------- follow-up loop ----------

def test_thin_themes_ignores_other_findings():
    state = {
        "themes": ["Applications", "Risks", "Regulation", theme_mod.OTHER_THEME],
        "approved": FINDINGS[:3],
    }
    assert researcher.thin_themes(state) == ["Risks", "Regulation"]


def test_followup_is_skipped_when_every_theme_is_covered():
    state = {"themes": ["Applications"], "approved": FINDINGS, "followup_rounds": 0, "events": []}
    result = researcher.followup(state)
    assert set(result) == {"events"}
    assert "skipped" in result["events"][-1]["detail"]


def test_followup_is_skipped_when_budget_is_spent(monkeypatch):
    monkeypatch.setattr(researcher, "MAX_FOLLOWUP_ROUNDS", 1)
    state = {"themes": ["Regulation"], "approved": FINDINGS, "followup_rounds": 1, "events": []}
    result = researcher.followup(state)
    assert "budget used" in result["events"][-1]["detail"]


def test_followup_round_adds_new_sources_with_new_ids(monkeypatch):
    calls = []

    class FakeClient:
        def search(self, **options):
            calls.append(options)
            return {"results": [
                {  # already a source: must not be added again
                    "title": "A",
                    "url": "https://www.bis.org/a",
                    "content": "AI agents and credit risk analysis rules. " * 5,
                    "score": 0.9,
                },
                {
                    "title": "New regulation page",
                    "url": "https://www.fsb.org/new",
                    "content": "Regulators address AI agents in credit risk analysis. " * 5,
                    "score": 0.8,
                },
            ]}

    monkeypatch.setattr(researcher, "MAX_FOLLOWUP_ROUNDS", 1)
    monkeypatch.setattr(researcher, "tavily_client", lambda: FakeClient())
    monkeypatch.setattr(
        researcher,
        "structured_model",
        lambda *a, **k: researcher.FollowupQueries(queries=["AI agents credit risk regulation"]),
    )

    state = {
        "question": QUESTION,
        "themes": ["Applications", "Regulation"],
        "approved": FINDINGS[:2],
        "sources": SOURCES,
        "read_source_ids": [1, 2, 3],
        "followup_rounds": 0,
        "followup_queries": ["earlier query"],
        "events": [],
    }

    result = researcher.followup(state)

    assert len(calls) == 1
    assert [s["title"] for s in result["sources"][3:]] == ["New regulation page"]
    assert result["sources"][3]["id"] == 4
    assert result["followup_rounds"] == 1
    assert result["followup_queries"] == ["earlier query", "AI agents credit risk regulation"]

    assert researcher.route_after_followup({**state, **result}) == "extractor"
    assert researcher.route_after_followup(state) == "writer"


def test_followup_queries_fall_back_when_the_model_fails(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(researcher, "structured_model", boom)
    queries = researcher.followup_queries(QUESTION, ["Regulation", "Costs"])
    assert len(queries) == 2
    assert queries[0].endswith("Regulation")


def test_graph_has_the_follow_up_loop():
    edges = {(e.source, e.target) for e in build_graph().get_graph().edges}
    assert ("verifier", "followup") in edges
    assert ("followup", "extractor") in edges
    assert ("followup", "writer") in edges
    assert ("writer", "__end__") in edges


# ---------- writer ----------

def test_clean_sentence():
    assert writer.clean_sentence("banks use agents") == "Banks use agents."
    assert writer.clean_sentence("Banks use agents. 【1】") == "Banks use agents."
    assert writer.clean_sentence("One sentence. Another sentence.") is None
    assert writer.clean_sentence("  ") is None


def test_writer_checks_numbers_and_topics():
    cited = [FINDINGS[0]]
    assert writer.passes_checks("Banks use AI agents to draft credit memos.", cited, QUESTION)
    assert not writer.passes_checks("Banks use AI agents for 40% of credit memos.", cited, QUESTION)
    assert not writer.passes_checks("Banks use agents for analysis of credit risk.", cited, QUESTION)
    assert not writer.passes_checks("Banks use AI agents.", [], QUESTION)


def _fake_writer_model(schema, system, user, tokens=600):
    if "opening answer" in system:
        return WriterOutput(sentences=[
            WrittenSentence(
                text="AI agents draft credit memos and summarise borrower filings for analysts",
                findings=[1, 2],
            ),
            WrittenSentence(  # invented: fails the deterministic checks
                text="AI agents cut credit losses by 30%.",
                findings=[1],
            ),
        ])

    if "SECTION:\nRisks" in user:
        return WriterOutput(sentences=[
            WrittenSentence(  # rejected by the verifier below
                text="Supervisors are worried that agent errors are unsafe.",
                findings=[1, 2],
            ),
        ])

    return WriterOutput(sentences=[
        WrittenSentence(
            text="Banks use AI agents to draft credit memos and summarise borrower filings.",
            findings=[1, 2, 99],  # an out-of-range number is ignored
        ),
    ])


def _fake_verify(report, sources):
    return {
        "claims": [
            {
                "claim_id": c["id"],
                "statement": c["statement"],
                "status": "insufficient_evidence" if "worried" in c["statement"] else "supported",
                "source_ids": c["source_ids"],
                "explanation": "checked",
                "line": c["line"],
                "section": c["section"],
                "section_type": c["section_type"],
            }
            for c in extract_claims(report)
        ],
        "verification_batches": 1,
    }


def test_writer_keeps_only_checked_and_verified_sentences(monkeypatch):
    monkeypatch.setattr(writer, "structured_model", _fake_writer_model)
    monkeypatch.setattr(writer, "verify_report", _fake_verify)

    written, reviews, stats = writer.write(QUESTION, FINDINGS, ["Applications", "Risks"])

    assert stats == {
        "drafted": 4,
        "checked": 3,
        "verified": 2,
        "dropped": {"format": 0, "uncited": 0, "number": 1, "drift": 0},
    }
    assert [s["text"] for s in written["answer"]] == [
        "AI agents draft credit memos and summarise borrower filings for analysts."
    ]
    assert written["answer"][0]["source_ids"] == [1, 2]
    assert "Risks" not in written["sections"]
    assert len(written["used_findings"]) == 2
    assert all(r["status"] == "supported" for r in reviews)


def test_writer_report_is_fully_verified_and_keeps_unused_findings(monkeypatch):
    monkeypatch.setattr(writer, "structured_model", _fake_writer_model)
    monkeypatch.setattr(writer, "verify_report", _fake_verify)

    finding_reviews = [
        {"statement": f["statement"], "source_ids": f["source_ids"], "status": "supported",
         "explanation": "ok", "quote": f["quote"]}
        for f in FINDINGS
    ]

    result = writer.writer({
        "question": QUESTION,
        "themes": ["Applications", "Risks", "Regulation"],
        "approved": FINDINGS,
        "evidence": [],
        "reviews": finding_reviews,
        "rejected": [{"statement": "x"}],
        "sources": SOURCES,
        "followup_rounds": 1,
        "events": [],
    })

    report = result["report"]

    assert report.index("## Answer") < report.index("## Key Findings")
    assert "Banks use AI agents to draft credit memos and summarise borrower filings." in report
    # the Risks paragraph was rejected, so its findings are listed instead
    assert "Supervisors expect human review of agent outputs." in report
    assert "worried" not in report
    # the writer used findings 1 and 2, so they are not repeated
    assert report.count("Banks use AI agents to draft credit memos.") == 0
    assert "## Gaps in the evidence" in report
    assert "Regulation, even after 1 follow-up search round" in report
    assert result["passed"], result["verification"]["issues"]


def test_writer_refuses_to_write_without_verified_findings():
    with pytest.raises(RuntimeError):
        writer.writer({"question": QUESTION, "approved": [], "sources": SOURCES, "events": []})


# ---------- gaps section ----------

def test_evidence_gaps_are_deterministic_and_not_claims():
    gaps = evidence_gaps(["Applications", "Risks", "Costs"], FINDINGS[:3], SOURCES, 2, 0)
    text = "\n".join(gaps)

    assert "No verified evidence was found for: Costs." in text
    assert "Only one verified finding supports: Risks." in text
    assert "2 extracted findings were rejected" in text

    report = assemble_report(QUESTION, FINDINGS[:3], SOURCES, ["Applications", "Risks"], gaps=gaps)
    statements = " ".join(c["statement"] for c in extract_claims(report))
    assert "No verified evidence" not in statements

    reviews = [
        {"statement": f["statement"], "source_ids": f["source_ids"], "status": "supported",
         "explanation": "ok", "quote": ""}
        for f in FINDINGS[:3]
    ]
    assert verify_assembled_report(report, reviews)["passed"]


def test_gaps_note_when_no_regulator_or_academic_source_is_cited():
    gaps = evidence_gaps(["Risks"], FINDINGS[3:], SOURCES, 0, 0)
    assert any("regulator" in line for line in gaps)


# ---------- API ----------

def test_request_length_limits():
    assert ResearchRequest(question="x" * 2000)
    with pytest.raises(ValidationError):
        ResearchRequest(question="too short")
    with pytest.raises(ValidationError):
        ResearchRequest(question="x" * 2001)