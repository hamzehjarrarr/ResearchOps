import researchops.audit as audit
import researchops.extractor as extractor
import researchops.planner as planner
import researchops.report as report_mod
import researchops.researcher as researcher
import researchops.store as store
import researchops.themes as theme_mod
import researchops.verifier as verifier
from researchops.api import ResearchRequest
from researchops.audit import extract_claims, split_sentences
from researchops.extractor import SourceFinding, SourceFindingsOutput
from researchops.graph import build_graph
from researchops.grounding import (
    classify_source,
    is_vendor_only,
    lacks_independent_source,
    numbers_grounded,
    quote_in_excerpt,
)
from researchops.planner import SearchPlan
from researchops.report import (
    assemble_report,
    findings_review_report,
    verify_assembled_report,
)

SOURCES = [
    {
        "id": 1,
        "title": "T1",
        "url": "https://arxiv.org/abs/1",
        "snippet": "x",
        "query": "q",
        "source_type": "academic",
    },
    {
        "id": 2,
        "title": "T2",
        "url": "https://vendor.com/blog",
        "snippet": "x",
        "query": "q",
        "source_type": "commercial",
    },
    {
        "id": 3,
        "title": "T3",
        "url": "https://www.federalreserve.gov/x",
        "snippet": "x",
        "query": "q",
        "source_type": "official",
    },
    {
        "id": 4,
        "title": "T4",
        "url": "https://vendor2.com/post",
        "snippet": "x",
        "query": "q",
        "source_type": "commercial",
    },
]

FINDINGS = [
    {"statement": "Banks use AI agents for fraud triage.", "source_ids": [2], "category": "adoption", "quote": "q"},
    {"statement": "Agents can draft credit memos.", "source_ids": [1], "category": "applications", "quote": "q"},
    {"statement": "Regulators require human oversight.", "source_ids": [3], "category": "governance", "quote": "q"},
    {"statement": "Evidence on agent adoption remains limited.", "source_ids": [4], "category": "limitations", "quote": "q"},
]


# ---------- basics ----------

def test_input_validation():
    assert ResearchRequest(question="How does inflation affect lending?").question


def test_graph_compiles():
    assert build_graph() is not None


# ---------- sentence and claim handling ----------

def test_sentences_split_after_citations():
    text = "First claim. 【1】 Second claim. 【2】【3】 Third claim. 【4】"
    assert split_sentences(text) == [
        "First claim. 【1】",
        "Second claim. 【2】【3】",
        "Third claim. 【4】",
    ]


def test_abbreviations_and_decimals_do_not_split():
    assert len(split_sentences("The U.S. Treasury cited 3.5 percent growth. 【1】")) == 1


def test_each_claim_keeps_its_own_citation():
    claims = extract_claims("## A\n\nOne. 【1】 Two. 【2】 Three. 【3】")
    assert [c["source_ids"] for c in claims] == [[1], [2], [3]]


# ---------- report assembly ----------

def test_report_has_no_uncited_claims():
    report = assemble_report("Line one\nline two", FINDINGS, SOURCES)
    claims = extract_claims(report)
    assert claims
    assert all(c["source_ids"] for c in claims)


def test_empty_sections_are_omitted():
    report = assemble_report("What do agents do?", FINDINGS[:1], SOURCES)
    assert "did not yield" not in report
    assert report.count("### ") == 2  # one theme heading + the evidence block


def test_source_links_show_type_and_evidence_quality():
    report = assemble_report("Question?", FINDINGS, SOURCES)
    assert "Vendor / commercial" in report
    assert "### Evidence quality" in report
    assert "2 of 4" in report


def test_evidence_quality_notes_are_not_claims():
    report = assemble_report("Question?", FINDINGS, SOURCES)
    statements = " ".join(c["statement"] for c in extract_claims(report))
    assert "Sources retrieved" not in statements


def test_review_draft_maps_lines_to_findings():
    draft, line_map = findings_review_report(FINDINGS)
    lines = draft.splitlines()
    for line_no, index in line_map.items():
        assert FINDINGS[index]["statement"] in lines[line_no - 1]
    assert sorted(line_map.values()) == [0, 1, 2, 3]


def _reviews(findings):
    return [
        {
            "statement": f["statement"],
            "source_ids": f["source_ids"],
            "status": "supported",
            "explanation": "Excerpt states this directly.",
            "quote": f.get("quote", ""),
        }
        for f in findings
    ]


def test_assembled_report_verifies_and_keeps_reviewer_explanation():
    report = assemble_report("Question here?", FINDINGS, SOURCES)
    result = verify_assembled_report(report, _reviews(FINDINGS))
    assert result["passed"], result["issues"]
    assert all(c["explanation"] == "Excerpt states this directly." for c in result["claims"])
    assert all(c["quote"] == "q" for c in result["claims"])


def test_unreviewed_statement_is_flagged():
    report = assemble_report("Question here?", FINDINGS, SOURCES)
    assert not verify_assembled_report(report, [])["passed"]


def test_statement_with_wrong_citation_is_flagged():
    reviews = _reviews(FINDINGS)
    reviews[0] = {**reviews[0], "source_ids": [1]}  # reviewed against a different source
    report = assemble_report("Question here?", FINDINGS, SOURCES)
    assert not verify_assembled_report(report, reviews)["passed"]


# ---------- grounding ----------

def test_classify_source():
    assert classify_source("https://arxiv.org/abs/1") == "academic"
    assert classify_source("https://www.federalreserve.gov/x") == "official"
    assert classify_source("https://reports.weforum.org/docs/x.pdf") == "institution"
    assert classify_source("https://rooseveltinstitute.org/p") == "institution"
    assert classify_source("https://www.reuters.com/a") == "news"
    assert classify_source("https://www.kognitos.com/blog/ai") == "commercial"
    assert classify_source("https://fin.ai/learn/evaluate-ai-agent-compliance") == "commercial"
    assert classify_source("https://www.csis.org/analysis/x") == "institution"
    assert classify_source("https://www.intereconomics.eu/contents/x.html") == "institution"
    assert classify_source("https://finance.yahoo.com/news/x.html") == "news"
    assert classify_source("https://www.whitecase.com/insight-alert/csbs-ai") == "other"
    assert classify_source("https://www.precedenceresearch.com/ai-agents-market") == "other"


EXCERPT = (
    "Banks adopting third-party agents often lack transparency into model "
    "updates, memory mechanics, or the agent's inherent logic. "
    "AI spending is projected to reach $97 billion by 2027."
)


def test_exact_quote_is_accepted():
    assert quote_in_excerpt(
        "Banks adopting third-party agents often lack transparency into model updates",
        EXCERPT,
    )


def test_quote_tolerates_spacing_and_punctuation():
    assert quote_in_excerpt(
        "banks adopting third‑party agents often lack transparency into modelupdates",
        EXCERPT,
    )


def test_paraphrase_is_rejected():
    assert not quote_in_excerpt(
        "Banks that use outside agents have poor visibility into how they change",
        EXCERPT,
    )


def test_short_quote_is_rejected():
    assert not quote_in_excerpt("Banks adopting", EXCERPT)


def test_elided_quote_needs_every_part_present():
    assert quote_in_excerpt(
        "Banks adopting third-party agents ... projected to reach $97 billion by 2027",
        EXCERPT,
    )
    assert not quote_in_excerpt(
        "Banks adopting third-party agents ... collapsed completely overnight",
        EXCERPT,
    )


def test_numbers_must_appear_in_excerpt():
    assert numbers_grounded("AI spending will reach $97 billion by 2027.", EXCERPT)
    assert not numbers_grounded("AI spending will reach $120 billion by 2027.", EXCERPT)


def test_vendor_only_flag():
    types = {1: "academic", 2: "commercial", 4: "commercial"}
    assert is_vendor_only([2], types)
    assert is_vendor_only([2, 4], types)
    assert not is_vendor_only([1, 2], types)
    assert not is_vendor_only([], types)


def test_unrecognised_sites_do_not_count_as_independent():
    types = {1: "academic", 2: "commercial", 3: "other", 4: "news"}
    assert lacks_independent_source([3], types)
    assert lacks_independent_source([2, 3], types)
    assert not lacks_independent_source([3, 4], types)
    assert not lacks_independent_source([1], types)


# ---------- extractor (model mocked) ----------

def test_extractor_keeps_only_grounded_findings(monkeypatch):
    excerpt = (
        "Banks adopting third-party agents often lack transparency into model "
        "updates, memory mechanics, or the agent's inherent logic for using tools. "
        "Financial institutions' AI expenditure is projected to reach $97 billion by 2027."
    )
    state = {
        "question": "How are AI agents used in banks?",
        "sources": [
            {"id": 1, "title": "A", "url": "https://a.com", "snippet": excerpt, "source_type": "commercial"},
            {"id": 2, "title": "B", "url": "https://b.com", "snippet": "short", "source_type": "commercial"},
        ],
        "events": [],
    }

    output = SourceFindingsOutput(findings=[
        SourceFinding(  # good
            statement="Banks often lack transparency into third-party agents' model updates",
            quote="Banks adopting third-party agents often lack transparency into model updates",
            category="operational_risk", relevance=5,
        ),
        SourceFinding(  # invented quote
            statement="Banks fully trust their agents.",
            quote="Banks fully trust the agents they adopt from every vendor",
            category="adoption", relevance=5,
        ),
        SourceFinding(  # real quote, wrong number
            statement="AI spending will reach $120 billion by 2027.",
            quote="AI expenditure is projected to reach $97 billion by 2027",
            category="adoption", relevance=5,
        ),
    ])

    calls = []

    def fake(schema, system, user, tokens=1400):
        calls.append(user)
        return output

    monkeypatch.setattr(extractor, "structured_model", fake)

    result = extractor.extractor(state)

    assert len(calls) == 1  # the short excerpt is skipped without a model call
    assert len(result["findings"]) == 1
    finding = result["findings"][0]
    assert finding["source_ids"] == [1]
    assert finding["statement"].endswith(".")
    assert finding["quote"].startswith("Banks adopting")
    assert finding["source_type"] == "commercial"


def test_extractor_returns_nothing_when_nothing_is_grounded(monkeypatch):
    state = {
        "question": "Q?",
        "sources": [{"id": 1, "title": "A", "url": "https://a.com", "snippet": "x" * 400, "source_type": "commercial"}],
        "events": [],
    }

    monkeypatch.setattr(
        extractor,
        "structured_model",
        lambda *a, **k: SourceFindingsOutput(findings=[
            SourceFinding(statement="Made up.", quote="not in the excerpt at all, none of it", category="adoption", relevance=5)
        ]),
    )

    result = extractor.extractor(state)
    assert result["findings"] == []
    assert result["read_source_ids"] == [1]
    assert "1 quote" in result["events"][-1]["detail"]


# ---------- store progress ----------

def test_store_progress_then_finish(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DB_PATH", str(tmp_path / "t.db"))
    run_id = store.start("A question that is long enough")

    store.progress(run_id, [{"agent": "planner", "duration_ms": 5, "detail": "x"}])
    run = store.get(run_id)
    assert run["status"] == "running"
    assert run["result"]["events"][0]["agent"] == "planner"

    store.finish(run_id, {"report": "done"})
    run = store.get(run_id)
    assert run["status"] == "completed"
    assert run["result"] == {"report": "done"}

    store.progress(run_id, [{"agent": "late"}])  # must not overwrite a finished run
    assert store.get(run_id)["result"] == {"report": "done"}


# ---------- scope, references and decomposed verification ----------

def test_reference_list_entries_are_detected():
    from researchops.grounding import looks_like_reference

    assert looks_like_reference(
        "Human player outwits Freysa AI agent in $47,000 crypto challenge; 2024 Nov [cited 2025 Sep18]"
    )
    assert looks_like_reference("Available from: https://example.org/report")
    assert looks_like_reference("doi:10.1000/xyz123")
    assert not looks_like_reference(
        "Banks adopting third-party agents often lack transparency into model updates"
    )


def test_extractor_rejects_bibliography_quotes(monkeypatch):
    excerpt = (
        "References. Human player outwits Freysa AI agent in $47,000 crypto challenge; "
        "2024 Nov [cited 2025 Sep18]. Banks adopting third-party agents often lack "
        "transparency into model updates, memory mechanics, or inherent logic."
    )
    state = {
        "question": "Q?",
        "sources": [{"id": 1, "title": "A", "url": "https://a.com", "snippet": excerpt, "source_type": "academic"}],
        "events": [],
    }

    monkeypatch.setattr(
        extractor,
        "structured_model",
        lambda *a, **k: SourceFindingsOutput(findings=[
            SourceFinding(
                statement="AI agents are used in financial institutions for crypto trading.",
                quote="Human player outwits Freysa AI agent in $47,000 crypto challenge; 2024 Nov [cited 2025 Sep18]",
                category="adoption", relevance=5,
            ),
            SourceFinding(
                statement="Banks often lack transparency into third-party agents' model updates.",
                quote="Banks adopting third-party agents often lack transparency into model updates",
                category="operational_risk", relevance=5,
            ),
        ]),
    )

    result = extractor.extractor(state)

    assert len(result["findings"]) == 1
    assert "transparency" in result["findings"][0]["statement"]
    assert "1 reference" in result["events"][-1]["detail"]


def test_independent_findings_come_first():
    findings = [
        {"statement": "Vendor claim.", "source_ids": [2], "category": "adoption", "source_type": "commercial"},
        {"statement": "Regulator claim.", "source_ids": [3], "category": "adoption", "source_type": "official"},
    ]
    groups = report_mod.grouped_findings(findings)
    assert groups["adoption"][0]["statement"] == "Regulator claim."
    assert report_mod.select_summary(findings, limit=1)[0]["statement"] == "Regulator claim."


def test_report_has_no_conclusion_section():
    report = assemble_report("Question?", FINDINGS, SOURCES)
    assert "## Conclusion" not in report


def test_cloud_security_alliance_is_an_industry_body():
    assert classify_source("https://labs.cloudsecurityalliance.org/research/x") == "institution"


def _batch(statement, excerpt):
    claim = {
        "id": 1,
        "statement": statement,
        "source_ids": [1],
        "line": 3,
        "section": "Findings",
        "section_type": "factual",
    }
    sources = [{"id": 1, "title": "T", "url": "https://a.com", "snippet": excerpt}]
    return [claim], sources


def test_supported_claim_with_unsupported_parts_is_downgraded(monkeypatch):
    batch, sources = _batch(
        "AI agents are used in financial institutions for crypto trading.",
        "A human player outwitted an AI agent in a $47,000 crypto challenge.",
    )

    monkeypatch.setattr(
        audit,
        "invoke_model",
        lambda *a, **k: audit.BatchAssessment(assessments=[
            audit.ClaimAssessment(
                claim_id=1,
                components=["agents used in financial institutions", "crypto trading"],
                unsupported_components=["agents used in financial institutions"],
                status="supported",
                explanation="The excerpt mentions a crypto challenge.",
            )
        ]),
    )

    result = audit.assess_batch(batch, sources)[0]

    assert result["status"] == "insufficient_evidence"
    assert "agents used in financial institutions" in result["explanation"]


def test_fully_supported_claim_stays_supported(monkeypatch):
    batch, sources = _batch(
        "A human player outwitted an AI agent in a $47,000 crypto challenge.",
        "A human player outwitted an AI agent in a $47,000 crypto challenge.",
    )

    monkeypatch.setattr(
        audit,
        "invoke_model",
        lambda *a, **k: audit.BatchAssessment(assessments=[
            audit.ClaimAssessment(
                claim_id=1,
                components=["human outwitted agent"],
                unsupported_components=[],
                status="supported",
                explanation="Stated directly.",
            )
        ]),
    )

    assert audit.assess_batch(batch, sources)[0]["status"] == "supported"


def test_assessment_schema_is_valid_for_strict_output():
    from researchops.llm import strict_schema

    strict_schema(audit.BatchAssessment)


# ---------- researcher (Tavily mocked) ----------

QUESTION = "How are AI agents being adopted by financial institutions, and what are the main risks?"

ON_TOPIC = "Financial institutions are adopting AI agents and face new risks from them. "


def test_researcher_runs_authoritative_searches_first(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, api_key=None):
            pass

        def search(self, **options):
            calls.append(options)
            if options.get("include_domains"):
                return {"results": [{
                    "title": "Official report",
                    "url": "https://www.bis.org/report",
                    "content": ON_TOPIC * 5,
                    "score": 0.8,
                }]}
            return {"results": [{
                "title": "Vendor blog",
                "url": "https://vendor.com/blog",
                "content": ON_TOPIC * 5,
                "score": 0.7,
            }]}

    monkeypatch.setenv("TAVILY_API_KEY", "test")
    monkeypatch.setattr("tavily.TavilyClient", FakeClient)

    result = researcher.researcher({"question": QUESTION, "queries": ["q1", "q2", "q3"], "events": []})

    restricted = [c for c in calls if c.get("include_domains")]
    assert len(restricted) == 2  # first two queries only
    assert all("bis.org" in c["include_domains"] for c in restricted)

    types = [s["source_type"] for s in result["sources"]]
    assert types[0] == "official"  # authoritative results get first pick
    assert "commercial" in types
    assert "recognised independent" in result["events"][-1]["detail"]
    assert result["sources"][0]["score"] == 0.8


def test_researcher_drops_off_topic_results(monkeypatch):
    class FakeClient:
        def __init__(self, api_key=None):
            pass

        def search(self, **options):
            if options.get("include_domains"):
                return {"results": [
                    {  # off-topic page on a trusted domain
                        "title": "Mercury",
                        "url": "https://www.who.int/news-room/fact-sheets/detail/mercury-and-health",
                        "content": "Mercury is a naturally occurring element found in air, water and soil. " * 4,
                        "score": 0.9,
                    },
                    {  # on-topic but low search score
                        "title": "Weak match",
                        "url": "https://www.bis.org/weak",
                        "content": ON_TOPIC * 4,
                        "score": 0.05,
                    },
                    {
                        "title": "Good report",
                        "url": "https://www.fsb.org/good",
                        "content": ON_TOPIC * 4,
                        "score": 0.6,
                    },
                ]}
            return {"results": []}

    monkeypatch.setenv("TAVILY_API_KEY", "test")
    monkeypatch.setattr("tavily.TavilyClient", FakeClient)

    result = researcher.researcher({"question": QUESTION, "queries": ["q1"], "events": []})

    assert [s["title"] for s in result["sources"]] == ["Good report"]
    assert "2 off-topic dropped" in result["events"][-1]["detail"]


def test_researcher_keeps_results_if_everything_is_filtered(monkeypatch):
    class FakeClient:
        def __init__(self, api_key=None):
            pass

        def search(self, **options):
            return {"results": [{
                "title": "Unrelated",
                "url": "https://example.com/a",
                "content": "Completely different subject matter entirely. " * 5,
                "score": 0.9,
            }]}

    monkeypatch.setenv("TAVILY_API_KEY", "test")
    monkeypatch.setattr("tavily.TavilyClient", FakeClient)

    result = researcher.researcher({"question": QUESTION, "queries": ["q1"], "events": []})

    assert len(result["sources"]) == 1  # filter never leaves the run empty


def test_is_on_topic():
    from researchops.grounding import is_on_topic

    assert is_on_topic(QUESTION, "Banks and financial institutions are adopting agentic AI, raising new risks.")
    assert not is_on_topic(QUESTION, "Mercury is a naturally occurring element found in air, water and soil.")
    assert is_on_topic("", "anything at all")


# ---------- rejected findings are kept for display ----------

def test_verifier_returns_rejected_findings_with_reasons(monkeypatch):
    findings = [
        {"statement": "Regulators require human oversight.", "source_ids": [1], "category": "governance", "quote": "q1", "source_type": "official"},
        {"statement": "AI agents trade crypto at banks.", "source_ids": [2], "category": "adoption", "quote": "q2", "source_type": "commercial"},
    ]
    sources = [
        {"id": 1, "title": "A", "url": "https://a.gov", "snippet": "x", "source_type": "official"},
        {"id": 2, "title": "B", "url": "https://b.com", "snippet": "y", "source_type": "commercial"},
    ]

    def fake_verify(report, srcs):
        claims = []
        for c in extract_claims(report):
            ok = "oversight" in c["statement"]
            claims.append({
                "claim_id": c["id"],
                "statement": c["statement"],
                "status": "supported" if ok else "insufficient_evidence",
                "source_ids": c["source_ids"],
                "explanation": "ok" if ok else "Not established by the cited excerpts: banks",
                "line": c["line"],
                "section": c["section"],
                "section_type": c["section_type"],
            })
        return {"claims": claims, "verification_batches": 1}

    monkeypatch.setattr(verifier, "verify_report", fake_verify)

    result = verifier.verifier({"question": "Q?", "findings": findings, "sources": sources, "events": []})

    assert len(result["evidence"]) == 1
    assert len(result["rejected"]) == 1
    rejected = result["rejected"][0]
    assert "crypto" in rejected["statement"]
    assert rejected["review_status"] == "insufficient_evidence"
    assert "banks" in rejected["review_explanation"]
    assert [f["statement"] for f in result["approved"]] == ["Regulators require human oversight."]
    assert result["reviews"][0]["status"] == "supported"


# ---------- dynamic themes, relevance and ranking ----------

def test_clean_themes_removes_reserved_and_duplicate_names():
    themes = theme_mod.clean_themes([
        "References", "Executive Summary", "US framework",
        "**EU framework**", "us framework", "  ",
    ])
    assert themes == ["US framework", "EU framework"]


def test_planner_returns_themes_and_falls_back_to_defaults(monkeypatch):
    monkeypatch.setattr(
        planner,
        "structured_model",
        lambda *a, **k: SearchPlan(
            subject="stablecoins",
            queries=["q one", "q two"],
            themes=["US rules", "EU rules", "Financial stability"],
        ),
    )
    result = planner.planner({"question": "Q?", "events": []})
    assert result["themes"] == ["US rules", "EU rules", "Financial stability"]
    assert result["subject"] == "stablecoins"

    monkeypatch.setattr(
        planner,
        "structured_model",
        lambda *a, **k: SearchPlan(subject="", queries=["q"], themes=["References"]),
    )
    result = planner.planner({"question": "Q?", "events": []})
    assert result["themes"] == theme_mod.DEFAULT_THEMES


def test_normalize_theme():
    themes = ["US framework", "EU framework"]
    assert theme_mod.normalize_theme("us framework", themes) == "US framework"
    assert theme_mod.normalize_theme("EU Frameworks", themes) == "EU framework"
    assert theme_mod.normalize_theme("something unrelated", themes) == theme_mod.OTHER_THEME


def test_report_sections_follow_planned_theme_order():
    findings = [
        {"statement": "US claim.", "source_ids": [3], "category": "US framework"},
        {"statement": "EU claim.", "source_ids": [1], "category": "EU framework"},
    ]
    report = assemble_report("Q?", findings, SOURCES, ["EU framework", "US framework"])
    assert report.index("### EU framework") < report.index("### US framework")
    assert "## Key Findings" in report


def test_summary_takes_best_finding_per_theme():
    findings = [
        {"statement": "Vendor A.", "source_ids": [2], "category": "T1", "source_type": "commercial", "relevance": 5},
        {"statement": "Official A.", "source_ids": [3], "category": "T1", "source_type": "official", "relevance": 3},
        {"statement": "Official B.", "source_ids": [3], "category": "T2", "source_type": "official", "relevance": 4},
        {"statement": "Official C.", "source_ids": [3], "category": "T2", "source_type": "official", "relevance": 5},
    ]
    chosen = [f["statement"] for f in report_mod.select_summary(findings, limit=4)]
    assert chosen == ["Official C.", "Official A."]  # one per theme, best first


def test_extractor_drops_low_relevance_findings(monkeypatch):
    excerpt = (
        "Central bank digital currencies are issued by central banks. "
        "Stablecoin issuers must hold reserves equal to the value of tokens in circulation."
    )
    state = {
        "question": "How are regulators approaching stablecoins?",
        "themes": ["Reserve rules"],
        "sources": [{"id": 1, "title": "A", "url": "https://a.com", "snippet": excerpt * 2, "source_type": "official"}],
        "events": [],
    }

    monkeypatch.setattr(
        extractor,
        "structured_model",
        lambda *a, **k: SourceFindingsOutput(findings=[
            SourceFinding(
                statement="Central banks issue digital currencies.",
                quote="Central bank digital currencies are issued by central banks",
                category="Reserve rules",
                relevance=2,
            ),
            SourceFinding(
                statement="Stablecoin issuers must hold reserves equal to tokens in circulation.",
                quote="Stablecoin issuers must hold reserves equal to the value of tokens in circulation",
                category="reserve rules",
                relevance=5,
            ),
        ]),
    )

    result = extractor.extractor(state)

    assert len(result["findings"]) == 1
    assert result["findings"][0]["category"] == "Reserve rules"
    assert result["findings"][0]["relevance"] == 5
    assert "1 relevance" in result["events"][-1]["detail"]