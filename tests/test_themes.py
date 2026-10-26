import researchops.themes as theme_mod
import researchops.writer as writer
from researchops.grounding import classify_source
from researchops.themes import ThemeGroup, ThemeGrouping


def _findings(count):
    return [
        {
            "statement": f"Finding number {i} about stablecoin rules.",
            "source_ids": [1],
            "category": "Overview",
            "quote": "q",
            "source_type": "official",
            "relevance": 4,
        }
        for i in range(1, count + 1)
    ]


def _grouping(*groups):
    return ThemeGrouping(themes=[
        ThemeGroup(name=name, finding_numbers=numbers) for name, numbers in groups
    ])


def test_staff_research_on_official_domains_is_classed_as_research():
    assert classify_source("https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp3199.en.pdf") == "academic"
    assert classify_source("https://www.bis.org/publications/working-paper-905-stablecoins") == "academic"
    assert classify_source(
        "https://www.federalreserve.gov/econres/notes/feds-notes/stablecoins-in-2025-20260408.html"
    ) == "academic"
    # policy documents and speeches stay official
    assert classify_source("https://www.fsb.org/uploads/P161025-1.pdf") == "official"
    assert classify_source("https://www.ecb.europa.eu/press/key/date/2026/html/ecb.sp260601.en.html") == "official"


def test_findings_are_regrouped_by_what_was_found(monkeypatch):
    findings = _findings(8)

    monkeypatch.setattr(
        theme_mod,
        "structured_model",
        lambda *a, **k: _grouping(
            ("Reserve rules", [1, 2, 3]),
            ("Cross-border coordination", [4, 5]),
            ("Financial stability risks", [6, 7, 8]),
        ),
    )

    regrouped, themes = theme_mod.group_findings("Q?", findings)

    assert themes == ["Reserve rules", "Cross-border coordination", "Financial stability risks"]
    assert [f["category"] for f in regrouped[:4]] == [
        "Reserve rules", "Reserve rules", "Reserve rules", "Cross-border coordination",
    ]
    assert findings[0]["category"] == "Overview"  # inputs are not mutated


def test_unassigned_findings_go_to_other_theme(monkeypatch):
    findings = _findings(8)

    monkeypatch.setattr(
        theme_mod,
        "structured_model",
        lambda *a, **k: _grouping(("Reserve rules", [1, 2, 3, 4]), ("Risks", [5, 6, 7])),
    )

    regrouped, themes = theme_mod.group_findings("Q?", findings)

    assert regrouped[7]["category"] == theme_mod.OTHER_THEME
    assert themes[-1] == theme_mod.OTHER_THEME


def test_unusable_grouping_falls_back(monkeypatch):
    findings = _findings(8)

    # only one theme
    monkeypatch.setattr(theme_mod, "structured_model", lambda *a, **k: _grouping(("All of it", [1, 2, 3, 4, 5, 6, 7, 8])))
    assert theme_mod.group_findings("Q?", findings) == (findings, None)

    # too few findings assigned
    monkeypatch.setattr(theme_mod, "structured_model", lambda *a, **k: _grouping(("A", [1, 2]), ("B", [3])))
    assert theme_mod.group_findings("Q?", findings) == (findings, None)

    # model failure
    def boom(*a, **k):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(theme_mod, "structured_model", boom)
    assert theme_mod.group_findings("Q?", findings) == (findings, None)


def test_small_runs_skip_grouping_without_calling_the_model(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("model must not be called")

    monkeypatch.setattr(theme_mod, "structured_model", boom)
    findings = _findings(3)
    assert theme_mod.group_findings("Q?", findings) == (findings, None)


def test_writer_builds_report_from_regrouped_themes(monkeypatch):
    findings = _findings(6)
    sources = [{"id": 1, "title": "A", "url": "https://a.gov", "snippet": "x", "source_type": "official"}]
    reviews = [
        {"statement": f["statement"], "source_ids": f["source_ids"], "status": "supported",
         "explanation": "ok", "quote": "q"}
        for f in findings
    ]
    evidence = [dict(f) for f in findings]

    monkeypatch.setattr(writer, "ENABLE_WRITER", False)
    monkeypatch.setattr(
        theme_mod,
        "structured_model",
        lambda *a, **k: _grouping(("Reserve rules", [1, 2, 3]), ("Risks to stability", [4, 5, 6])),
    )

    result = writer.writer({
        "question": "Q?",
        "themes": ["Overview"],
        "approved": findings,
        "evidence": evidence,
        "reviews": reviews,
        "rejected": [],
        "sources": sources,
        "events": [],
    })

    assert result["themes"] == ["Reserve rules", "Risks to stability"]
    assert "### Reserve rules" in result["report"]
    assert "### Risks to stability" in result["report"]
    assert "### Overview" not in result["report"]
    assert {e["category"] for e in evidence} == {"Reserve rules", "Risks to stability"}
    assert result["passed"], result["verification"]["issues"]
    assert "writer off" in result["events"][-1]["detail"]