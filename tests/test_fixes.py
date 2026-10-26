"""Fixes from the first real run with qwen3:4b-instruct."""

import researchops.audit as audit
import researchops.extractor as extractor
import researchops.writer as writer
from researchops.audit import extract_claims, split_sentences
from researchops.extractor import SourceFinding, SourceFindingsOutput
from researchops.planner import clean_subject
from researchops.report import assemble_report, evidence_gaps, subject_coverage


def test_words_ending_in_ms_do_not_block_sentence_splits():
    text = (
        "AI predicts defaults using machine learning algorithms. 【9】 "
        "Generative AI monitors signals. 【18】 "
        "Controls fall short in the era of LLMs. 【16】 The result was a boost. 【17】"
    )
    assert len(split_sentences(text)) == 4


def test_real_abbreviations_still_do_not_split():
    assert len(split_sentences("Ms. Smith met the U.S. Treasury, e.g. its staff. 【1】")) == 1


def _claims(*ids):
    return [
        {"id": i, "statement": f"Claim {i}.", "source_ids": [1], "line": i,
         "section": "Findings", "section_type": "factual"}
        for i in ids
    ]


def test_claim_skipped_in_a_batch_is_reviewed_alone(monkeypatch):
    calls = []

    def fake(system, context, **kwargs):
        ids = [int(x) for x in __import__("re").findall(r"CLAIM ID: (\d+)", context)]
        calls.append(ids)
        return audit.BatchAssessment(assessments=[
            audit.ClaimAssessment(claim_id=ids[0], components=["x"],
                                  unsupported_components=[], status="supported",
                                  explanation="ok")
        ])

    monkeypatch.setattr(audit, "invoke_model", fake)
    sources = [{"id": 1, "title": "T", "snippet": "text"}]

    results = audit.assess_with_retry(_claims(1, 2), sources)

    assert calls == [[1, 2], [2]]
    assert [r["status"] for r in results] == ["supported", "supported"]


def test_statements_that_need_their_neighbours_are_dropped(monkeypatch):
    excerpt = (
        "Our results are visualized through clear, interpretable graphical representations "
        "for lenders. Banks use AI agents to draft credit memos for analysts every day. " * 3
    )
    state = {
        "question": "How are AI agents changing credit risk analysis?",
        "sources": [{"id": 1, "title": "A", "url": "https://a.com", "snippet": excerpt, "source_type": "academic"}],
        "events": [],
    }

    monkeypatch.setattr(
        extractor,
        "structured_model",
        lambda *a, **k: SourceFindingsOutput(findings=[
            SourceFinding(
                statement="Our results are visualized through clear graphical representations.",
                quote="Our results are visualized through clear, interpretable graphical representations",
                category="x", relevance=4,
            ),
            SourceFinding(
                statement="Banks use AI agents to draft credit memos for analysts.",
                quote="Banks use AI agents to draft credit memos for analysts every day",
                category="x", relevance=5,
            ),
        ]),
    )

    result = extractor.extractor(state)

    assert [f["statement"] for f in result["findings"]] == [
        "Banks use AI agents to draft credit memos for analysts."
    ]
    assert "1 context" in result["events"][-1]["detail"]


# ---------- subject coverage in the gaps section ----------

def _finding(statement):
    return {"statement": statement, "source_ids": [1], "category": "T",
            "relevance": 4, "source_type": "official"}


SOURCES = [{"id": 1, "title": "A", "url": "https://bis.org/a", "snippet": "x",
            "source_type": "official"}]

MOSTLY_GENERAL = [
    _finding("AI agents draft credit memos."),
    _finding("Machine learning predicts loan defaults."),
    _finding("AI systems monitor credit risk in real time."),
    _finding("Regulators expect human oversight of AI models."),
]


def test_subject_coverage_counts_findings_about_the_subject():
    assert subject_coverage("AI agents", MOSTLY_GENERAL) == (1, 4)
    assert subject_coverage("agentic AI", [_finding("Agents act alone.")]) == (1, 1)
    assert subject_coverage("AI", MOSTLY_GENERAL) is None  # too short to match on
    assert subject_coverage("", MOSTLY_GENERAL) is None


def test_gaps_say_when_little_evidence_is_about_the_subject():
    gaps = evidence_gaps(["T"], MOSTLY_GENERAL, SOURCES, 0, 0, subject="AI agents")
    assert gaps[0].startswith("- Only 1 of 4 verified findings are specifically about AI agents")

    report = assemble_report("Q?", MOSTLY_GENERAL, SOURCES, ["T"], gaps=gaps)
    statements = " ".join(c["statement"] for c in extract_claims(report))
    assert "specifically about" not in statements  # a note, not a claim


def test_no_coverage_note_when_most_findings_are_on_subject():
    findings = MOSTLY_GENERAL[:1] + [_finding("AI agents summarise filings.")]
    gaps = evidence_gaps(["T"], findings, SOURCES, 0, 0, subject="AI agents")
    assert not any("specifically about" in line for line in gaps)


def test_clean_subject():
    assert clean_subject(" AI   agents. ") == "AI agents"
    assert clean_subject("a very long answer that is clearly not a short subject at all") == ""


# ---------- writer: why sentences are dropped ----------

def test_writer_reports_why_each_sentence_failed():
    cited = [{"statement": "Banks use AI agents to draft credit memos.", "source_ids": [1],
              "quote": "agents draft credit memos"}]
    question = "How are AI agents changing credit risk analysis?"

    assert writer.check_failure("Banks use AI agents to draft credit memos.", cited, question) is None
    assert writer.check_failure("Banks use AI agents for 40% of memos.", cited, question) == "number"
    assert writer.check_failure("Agents change credit risk analysis.", cited, question) == "drift"
    assert writer.check_failure("Banks use AI agents.", [], question) == "uncited"


# ---------- writer drift check ignores the question's framing verbs ----------

STABLECOIN_Q = "How are central banks and regulators approaching the regulation of stablecoins?"

# Real findings from the stablecoins run (sources 12 and 3).
RESERVES = {"statement": "Payment stablecoins must be backed on a one-for-one basis by "
            "high-quality, highly liquid reserve assets such as cash, deposits at insured "
            "depository institutions, balances at Federal Reserve Banks, and short-term "
            "U.S. Treasury securities.", "source_ids": [12], "quote": ""}

DUAL_BANKING = {"statement": "Depository institution stablecoin issuers would be regulated "
                "by their primary Federal banking regulator.", "source_ids": [3], "quote": ""}


def test_writer_may_answer_in_the_questions_own_words():
    sentence = ("Regulators are approaching stablecoins by supervising issuers through "
                "their primary federal banking regulator.")
    assert writer.check_failure(sentence, [DUAL_BANKING], STABLECOIN_Q) is None

    # Before this fix the same sentence was dropped for "approaching" alone.
    from researchops.grounding import added_question_terms

    evidence = DUAL_BANKING["statement"]
    assert added_question_terms(sentence, evidence, STABLECOIN_Q) == ["appro"]

    credit_q = "How are AI agents changing credit risk analysis?"
    memo = {"statement": "AI agents draft credit memos and flag early warning signals.",
            "source_ids": [1], "quote": ""}
    assert writer.check_failure("AI agents are changing credit work by drafting credit memos.",
                                [memo], credit_q) is None


def test_writer_drift_check_still_catches_new_topics():
    # "central banks" is a topic of the question but not of these findings.
    sentence = "Central banks require one-for-one reserve backing for stablecoins."
    assert writer.check_failure(sentence, [RESERVES], STABLECOIN_Q) == "drift"

    credit_q = "How are AI agents changing credit risk analysis?"
    ml = {"statement": "A machine-learning engine predicts loan defaults.", "source_ids": [1],
          "quote": ""}
    assert writer.check_failure("AI agents predict loan defaults.", [ml], credit_q) == "drift"


def test_framing_words_are_matched_as_whole_words():
    assert "transactions" in writer.topic_words("How is AI transforming transactions?")
    assert "approaching" not in writer.topic_words(STABLECOIN_Q)


def test_extractor_drift_check_is_unchanged():
    from researchops.grounding import added_question_terms

    excerpt = "Payment stablecoins must be backed one-for-one by reserve assets."
    assert added_question_terms("Regulators are approaching stablecoins with reserve rules.",
                                excerpt, STABLECOIN_Q)