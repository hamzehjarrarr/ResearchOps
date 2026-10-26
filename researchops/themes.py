"""Report themes: planning, normalising and regrouping by content."""

import difflib
import re

from pydantic import BaseModel, ConfigDict

from researchops.state import structured_model

DEFAULT_THEMES = [
    "Overview",
    "Evidence and examples",
    "Risks and concerns",
    "Policy and governance",
]


OTHER_THEME = "Other findings"


RESERVED_THEMES = {
    "executive summary",
    "key findings",
    "source links",
    "references",
    "bibliography",
    "research report",
    "question",
}


class ThemeGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    finding_numbers: list[int]


class ThemeGrouping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    themes: list[ThemeGroup]


def clean_themes(values, limit=5):
    """Safe, unique theme names usable as report headings."""
    themes = []
    seen = set()

    for value in values or []:
        name = re.sub(
            r"[#*_`>\[\]]",
            "",
            " ".join(str(value).split()),
        ).strip(" .:-")[:60]

        key = name.lower()

        if not name or key in RESERVED_THEMES or key in seen:
            continue

        seen.add(key)
        themes.append(name)

        if len(themes) >= limit:
            break

    return themes


def normalize_theme(value, themes):
    """Map the model's chosen theme onto one of the planned themes."""
    text = " ".join(str(value).split()).strip().lower()

    lookup = {theme.lower(): theme for theme in themes}

    if text in lookup:
        return lookup[text]

    close = difflib.get_close_matches(
        text,
        list(lookup),
        n=1,
        cutoff=0.6,
    )

    return lookup[close[0]] if close else OTHER_THEME


def theme_order(findings, themes=None):
    """Planned themes first (in planned order), then any others."""
    present = []

    for item in findings:
        if item["category"] not in present:
            present.append(item["category"])

    ordered = [t for t in (themes or []) if t in present]
    ordered += [t for t in present if t not in ordered]

    return ordered


MIN_FINDINGS_TO_GROUP = 6


GROUPING_SYSTEM = """
You organise verified research findings into report sections.

You are given a research question and numbered findings. Group
the findings into 3 to 5 themes.

Rules:
- Theme names are 2 to 5 words and describe what the findings
  in that theme say, for example "Reserve and redemption rules"
  or "Cross-border coordination". Do not use generic names such
  as "Overview", "Key findings" or "Other".
- Every finding number must appear in exactly one theme.
- Each theme should hold at least 2 findings, and no theme
  should hold more than about half of all findings.
- Do not rewrite, merge or add findings. Use numbers only.

Return structured output only.
"""


def group_findings(question, findings):
    """
    Organise verified findings into themes based on what was found.

    Returns (findings_with_new_categories, ordered_theme_names), or
    (findings, None) when grouping is skipped or the model's answer
    is not usable, in which case the planner's themes stay in effect.
    """

    if len(findings) < MIN_FINDINGS_TO_GROUP:
        return findings, None

    numbered = "\n".join(
        f"{number}. {item['statement']}"
        for number, item in enumerate(findings, 1)
    )

    try:
        response = structured_model(
            ThemeGrouping,
            GROUPING_SYSTEM,
            f"QUESTION:\n{question}\n\nFINDINGS:\n{numbered}",
            tokens=900,
        )

    except Exception as exc:
        print(
            "Theme grouping skipped:",
            type(exc).__name__,
            str(exc)[:150],
        )
        return findings, None

    names = {}
    assigned = {}

    for group in response.themes:
        cleaned = clean_themes([group.name], limit=1)

        if not cleaned:
            continue

        name = names.setdefault(
            cleaned[0].lower(),
            cleaned[0],
        )

        for number in group.finding_numbers:
            if (
                1 <= number <= len(findings)
                and number not in assigned
            ):
                assigned[number] = name

    used = list(dict.fromkeys(assigned.values()))

    # Reject unusable output: one theme, or too many findings left over.
    if len(used) < 2 or len(assigned) < 0.7 * len(findings):
        return findings, None

    regrouped = []

    for number, item in enumerate(findings, 1):
        updated = dict(item)
        updated["category"] = assigned.get(number, OTHER_THEME)
        regrouped.append(updated)

    if len(assigned) < len(findings):
        used.append(OTHER_THEME)

    return regrouped, used