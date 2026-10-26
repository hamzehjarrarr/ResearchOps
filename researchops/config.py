"""Settings, read from environment variables (or a local .env file)."""

import os

from dotenv import load_dotenv

load_dotenv()


def number(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
        return max(minimum, min(maximum, value))
    except (ValueError, TypeError):
        return default


def fraction(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
        return max(minimum, min(maximum, value))
    except (ValueError, TypeError):
        return default


def flag(name: str, default: bool) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


# ------------------------------------------------------------
# Local model (Ollama)
# ------------------------------------------------------------

MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
).rstrip("/")

# 8K tokens is enough for every call this app makes and keeps
# memory use low on a laptop.
OLLAMA_NUM_CTX = number("OLLAMA_NUM_CTX", 8192, 2048, 131072)
OLLAMA_TIMEOUT = number("OLLAMA_TIMEOUT", 180, 15, 900)

# One retry when the model returns unusable output or times out.
MODEL_ATTEMPTS = number("MODEL_ATTEMPTS", 2, 1, 3)

# ------------------------------------------------------------
# Search (Tavily)
# ------------------------------------------------------------

MAX_SEARCHES = number("MAX_SEARCHES", 4, 1, 8)
MAX_SOURCES = number("MAX_SOURCES", 12, 2, 20)

# "advanced" returns the most relevant parts of each page and
# costs 2 Tavily credits per search instead of 1.
SEARCH_DEPTH = os.getenv("SEARCH_DEPTH", "advanced").strip().lower()
if SEARCH_DEPTH not in {"basic", "advanced"}:
    SEARCH_DEPTH = "advanced"

# Fetch page text and keep the passages most related to the question.
USE_PAGE_TEXT = flag("USE_PAGE_TEXT", True)

# Search results scored below this by Tavily are dropped as off-topic.
MIN_SOURCE_SCORE = fraction("MIN_SOURCE_SCORE", 0.3, 0.0, 1.0)

# Results from the regulator / academic search must clear a higher bar:
# those domains always return something, even when nothing fits.
MIN_AUTHORITATIVE_SCORE = fraction("MIN_AUTHORITATIVE_SCORE", 0.45, 0.0, 1.0)

# Characters of evidence per source shown to the extractor and verifier.
EXCERPT_CHARS = number("EXCERPT_CHARS", 2400, 600, 6000)

# ------------------------------------------------------------
# Findings and verification
# ------------------------------------------------------------

MAX_FINDINGS_PER_SOURCE = number("MAX_FINDINGS_PER_SOURCE", 3, 1, 5)

# Claims reviewed per verification call. Small models judge one or
# two claims far more reliably than four.
VERIFY_BATCH_SIZE = number("VERIFY_BATCH_SIZE", 2, 1, 6)

# ------------------------------------------------------------
# Agent behaviour
# ------------------------------------------------------------

# A theme with fewer verified findings than this triggers a
# follow-up search.
MIN_FINDINGS_PER_THEME = number("MIN_FINDINGS_PER_THEME", 2, 1, 5)

# Follow-up rounds after the first pass (0 turns them off).
MAX_FOLLOWUP_ROUNDS = number("MAX_FOLLOWUP_ROUNDS", 1, 0, 2)

# Write a synthesised report (every sentence verified). When off,
# the report lists the verified findings under each theme.
ENABLE_WRITER = flag("ENABLE_WRITER", True)

# ------------------------------------------------------------
# Storage
# ------------------------------------------------------------

DB_PATH = os.getenv("DATABASE_PATH", "data/runs.db")
VERIFICATION_DB = os.getenv("VERIFICATION_DB", "data/verification_checkpoints.db")