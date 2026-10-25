"""Conservative lexical query interpretation baseline; no generated concepts."""

import re

from pda.models import Interpretation

STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "for",
    "to",
    "in",
    "on",
    "and",
    "or",
    "is",
    "are",
    "how",
    "what",
    "which",
    "can",
    "with",
    "using",
    "research",
    "papers",
    "paper",
    "find",
    "about",
    "me",
    "show",
    "does",
    "do",
    "i",
    "want",
    "study",
    "studies",
}


def tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[\w]+(?:[-'][\w]+)*", text)
        if token.lower() not in STOPWORDS and len(token) > 1
    ]


def terms(text: str) -> list[str]:
    return list(dict.fromkeys(tokenize(text)))


def interpret(query: str) -> Interpretation:
    query = " ".join(query.split())
    if not 3 <= len(query) <= 1000:
        raise ValueError("Enter a research topic between 3 and 1,000 characters.")
    topic = re.sub(
        r"^(?:i\s+am\s+looking\s+for|i\s+want\s+to\s+(?:find|learn|know)|"
        r"please\s+(?:find|show)(?:\s+me)?|can\s+you\s+(?:find|show)(?:\s+me)?)"
        r"\s+(?:information\s+(?:about|on)\s+)?",
        "",
        query,
        flags=re.IGNORECASE,
    )
    quoted = re.fullmatch(r'"([^"\n]+)"', topic)
    concepts = terms(topic)
    if not concepts:
        raise ValueError("Include at least one specific research concept.")
    warnings = []
    if len(concepts) > 16:
        warnings.append("Search uses the first 16 terms; ranking uses your complete query.")
    if set(concepts) & {"not", "without", "excluding", "except"}:
        warnings.append("This baseline does not enforce exclusions; inspect the search terms.")
    # Preserve an explicitly quoted title/phrase, including its stopwords.
    # Otherwise require topic terms together to avoid unrelated OR candidates.
    search = (
        f'all:"{quoted.group(1)}"'
        if quoted
        else " AND ".join(f'all:"{term}"' for term in concepts[:16])
    )
    return Interpretation(
        original=query,
        concepts=concepts,
        search_query=search,
        warnings=warnings,
        quoted_title=quoted.group(1) if quoted else None,
    )
