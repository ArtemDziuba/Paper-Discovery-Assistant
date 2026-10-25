"""Validated pipeline records."""

from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl


class Paper(BaseModel):
    arxiv_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    abstract: str = Field(min_length=1)
    authors: list[str] = Field(min_length=1)
    published: datetime
    categories: list[str]
    url: HttpUrl

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.abstract}"


class Interpretation(BaseModel):
    original: str
    concepts: list[str]
    search_query: str
    quoted_title: str | None = None
    warnings: list[str] = Field(default_factory=list)


class Match(BaseModel):
    paper: Paper
    raw_similarity: float
    display_score: float
    evidence: str
    matched_terms: list[str]


class SearchResult(BaseModel):
    interpretation: Interpretation
    matches: list[Match]
    overview: list[dict[str, str]]
    metadata: dict
