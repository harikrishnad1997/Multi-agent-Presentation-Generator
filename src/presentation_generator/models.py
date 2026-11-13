"""Structured data for the presentation pipeline.

Every agent speaks JSON validated against these models, so the next
agent never has to parse free-form prose. That is the whole point of
using LangChain structured output here: a writer cannot return a
half-finished slide and a LaTeX step cannot receive a paragraph
where it expected a bullet list.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ChartSpec(BaseModel):
    """A quantitative visual for one slide. Data is supplied by the
    writer agent as structured numbers, then rendered deterministically
    with matplotlib — the LLM never draws the chart itself, so labels
    and values always match. Values are illustrative unless the
    source_note says otherwise; check them before presenting."""

    chart_type: str = Field(description="bar, line, or pie")
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    labels: list[str] = Field(default_factory=list, description="3-8 category labels")
    values: list[float] = Field(default_factory=list, description="One number per label, same length as labels")
    source_note: str = "Illustrative data — verify before presenting."


class SlideBrief(BaseModel):
    number: int
    title: str
    brief: str = Field(description="One or two sentences on what this slide must cover.")
    key_points: list[str] = Field(default_factory=list)
    has_visual: bool = False
    visual_hint: str = Field(default="", description="What the chart should show, if has_visual.")


class Outline(BaseModel):
    title: str
    audience: str = ""
    style_guide: str = Field(
        default="Clear, concise bullets. One idea per bullet. No jargon without a plain explanation."
    )
    slides: list[SlideBrief]


class SlideContent(BaseModel):
    number: int
    title: str
    bullets: list[str] = Field(description="3-5 bullets, each one idea, plain text, no LaTeX.")
    notes: str = Field(default="", description="Speaker notes, 1-3 sentences, plain text.")
    chart: ChartSpec | None = None
    chart_file: str | None = None  # set by the chart renderer, e.g. charts/slide_03.png


class Deck(BaseModel):
    title: str
    slides: list[SlideContent]


class ReviewIssue(BaseModel):
    severity: str = Field(default="minor", description="critical, major, or minor")
    slide: int | None = None
    issue: str
    fix: str = ""


class ReviewResult(BaseModel):
    passed: bool
    issues: list[ReviewIssue] = Field(default_factory=list)
    summary: str = ""
