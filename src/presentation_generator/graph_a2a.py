"""A2A orchestrator: the same LangGraph pipeline, but every agent
step is an A2A call over HTTP instead of an in-process function call.

LangGraph still owns the flow — planner -> Send fan-out to writers
-> editor -> charts -> LaTeX -> reviewer -> compile — but each node
is now an A2A client: it resolves the agent's Card, sends a Message
with the JSON payload, and reads the Task artifact back. Agents run
as separate servers (see a2a_server.py / run_a2a.py) and can be
swapped, restarted, or reused independently of the orchestrator.
LaTeX rendering and compilation stay local and deterministic; they
are not agents, they are tools.
"""
from __future__ import annotations

import json

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .a2a_agents import PORTS
from .a2a_client import call_agent_sync
from .graph import (
    MAX_ATTEMPTS,
    PresentationState,
    compile_output,
    add_references,
    render_latex,
    route_reviewer,
)
from .models import Deck, Outline, ReviewResult, SlideContent

DEFAULT_URLS = {name: f"http://127.0.0.1:{port}/" for name, port in PORTS.items()}


def _urls(urls: dict | None) -> dict:
    return {**DEFAULT_URLS, **(urls or {})}


def make_a2a_graph(urls: dict | None = None):
    base = _urls(urls)

    def planner(state: PresentationState) -> dict:
        result = call_agent_sync(base["planner"], {
            "topic": state["topic"], "author": state.get("author", ""),
            "num_slides": state.get("num_slides", 6), "model": state.get("model"),
            "charts_enabled": bool(state.get("charts_enabled", True)),
            "source_text": state.get("source_text", ""),
            "review_feedback": state.get("review_issues", ""),
        })
        return {
            "outline": result["outline"], "attempts": 0, "slides": [],
            "deck": {}, "latex": "", "chart_count": 0, "review_issues": "",
            "build_attempt": state.get("build_attempt", 0) + 1,
        }

    def route_planner(state: PresentationState):
        outline = Outline.model_validate(state["outline"])
        briefs = sorted(outline.slides, key=lambda b: b.number)
        return [
            Send("write_slide", {
                **state, "brief": brief.model_dump(),
                "prev_title": briefs[i - 1].title if i > 0 else "",
                "next_title": briefs[i + 1].title if i + 1 < len(briefs) else "",
            })
            for i, brief in enumerate(briefs)
        ]

    def write_slide(state: PresentationState) -> dict:
        result = call_agent_sync(base["writer"], {
            "brief": state["brief"], "outline": state["outline"],
            "prev_title": state.get("prev_title", ""), "next_title": state.get("next_title", ""),
            "model": state.get("model"), "source_text": state.get("source_text", ""),
        })
        slide = result["slide"]
        slide["build_attempt"] = state.get("build_attempt", 1)
        return {"slides": [slide]}

    def editor(state: PresentationState) -> dict:
        # Only this build's slides: the accumulator still holds the
        # failed attempts' slides (see write_slide's build_attempt tag).
        slides = state["deck"]["slides"] if state.get("deck") else state.get("slides", [])
        if not state.get("deck"):
            current = state.get("build_attempt", 1)
            slides = [s for s in slides if s.get("build_attempt", current) == current]
        result = call_agent_sync(base["editor"], {
            "outline": state["outline"], "slides": slides,
            "review_issues": state.get("review_issues", ""),
            "topic": state.get("topic", ""), "num_slides": state.get("num_slides"),
            "model": state.get("model"), "comments": state.get("comments", ""),
        })
        deck = result["deck"]
        return {"deck": deck, "slides": deck["slides"]}

    def render_charts(state: PresentationState) -> dict:
        result = call_agent_sync(base["chart"], {
            "deck": state["deck"], "output_dir": state.get("output_dir") or "out",
        })
        return {"deck": result["deck"], "chart_count": result["chart_count"]}

    def reviewer(state: PresentationState) -> dict:
        deck = Deck.model_validate(state["deck"])
        result = call_agent_sync(base["reviewer"], {
            "deck": state["deck"], "latex": state.get("latex", ""),
            "chart_count": state.get("chart_count", 0), "model": state.get("model"),
            "author": state.get("author", ""), "institute": state.get("institute", ""),
        })
        review = ReviewResult.model_validate(result["review"])
        attempts = state.get("attempts", 0) + 1
        return {
            "review": review.model_dump(),
            "attempts": attempts,
            "review_issues": json.dumps([i.model_dump() for i in review.issues], indent=2),
        }

    builder = StateGraph(PresentationState)
    builder.add_node("planner", planner)
    builder.add_node("write_slide", write_slide)
    builder.add_node("editor", editor)
    builder.add_node("render_charts", render_charts)
    builder.add_node("add_references", add_references)
    builder.add_node("render_latex", render_latex)
    builder.add_node("reviewer", reviewer)
    builder.add_node("compile", compile_output)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", route_planner)
    builder.add_edge("write_slide", "editor")
    builder.add_edge("editor", "render_charts")
    builder.add_edge("render_charts", "add_references")
    builder.add_edge("add_references", "render_latex")
    builder.add_edge("render_latex", "reviewer")
    builder.add_conditional_edges("reviewer", route_reviewer, {"compile": "compile", "restart": "planner"})
    builder.add_edge("compile", END)
    return builder.compile()


def run_a2a(topic: str, author: str = "", institute: str = "", num_slides: int = 6,
            model: str | None = None, output_dir: str = "out",
            charts_enabled: bool = True, urls: dict | None = None,
            source_text: str = "", comments: str = "", search: bool = True,
            shallow: bool = False) -> dict:
    from .tracing import flush, get_callbacks

    grounding = {"mode": "source", "sources": []}
    references: list[str] = []
    if not source_text and search:
        from .research import research_topic

        found = research_topic(topic, model, shallow=shallow)
        if found.brief:
            source_text = found.brief
            grounding = {"mode": found.mode, "sources": found.sources, "queries": found.queries}
            references.extend(found.sources)
        else:
            grounding = {"mode": "none", "sources": []}
    elif not source_text:
        grounding = {"mode": "none", "sources": []}

    graph = make_a2a_graph(urls)
    initial: PresentationState = {
        "topic": topic, "author": author, "institute": institute,
        "num_slides": num_slides, "model": model or "", "output_dir": output_dir,
        "slides": [], "attempts": 0, "build_attempt": 0, "charts_enabled": charts_enabled, "chart_count": 0,
        "source_text": source_text, "comments": comments, "references": references,
    }
    try:
        # Same OpikTracer callback as the in-process run: it traces the
        # orchestrator's nodes and LLM calls. Each A2A agent also traces
        # its own span server-side (see a2a_agents._JsonAgent).
        result = graph.invoke(initial, config={"callbacks": get_callbacks()})
        result["grounding"] = grounding
        return result
    finally:
        flush()
