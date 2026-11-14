"""A2A agents: each pipeline agent as a real Agent2Agent agent.

Protocol (Google A2A, a2a-sdk 0.3.26 — the classic Starlette
implementation; 1.x moved AgentCard to protobuf types and dropped
the app factory, the wire protocol is the same):
- Every agent publishes an Agent Card at /.well-known/agent-card.json
  (name, description, skills, input/output modes, url).
- A client sends a Message (JSON payload as text) via JSON-RPC
  message/send; the agent replies with a Task whose artifact holds
  the JSON result, then a completed status.
- Agents are stateless and share the core functions in graph.py —
  plan_outline / write_one_slide / edit_deck / render_deck_charts /
  review_deck — so there is one implementation, two transports
  (in-process LangGraph, and A2A over HTTP). The LLM work runs in a
  thread (asyncio.to_thread) so the server's event loop stays free
  while a GPT call is in flight.
"""
from __future__ import annotations

import asyncio
import json
import uuid

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Artifact,
    Part,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)

from .graph import edit_deck, plan_outline, render_deck_charts, review_deck, write_one_slide
from .tracing import span as opik_span
from .latex_utils import render_beamer
from .models import Deck, Outline, SlideBrief, SlideContent

PORTS = {"planner": 8101, "writer": 8102, "editor": 8103, "chart": 8104, "reviewer": 8105}


def _payload(context: RequestContext) -> dict:
    text = context.get_user_input()
    return json.loads(text) if text.strip() else {}


async def _finish(event_queue: EventQueue, context: RequestContext, result: dict, name: str) -> None:
    artifact = Artifact(
        artifact_id=str(uuid.uuid4()),
        name=name,
        parts=[Part(root=TextPart(text=json.dumps(result)))],
    )
    await event_queue.enqueue_event(
        TaskArtifactUpdateEvent(
            append=False, last_chunk=True,
            context_id=context.context_id, task_id=context.task_id, artifact=artifact,
        )
    )
    await event_queue.enqueue_event(
        TaskStatusUpdateEvent(
            final=True,
            context_id=context.context_id,
            task_id=context.task_id,
            status=TaskStatus(state=TaskState.completed),
        )
    )


class _JsonAgent(AgentExecutor):
    """Base: parse JSON input, run blocking work in a thread, reply with a JSON artifact."""

    artifact_name = "result"
    agent_name = "agent"

    def run(self, payload: dict) -> dict:
        raise NotImplementedError

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        payload = _payload(context)
        # Opik span per agent call (no-op without OPIK_API_KEY; small
        # metadata only — never the full payload). Runs inside the
        # thread with the work, so the span covers the LLM call.
        with opik_span(f"presentation-{self.agent_name}", metadata={"agent": self.agent_name}):
            result = await asyncio.to_thread(self.run, payload)
        await _finish(event_queue, context, result, self.artifact_name)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Our agents run one short synchronous job per task; there is
        # nothing long-running to interrupt, so cancel is a no-op that
        # reports the task as canceled.
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                final=True,
                context_id=context.context_id,
                task_id=context.task_id,
                status=TaskStatus(state=TaskState.canceled),
            )
        )


class PlannerAgent(_JsonAgent):
    agent_name = "planner"
    artifact_name = "outline"

    def run(self, payload: dict) -> dict:
        outline = plan_outline(
            payload["topic"], payload.get("author", ""), int(payload.get("num_slides", 6)),
            payload.get("model"), bool(payload.get("charts_enabled", True)),
            source_text=payload.get("source_text", ""),
        )
        return {"outline": outline.model_dump()}


class WriterAgent(_JsonAgent):
    agent_name = "writer"
    artifact_name = "slide"

    def run(self, payload: dict) -> dict:
        brief = SlideBrief.model_validate(payload["brief"])
        outline = Outline.model_validate(payload["outline"])
        slide = write_one_slide(
            brief, outline, payload.get("prev_title", ""), payload.get("next_title", ""),
            payload.get("model"), source_text=payload.get("source_text", ""),
        )
        return {"slide": slide.model_dump()}


class EditorAgent(_JsonAgent):
    agent_name = "editor"
    artifact_name = "deck"

    def run(self, payload: dict) -> dict:
        from .comments import apply_comments, parse_comments

        outline = Outline.model_validate(payload["outline"])
        slides = [SlideContent.model_validate(s) for s in payload.get("slides", [])]
        raw_comments = payload.get("comments", "")
        if raw_comments:
            deck = apply_comments(
                Deck(title=outline.title, slides=slides), outline,
                parse_comments(text=raw_comments), payload.get("model"),
                topic=payload.get("topic", ""),
            )
        else:
            deck = edit_deck(
                outline, slides, payload.get("model"),
                review_issues=payload.get("review_issues", ""),
                topic=payload.get("topic", ""),
                num_slides=payload.get("num_slides"),
            )
        return {"deck": deck.model_dump()}


class ChartAgent(_JsonAgent):
    agent_name = "chart"
    artifact_name = "charts"

    def run(self, payload: dict) -> dict:
        deck = Deck.model_validate(payload["deck"])
        deck, count = render_deck_charts(deck, payload.get("output_dir", "out"))
        return {"deck": deck.model_dump(), "chart_count": count}


class ReviewerAgent(_JsonAgent):
    agent_name = "reviewer"
    artifact_name = "review"

    def run(self, payload: dict) -> dict:
        deck = Deck.model_validate(payload["deck"])
        latex = payload.get("latex") or render_beamer(deck, payload.get("author", ""), payload.get("institute", ""))
        result = review_deck(deck, latex, int(payload.get("chart_count", 0)), payload.get("model"))
        return {"review": result.model_dump()}


AGENT_CLASSES = {
    "planner": PlannerAgent,
    "writer": WriterAgent,
    "editor": EditorAgent,
    "chart": ChartAgent,
    "reviewer": ReviewerAgent,
}

AGENT_DESCRIPTIONS = {
    "planner": "Plans a presentation: topic -> structured outline with one brief per slide, validated (exact slide count, exact visual count, exactly one ending slide).",
    "writer": "Writes one slide from a brief: bullets + speaker notes + an optional chart spec (bar/line/pie).",
    "editor": "Flow editor: reads a deck in order, drops duplicated slides, fixes transitions and tone, fills any shortfall.",
    "chart": "Renders chart specs to PNG with matplotlib, deterministically.",
    "reviewer": "Reviews a deck and its LaTeX: duplicates, bullet counts, notes, chart validity, structural checks.",
}


def make_card(name: str, port: int) -> AgentCard:
    return AgentCard(
        name=f"presentation-{name}",
        description=AGENT_DESCRIPTIONS[name],
        url=f"http://127.0.0.1:{port}/",
        version="0.2.0",
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[
            AgentSkill(
                id=name,
                name=f"presentation-{name}",
                description=AGENT_DESCRIPTIONS[name],
                tags=["presentation", "beamer", "langchain"],
                examples=[f"Run the {name} step of a presentation build."],
            )
        ],
    )
