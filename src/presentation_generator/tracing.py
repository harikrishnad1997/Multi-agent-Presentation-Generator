"""Opik tracing for the presentation generator (user directive,
2026-11-02, standing rule: "Always add it" — same pattern as Reel
Chat's graph/tracing.py, project "reel-places").

One Opik trace per deck build (project from OPIK_PROJECT_NAME,
default "multi-agent-presentation-generator"), with:
- a LangChain OpikTracer callback, so every LLM call and every
  LangGraph node is traced automatically in both the in-process and
  the A2A orchestrator runs;
- a span per A2A agent call (planner / writer / editor / chart /
  reviewer), with small metadata only — never the full deck.

Activation: tracing is ON when OPIK_API_KEY is present in the
environment. Without a key this module is a strict no-op — a deck
build is never affected. Tracing failures must never break a build,
so everything here fails soft. The Opik project is created
automatically when the first trace lands — nothing to provision.

Version note: opik is pinned to ==2.2.138 in requirements.txt. A
newer unpinned Opik rejected the 'llm' span type at import and
crashed Reel Chat in Nov 2026 (KeyError: 'llm'); do not unpin it
without re-testing the import.
"""
from __future__ import annotations

import contextlib
import functools
import os
from contextvars import ContextVar
from contextlib import contextmanager

# Workspace pinned in code: traces go to the "harikrishnad1997"
# workspace unless OPIK_WORKSPACE is explicitly set in the
# environment. (Same rule as Reel Chat, 2026-11-03.)
os.environ.setdefault("OPIK_WORKSPACE", "harikrishnad1997")
_PROJECT = os.environ.get("OPIK_PROJECT_NAME", "multi-agent-presentation-generator")
_import_ok = None
_REQUEST_OPIK_KEY: ContextVar[str | None] = ContextVar("presentation_request_opik_key", default=None)
_REQUEST_OPIK_CLIENT: ContextVar[object | None] = ContextVar("presentation_request_opik_client", default=None)


@contextmanager
def use_opik_api_key(api_key: str | None):
    """Set a tracing key for one generation without changing process env."""
    token = _REQUEST_OPIK_KEY.set(api_key.strip() if api_key and api_key.strip() else None)
    client_token = _REQUEST_OPIK_CLIENT.set(None)
    try:
        yield
    finally:
        _REQUEST_OPIK_CLIENT.reset(client_token)
        _REQUEST_OPIK_KEY.reset(token)


def _opik():
    global _import_ok
    if _import_ok is None:
        try:
            import opik  # noqa: F401

            _import_ok = True
        except Exception:
            _import_ok = False
    return _import_ok


def enabled() -> bool:
    if os.environ.get("PRESENTATION_OPIK", "").lower() in ("off", "0", "no"):
        return False
    return bool(_REQUEST_OPIK_KEY.get() or os.environ.get("OPIK_API_KEY")) and _opik()


def project_name() -> str:
    return _PROJECT


def status():
    """(enabled, workspace label, project) — the workspace label shows
    exactly where traces will land, so a wrong-workspace key is visible
    instead of silent."""
    ws = os.environ.get("OPIK_WORKSPACE") or "harikrishnad1997"
    return enabled(), ws, _PROJECT


def status_line() -> str:
    on, ws, project = status()
    if on:
        return f"Tracing (Opik): on — workspace {ws} · project {project}"
    return "Tracing (Opik): off (set OPIK_API_KEY to turn it on)"


def get_callbacks() -> list:
    """LangChain callbacks for graph.invoke(config=...). Empty list
    when tracing is off, so callers can pass it unconditionally."""
    if not enabled():
        return []
    try:
        import opik
        from opik.integrations.langchain import OpikTracer

        api_key = _REQUEST_OPIK_KEY.get() or os.environ.get("OPIK_API_KEY")
        client = opik.Opik(
            project_name=_PROJECT,
            workspace=os.environ.get("OPIK_WORKSPACE") or "harikrishnad1997",
            api_key=api_key,
        )
        _REQUEST_OPIK_CLIENT.set(client)

        class RequestOpikTracer(OpikTracer):
            @property
            def _opik_client(self):
                return client

        return [RequestOpikTracer(project_name=_PROJECT)]
    except Exception:
        return []


def prepare_graph(graph):
    """Attach Opik to a compiled LangGraph and return its invoke config.

    ``track_langgraph`` records the compiled graph structure as well as run
    spans. Keep the callback-only path as a compatibility fallback so tracing
    remains best-effort if graph wrapping is unavailable.
    """
    callbacks = get_callbacks()
    if not callbacks:
        return graph, {}
    try:
        from opik.integrations.langchain import track_langgraph

        return track_langgraph(graph, callbacks[0]), {}
    except Exception:
        return graph, {"callbacks": callbacks}


def track(name, span_type="general"):
    """Decorator factory: opik.track when enabled, identity otherwise.

    LAZY on purpose: nothing touches opik until the decorated function
    is first CALLED, so a broken opik can never break an import. Any
    opik failure at call time degrades to an untraced call, never a
    crash. (Copied from Reel Chat's tracing.py.)"""
    def lazy_deco(fn):
        state = {"wrapped": None, "tried": False}

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not state["tried"]:
                state["tried"] = True
                if enabled():
                    try:
                        import opik

                        deco = opik.track(name=name, type=span_type, project_name=_PROJECT)
                        state["wrapped"] = deco(fn)
                    except Exception:
                        state["wrapped"] = None
            target = state["wrapped"] if state["wrapped"] is not None else fn
            return target(*args, **kwargs)

        return wrapper

    return lazy_deco


@contextlib.contextmanager
def span(name: str, metadata: dict | None = None):
    """Explicit Opik span for one agent call (used by the A2A agents,
    where a decorator would capture the agent object as input). The
    span is a no-op context manager when tracing is off, and any Opik
    failure is swallowed — the wrapped work always runs."""
    trace = None
    if enabled():
        try:
            import opik

            client = opik.Opik(project_name=_PROJECT)
            trace = client.trace(name=name, metadata=metadata or {})
        except Exception:
            trace = None
    try:
        yield trace
    finally:
        if trace is not None:
            try:
                trace.end()
            except Exception:
                pass


def flush():
    """Push pending traces (Opik batches in a background thread).
    Never raises; a failed flush is silently dropped."""
    if not enabled():
        return
    try:
        client = _REQUEST_OPIK_CLIENT.get()
        if client is not None:
            client.flush()
            return
        import opik

        opik.flush_tracker()
    except Exception:
        pass
