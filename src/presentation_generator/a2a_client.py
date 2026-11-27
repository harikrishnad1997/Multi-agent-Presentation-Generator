"""A2A client: call one agent over HTTP, get back its JSON artifact.

Flow: resolve the Agent Card from /.well-known/agent-card.json,
send a Message carrying the JSON payload as text, read the Task's
first artifact back as JSON. Plain A2A — no LangChain, no shared
memory between orchestrator and agent.
"""
from __future__ import annotations

import asyncio
import json
import uuid

import httpx
from a2a.client import A2AClient
from a2a.client.card_resolver import A2ACardResolver
from a2a.types import Message, MessageSendParams, Part, SendMessageRequest, TextPart

from .tracing import TRACE_KEY, trace_headers


def _artifact_text(result) -> str:
    # SendMessageResponse.result is a Task (with artifacts) or a Message.
    artifacts = getattr(result, "artifacts", None)
    if artifacts:
        return artifacts[0].parts[0].root.text
    parts = getattr(result, "parts", None) or []
    if parts:
        return parts[0].root.text
    raise RuntimeError(f"A2A agent returned no artifact: {result!r}")


async def call_agent(base_url: str, payload: dict, timeout: float = 600) -> dict:
    # trust_env=False: this VM's proxy env (NO_PROXY with bracketed
    # IPv6 entries like [::1]) breaks httpx's proxy parsing at client
    # construction. A2A calls here are always to 127.0.0.1, no proxy.
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as httpx_client:
        resolver = A2ACardResolver(httpx_client, base_url)
        card = await resolver.get_agent_card()
        client = A2AClient(httpx_client, card)
        message = Message(
            role="user",
            parts=[Part(root=TextPart(text=json.dumps(payload)))],
            message_id=str(uuid.uuid4()),
        )
        request = SendMessageRequest(
            id=str(uuid.uuid4()), params=MessageSendParams(message=message)
        )
        response = await client.send_message(request)
        # SendMessageResponse is a RootModel: .root is the success
        # response, whose .result is the Task (or Message) — or an
        # error response, which we surface instead of crashing on .result.
        root = response.root
        if not hasattr(root, "result"):
            raise RuntimeError(f"A2A agent error: {getattr(root, 'error', root)}")
        return json.loads(_artifact_text(root.result))


def call_agent_sync(base_url: str, payload: dict, timeout: float = 600, trace: dict | None = None) -> dict:
    """trace: the caller's trace_headers(). Read here by default; pass it
    explicitly from worker threads, which don't see the node's trace."""
    headers = trace_headers() if trace is None else trace
    if headers:
        payload = {**payload, TRACE_KEY: headers}
    return asyncio.run(call_agent(base_url, payload, timeout=timeout))
