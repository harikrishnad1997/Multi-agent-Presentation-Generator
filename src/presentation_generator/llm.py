"""LLM access for the pipeline.

Two backends, one interface (a LangChain chat model):

1. ``OPENAI_API_KEY`` set -> ``ChatOpenAI`` (the normal path for anyone
   cloning this repo).
2. No key in the environment -> ``HelperChatModel``, a small
   ``BaseChatModel`` that shells out to Muse's secure OpenAI connector
   helper (``openai_chat.py``). The helper injects the key at request
   time via a surrogate exchange; the key never appears in an env var,
   a file, this code, or any output. This is the path used on Hari's
   machine, following the pattern in ``reel-to-map/graph/llm.py``.

Structured output is done by prompting for JSON-only and parsing it
in ``graph.py``, not by LangChain tool-calling: the helper backend
only does plain chat completions, so tool-calling would silently
fail there. JSON-only prompting works identically on both backends.
"""
from __future__ import annotations

import os
import subprocess
import sys
from contextvars import ContextVar
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
HELPER_PATH = Path(
    os.environ.get(
        "OPENAI_HELPER",
        Path.home() / "workspace/custom-connectors/custom.openai/bin/openai_chat.py",
    )
)
_REQUEST_API_KEY: ContextVar[str | None] = ContextVar("presentation_request_api_key", default=None)


@contextmanager
def use_api_key(api_key: str | None):
    """Use a per-job API key without changing the process environment."""
    token = _REQUEST_API_KEY.set(api_key.strip() if api_key and api_key.strip() else None)
    try:
        yield
    finally:
        _REQUEST_API_KEY.reset(token)


class HelperChatModel(BaseChatModel):
    """Chat model backed by the Muse OpenAI connector helper."""

    model: str = DEFAULT_MODEL
    helper_path: str = str(HELPER_PATH)

    @property
    def _llm_type(self) -> str:
        return "openai-helper"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "backend": "openai-helper"}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        system_parts = [m.content for m in messages if m.type == "system"]
        user_parts = [m.content for m in messages if m.type != "system"]
        system = "\n\n".join(str(p) for p in system_parts)
        user = "\n\n".join(str(p) for p in user_parts)
        proc = subprocess.run(
            [sys.executable, self.helper_path, "--system", system, "--user", user, "--model", self.model],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"OpenAI helper failed: {(proc.stderr or '').strip()[:500]}")
        text = (proc.stdout or "").strip()
        if not text:
            raise RuntimeError("OpenAI helper returned an empty reply")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def get_llm(model: str | None = None) -> BaseChatModel:
    """Return a LangChain chat model for the given model name."""
    name = model or DEFAULT_MODEL
    api_key = _REQUEST_API_KEY.get() or os.environ.get("OPENAI_API_KEY")
    if api_key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=name, temperature=0.7, api_key=api_key)
    if HELPER_PATH.is_file():
        return HelperChatModel(model=name)
    raise RuntimeError(
        "No LLM backend: set OPENAI_API_KEY, or run on a machine with "
        "the Muse OpenAI connector helper at ~/workspace/custom-connectors/custom.openai/bin/openai_chat.py"
    )
