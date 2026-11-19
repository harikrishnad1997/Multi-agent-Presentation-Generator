"""Multi-agent presentation generator (LangChain + LangGraph)."""
from __future__ import annotations

import os
from pathlib import Path

__version__ = "0.1.0"


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from a .env in the working directory or the
    project root. Variables already set in the environment win, and empty
    values are skipped so a blank OPENAI_API_KEY= keeps the connector
    fallback. No dependency: the format is the one in .env.example."""
    for path in (Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.removeprefix("export ").strip()
            value = value.strip().strip("'\"")
            if key and value:
                os.environ.setdefault(key, value)
        return


_load_dotenv()
