#!/usr/bin/env python3
"""Start the UI-facing FastAPI server."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import uvicorn

if __name__ == "__main__":
    uvicorn.run("presentation_generator.api:app", host="0.0.0.0", port=8000, reload=False)
