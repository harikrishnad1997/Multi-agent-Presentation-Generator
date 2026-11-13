"""Run one A2A agent as an HTTP server.

Usage:
    python -m presentation_generator.a2a_server --agent planner --port 8101
    python -m presentation_generator.a2a_server --agent writer --port 8102
    # ... editor 8103, chart 8104, reviewer 8105

Serves the Agent Card at /.well-known/agent-card.json and the
JSON-RPC endpoint at /. Use run_a2a.py to start all five at once.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from presentation_generator.a2a_agents import AGENT_CLASSES, PORTS, make_card


def build_app(agent_name: str, port: int):
    card = make_card(agent_name, port)
    executor = AGENT_CLASSES[agent_name]()
    handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
    return A2AStarletteApplication(agent_card=card, http_handler=handler).build()


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one presentation A2A agent")
    ap.add_argument("--agent", required=True, choices=sorted(AGENT_CLASSES))
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    port = args.port or PORTS[args.agent]
    app = build_app(args.agent, port)
    import uvicorn

    print(f"presentation-{args.agent} A2A agent on http://{args.host}:{port}/ (card: /.well-known/agent-card.json)")
    uvicorn.run(app, host=args.host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
