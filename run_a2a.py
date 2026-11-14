#!/usr/bin/env python3
"""Run the presentation pipeline over A2A.

Starts the five A2A agent servers (planner 8101, writer 8102,
editor 8103, chart 8104, reviewer 8105), waits for their Agent
Cards, runs the LangGraph orchestrator as an A2A client, then
shuts the servers down.

Usage:
    python run_a2a.py --topic "Google Gemini" --slides 30 --out out/gemini30-a2a
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from presentation_generator.a2a_agents import AGENT_CLASSES, PORTS  # noqa: E402
from presentation_generator.graph_a2a import run_a2a  # noqa: E402
from presentation_generator.tracing import status_line  # noqa: E402


def _wait_for_card(port: int, timeout: float = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/.well-known/agent-card.json", timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-agent presentation generator over A2A (GPT)")
    ap.add_argument("--topic", required=True)
    ap.add_argument("--author", default="Harikrishna Dev")
    ap.add_argument("--institute", default="University of Texas at Dallas")
    ap.add_argument("--slides", type=int, default=6)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="out-a2a")
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).parent
    env = {"PYTHONPATH": str(root / "src")}
    import os

    env = {**os.environ, **env}
    procs: list[subprocess.Popen] = []
    try:
        for name in AGENT_CLASSES:
            port = PORTS[name]
            proc = subprocess.Popen(
                [sys.executable, "-m", "presentation_generator.a2a_server", "--agent", name, "--port", str(port)],
                cwd=str(root), env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            procs.append(proc)
            print(f"Starting presentation-{name} on :{port} ...")
        for name, port in PORTS.items():
            if not _wait_for_card(port):
                print(f"ERROR: presentation-{name} did not come up on :{port}")
                return 1
            print(f"  card OK: http://127.0.0.1:{port}/.well-known/agent-card.json")
        print(f"\nTopic: {args.topic}  |  slides: {args.slides}  |  charts: {'off' if args.no_charts else 'on'}  |  transport: A2A")
        print(status_line())
        started = time.time()
        result = run_a2a(
            topic=args.topic, author=args.author, institute=args.institute,
            num_slides=args.slides, model=args.model, output_dir=args.out,
            charts_enabled=not args.no_charts,
        )
        elapsed = time.time() - started
        deck = result.get("deck", {})
        review = result.get("review", {})
        print(f"\nDone in {elapsed:.0f}s — {len(deck.get('slides', []))} slides, {result.get('chart_count', 0)} charts, review rounds: {result.get('attempts')}, passed: {review.get('passed')}")
        for slide in deck.get("slides", []):
            chart = " + chart" if slide.get("chart_file") else ""
            print(f"  {slide['number']}. {slide['title']} ({len(slide.get('bullets', []))} bullets{chart})")
        if review.get("issues"):
            for issue in review["issues"]:
                print(f"  - [{issue.get('severity')}] slide {issue.get('slide')}: {issue.get('issue')}")
        print(f"\nFiles in {args.out}/: presentation.tex, deck.json, outline.json, review.json" + (", charts/*.png" if result.get("chart_count") else ""))
        print(f"PDF: {result.get('pdf_path') or 'not built — no tectonic/pdflatex found'}")
        return 0
    finally:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
