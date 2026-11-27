#!/usr/bin/env python3
"""Run the presentation pipeline over A2A.

Starts the A2A agent servers (planner 8101, writer 8102,
editor 8103, chart 8104, reviewer 8105, per-slide LaTeX 8106), waits for their Agent
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
from presentation_generator.graph import MAX_ATTEMPTS  # noqa: E402
from presentation_generator.graph_a2a import run_a2a  # noqa: E402
from presentation_generator.sources import load_source  # noqa: E402
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
    ap.add_argument("--topic", default=None, help="Presentation topic (defaults to the source title with --source)")
    ap.add_argument("--source", default=None, help="Build the deck from a source: a PDF, a .txt/.md file, a local folder/repo, or a public GitHub repo URL")
    ap.add_argument("--comments", default=None, help="User comments to apply while editing, one per line")
    ap.add_argument("--comments-file", default=None, help="File with comments, one per line or JSON")
    ap.add_argument("--author", default="Harikrishna Dev")
    ap.add_argument("--institute", default="University of Texas at Dallas")
    ap.add_argument("--slides", type=int, default=6)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="out-a2a")
    ap.add_argument("--no-charts", action="store_true")
    ap.add_argument("--no-search", action="store_true", help="Skip grounding entirely (deck is from model knowledge only)")
    ap.add_argument("--shallow", action="store_true", help="Skip deep research; use the quick Wikipedia-only grounding instead")
    args = ap.parse_args()

    root = Path(__file__).parent
    env = {"PYTHONPATH": str(root / "src")}
    import os

    env = {**os.environ, **env}
    procs: list[subprocess.Popen] = []
    # Each server's output goes to <out>/agent-logs/<name>.log, so a server
    # that fails to start (e.g. a missing dependency) explains itself.
    log_dir = Path(args.out) / "agent-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        for name in AGENT_CLASSES:
            port = PORTS[name]
            with open(log_dir / f"{name}.log", "w", encoding="utf-8") as log:
                proc = subprocess.Popen(
                    [sys.executable, "-m", "presentation_generator.a2a_server", "--agent", name, "--port", str(port)],
                    cwd=str(root), env=env,
                    stdout=log, stderr=subprocess.STDOUT,
                )
            procs.append(proc)
            print(f"Starting presentation-{name} on :{port} ...")
        for name, port in PORTS.items():
            if not _wait_for_card(port):
                print(f"ERROR: presentation-{name} did not come up on :{port}")
                log_file = log_dir / f"{name}.log"
                tail = log_file.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-15:]
                if tail:
                    print(f"Last lines of {log_file}:\n  " + "\n  ".join(tail))
                return 1
            print(f"  card OK: http://127.0.0.1:{port}/.well-known/agent-card.json")
        comments = ""
        if args.comments_file:
            comments = Path(args.comments_file).read_text(encoding="utf-8")
        elif args.comments:
            comments = args.comments
        source_text = ""
        topic = args.topic
        if args.source:
            source = load_source(args.source)
            print(f"Source: {source.kind} — {source.ref} ({len(source.text)} chars)")
            source_text = source.text
            topic = topic or source.title
        if not topic:
            print("--topic is required (or pass --source so the topic comes from the source).")
            return 2
        print(f"\nTopic: {topic}  |  slides: {args.slides}  |  charts: {'off' if args.no_charts else 'on'}  |  transport: A2A")
        print(status_line())
        started = time.time()
        result = run_a2a(
            topic=topic, author=args.author, institute=args.institute,
            num_slides=args.slides, model=args.model, output_dir=args.out,
            charts_enabled=not args.no_charts, source_text=source_text, comments=comments,
            search=not args.no_search, shallow=args.shallow,
        )
        elapsed = time.time() - started
        deck = result.get("deck", {})
        review = result.get("review", {})
        print(f"\nDone in {elapsed:.0f}s — {len(deck.get('slides', []))} slides, {result.get('chart_count', 0)} charts, build {result.get('build_attempt')} of {MAX_ATTEMPTS}, passed: {review.get('passed')}")
        for slide in deck.get("slides", []):
            chart = " + chart" if slide.get("chart_file") else ""
            print(f"  {slide['number']}. {slide['title']} ({len(slide.get('bullets', []))} bullets{chart})")
        grounding = result.get("grounding", {})
        if grounding.get("mode") == "research":
            print(f"Deep research: {len(grounding.get('sources', []))} sources across {len(grounding.get('queries', []))} questions")
            for src in grounding.get("sources", []):
                print(f"  - {src}")
        elif grounding.get("mode") == "search":
            print(f"Grounded in web search: {'; '.join(grounding.get('sources', []))}")
        elif grounding.get("mode") == "source":
            print("Grounded in the source you gave (--source).")
        elif grounding.get("mode") == "none":
            print("Note: search found no reliable sources — this deck is from model knowledge only, verify before presenting.")
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
