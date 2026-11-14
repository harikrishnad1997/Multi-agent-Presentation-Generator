#!/usr/bin/env python3
"""CLI: generate a Beamer presentation from a topic with LangChain agents.

Usage:
    python main.py --topic "Google Gemini" --author "Harikrishna Dev" \
        --institute "University of Texas at Dallas" --slides 6 --out out/gemini

GPT is the LLM (default gpt-4o-mini, --model to change). The key comes
from OPENAI_API_KEY if set, otherwise from Muse's secure OpenAI
connector helper — see src/presentation_generator/llm.py. No key is
ever written to the repo or to the output files.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from presentation_generator.graph import run  # noqa: E402
from presentation_generator.tracing import status_line  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-agent presentation generator (LangChain + LangGraph, GPT)")
    ap.add_argument("--topic", required=True, help="Presentation topic, e.g. 'Google Gemini'")
    ap.add_argument("--author", default="Harikrishna Dev")
    ap.add_argument("--institute", default="University of Texas at Dallas")
    ap.add_argument("--slides", type=int, default=6, help="Number of content slides (title slide is extra)")
    ap.add_argument("--model", default=None, help="GPT model, default gpt-4o-mini (or OPENAI_MODEL)")
    ap.add_argument("--out", default="out", help="Output directory")
    ap.add_argument("--no-charts", action="store_true", help="Disable visuals/graphs")
    args = ap.parse_args()

    started = time.time()
    print(f"Topic: {args.topic}  |  slides: {args.slides}  |  charts: {'off' if args.no_charts else 'on (~1 in 3 slides)'}  |  model: {args.model or 'gpt-4o-mini (default)'}")
    print(status_line())
    print("Agents: planner -> parallel slide writers -> flow editor -> charts -> LaTeX -> reviewer -> compiler")
    result = run(
        topic=args.topic,
        author=args.author,
        institute=args.institute,
        num_slides=args.slides,
        model=args.model,
        output_dir=args.out,
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
        print("Review issues:")
        for issue in review["issues"]:
            print(f"  - [{issue.get('severity')}] {issue.get('issue')}")
    print(f"\nFiles in {args.out}/: presentation.tex, deck.json, outline.json, review.json" + (", charts/*.png" if result.get('chart_count') else ""))
    print(f"PDF: {result.get('pdf_path') or 'not built — no tectonic/pdflatex found (see compile note in output)'}")
    print(f"Compile log tail: {(result.get('compile_log') or '')[-300:]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
