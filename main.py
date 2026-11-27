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

from presentation_generator.comments import edit_deck_file  # noqa: E402
from presentation_generator.graph import MAX_ATTEMPTS, run  # noqa: E402
from presentation_generator.sources import load_source  # noqa: E402
from presentation_generator.tracing import status_line  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-agent presentation generator (LangChain + LangGraph, GPT)")
    ap.add_argument("--topic", default=None, help="Presentation topic, e.g. 'Google Gemini' (defaults to the source title with --source)")
    ap.add_argument("--source", default=None, help="Build the deck from a source: a PDF, a .txt/.md file, a local folder/repo, or a public GitHub repo URL")
    ap.add_argument("--comments", default=None, help="User comments to apply while editing, one per line (e.g. 'Slide 5: shorten the bullets')")
    ap.add_argument("--comments-file", default=None, help="File with comments, one per line or JSON (see comments.py)")
    ap.add_argument("--edit", default=None, help="Edit an existing deck.json instead of building a new deck (no slide regenerated from scratch)")
    ap.add_argument("--outline", default=None, help="outline.json for --edit (else rebuilt from the deck)")
    ap.add_argument("--author", default="Harikrishna Dev")
    ap.add_argument("--institute", default="University of Texas at Dallas")
    ap.add_argument("--slides", type=int, default=6, help="Number of content slides (title slide is extra)")
    ap.add_argument("--model", default=None, help="GPT model, default gpt-4o-mini (or OPENAI_MODEL)")
    ap.add_argument("--out", default="out", help="Output directory")
    ap.add_argument("--no-charts", action="store_true", help="Disable visuals/graphs")
    ap.add_argument("--images", action="store_true", help="Select relevant, licensed images from the topic's Wikipedia article")
    ap.add_argument("--no-search", action="store_true", help="Skip grounding entirely (deck is from model knowledge only)")
    ap.add_argument("--shallow", action="store_true", help="Skip deep research; use the quick Wikipedia-only grounding instead")
    args = ap.parse_args()

    comments = ""
    if args.comments_file:
        comments = Path(args.comments_file).read_text(encoding="utf-8")
    elif args.comments:
        comments = args.comments

    if args.edit:
        if not comments:
            print("--edit needs --comments or --comments-file (what should change?).")
            return 2
        started = time.time()
        print(f"Editing deck: {args.edit}")
        print(status_line())
        result = edit_deck_file(
            deck_path=args.edit, outline_path=args.outline, comments=comments,
            model=args.model, author=args.author, institute=args.institute,
            output_dir=args.out,
        )
        elapsed = time.time() - started
        deck = result.get("deck", {})
        review = result.get("review", {})
        print(f"\nDone in {elapsed:.0f}s — {len(deck.get('slides', []))} slides, {result.get('chart_count', 0)} charts, passed: {review.get('passed')}")
        for slide in deck.get("slides", []):
            chart = " + chart" if slide.get("chart_file") else ""
            print(f"  {slide['number']}. {slide['title']} ({len(slide.get('bullets', []))} bullets{chart})")
        if review.get("issues"):
            print("Review issues:")
            for issue in review["issues"]:
                print(f"  - [{issue.get('severity')}] {issue.get('issue')}")
        print(f"\nFiles in {args.out}/: presentation.tex, deck.json, outline.json, review.json")
        print(f"PDF: {result.get('pdf_path') or 'not built — no tectonic/pdflatex found'}")
        return 0

    source_text = ""
    references = []
    topic = args.topic
    if args.source:
        source = load_source(args.source)
        print(f"Source: {source.kind} — {source.ref} ({len(source.text)} chars)")
        source_text = source.text
        references.append(source.ref)
        topic = topic or source.title
    if not topic:
        print("--topic is required (or pass --source so the topic comes from the source).")
        return 2

    started = time.time()
    print(f"Topic: {topic}  |  slides: {args.slides}  |  charts: {'off' if args.no_charts else 'on (~1 in 3 slides)'}  |  model: {args.model or 'gpt-4o-mini (default)'}")
    print(status_line())
    print("Agents: planner -> parallel slide writers -> flow editor -> charts -> image selector -> LaTeX -> reviewer -> compiler")
    result = run(
        topic=topic,
        author=args.author,
        institute=args.institute,
        num_slides=args.slides,
        model=args.model,
        output_dir=args.out,
        charts_enabled=not args.no_charts,
        source_text=source_text,
        comments=comments,
        search=not args.no_search,
        shallow=args.shallow,
        images_enabled=args.images,
        references=references,
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
        print("Review issues:")
        for issue in review["issues"]:
            print(f"  - [{issue.get('severity')}] {issue.get('issue')}")
    print(f"\nFiles in {args.out}/: presentation.tex, deck.json, outline.json, review.json" + (", charts/*.png" if result.get('chart_count') else ""))
    print(f"PDF: {result.get('pdf_path') or 'not built — no tectonic/pdflatex found (see compile note in output)'}")
    print(f"Compile log tail: {(result.get('compile_log') or '')[-300:]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
