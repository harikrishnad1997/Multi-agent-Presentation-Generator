"""User comments that edit a deck.

A comment is free text, optionally aimed at one slide:

    Slide 5: shorten the bullets to one line each
    overall: add a timeline chart to the history slide
    Drop slide 12, it repeats slide 9.

Comments arrive as `--comments "..."` (one per line) or a JSON file
(`--comments-file comments.json`):
    {"comments": [{"slide": 5, "comment": "..."}, {"slide": null, "comment": "..."}]}
or simply: ["Slide 5: ...", "overall: ..."]

Two uses:
1. On a fresh build (`--comments`), the comments are applied as an
   extra editor pass after the normal pipeline, before compiling.
2. On an existing deck (`--edit deck.json --comments ...`), only the
   editor pass runs — no slide is regenerated from scratch. The
   editor rewrites the targeted slides' bullets/notes, drops slides
   it is asked to drop, and adds a slide when a comment asks for one.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .graph import _fill_briefs, edit_deck, write_one_slide
from .llm import get_llm
from .models import Deck, Outline, SlideContent


def parse_comments(text: str | None = None, path: str | None = None) -> list[dict]:
    """Return a list of {"slide": int|None, "comment": str}."""
    raw: str
    if path:
        raw = Path(path).read_text(encoding="utf-8")
    else:
        raw = text or ""
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("{") or raw.startswith("["):
        try:
            data = json.loads(raw)
            items = data.get("comments", data) if isinstance(data, dict) else data
            out = []
            for item in items:
                if isinstance(item, str):
                    out.append(_parse_line(item))
                else:
                    out.append({"slide": item.get("slide"), "comment": str(item.get("comment", ""))})
            return [c for c in out if c["comment"]]
        except json.JSONDecodeError:
            pass  # fall through to plain-text parsing
    return [_parse_line(line) for line in raw.splitlines() if line.strip()]


def _parse_line(line: str) -> dict:
    line = line.strip()
    match = re.match(r"(?i)^slide\s+(\d+)\s*[:\-]\s*(.+)$", line)
    if match:
        return {"slide": int(match.group(1)), "comment": match.group(2).strip()}
    if line.lower().startswith("overall:"):
        return {"slide": None, "comment": line[len("overall:"):].strip()}
    return {"slide": None, "comment": line}


def format_comments(comments: list[dict]) -> str:
    lines = []
    for c in comments:
        target = f"Slide {c['slide']}" if c.get("slide") else "Whole deck"
        lines.append(f"- {target}: {c['comment']}")
    return "\n".join(lines)


def apply_comments(deck: Deck, outline: Outline, comments: list[dict],
                   model: str | None, topic: str = "", source_material: str = "") -> Deck:
    """Apply user comments via the Flow Editor, with add/drop support.

    Order matters, learned from a bad test run: slide numbers are
    resolved against the ORIGINAL deck first ("Slide 30" means the
    slide that was 30th when the user looked at it — adding a slide
    must not shift the target). Drops happen deterministically by
    number before the editor runs, so the editor never has to guess
    what to drop. An 'add a slide about X' request is checked against
    existing titles first — if the deck already has a slide on X, the
    comment becomes an edit of that slide instead of a duplicate."""
    if not comments:
        return deck
    llm = get_llm(model)
    slides = sorted(deck.slides, key=lambda s: s.number)
    by_number = {s.number: s for s in slides}
    topic_words = set(re.findall(r"[a-z0-9]+", (topic or deck.title).lower()))
    from .graph import norm_title_key

    # 1) Resolve numbered comments to titles; collect drops.
    resolved: list[dict] = []
    drop_numbers: set[int] = set()
    dropped_titles: set[str] = set()
    for c in comments:
        n = c.get("slide")
        comment = c["comment"]
        if n is not None and n in by_number:
            resolved.append({"slide": None,
                             "comment": f"the slide titled '{by_number[n].title}' (was slide {n}): {comment}"})
        else:
            resolved.append({"slide": None, "comment": comment})
        m = re.search(r"\bdrop (?:slide )?(\d+)\b", comment, re.I)
        if m and int(m.group(1)) in by_number:
            drop_numbers.add(int(m.group(1)))
            dropped_titles.add(by_number[int(m.group(1))].title.strip().lower())

    # 2) Deterministic drop on the original numbering.
    if drop_numbers:
        slides = [s for s in slides if s.number not in drop_numbers]
        for i, s in enumerate(slides, start=1):
            s.number = i

    # 3) Add requests: skip if a near-duplicate title already exists,
    # and turn the request into an edit of that slide instead.
    add_requests = [c for c in resolved
                    if re.search(r"\badd (a|another|one|a new)\b.*\bslide\b", c["comment"], re.I)]
    existing_keys = {norm_title_key(s.title, topic_words): s for s in slides}
    for req in add_requests:
        m = re.search(r"\badd (?:a|another|one|a new) slide (?:about|on|for|covering)?\s*(.+)", req["comment"], re.I)
        wanted = (m.group(1) if m else req["comment"]).strip(" .")
        key = norm_title_key(wanted, topic_words)
        hit = existing_keys.get(key)
        if hit is not None:
            req["comment"] = (f"the slide titled '{hit.title}': cover this too — {wanted} "
                              f"(user asked for a new slide on it, but the deck already has one)")
            req["was_add"] = True
    real_adds = [c for c in add_requests if not c.get("was_add")]
    if real_adds:
        try:
            ending = slides[-1] if slides and slides[-1].title.lower().startswith(("conclusion", "summary")) else None
            body = slides[:-1] if ending else slides
            for req in real_adds:
                briefs = _fill_briefs(llm, topic or deck.title, deck.title,
                                      [s.title for s in slides], 1, len(body) + 1)
                if not briefs:
                    continue
                brief = briefs[0]
                if norm_title_key(brief.title, topic_words) in existing_keys:
                    # The planner still produced a near-duplicate title:
                    # fold it into the existing slide instead.
                    hit = existing_keys[norm_title_key(brief.title, topic_words)]
                    resolved.append({"slide": None,
                                     "comment": f"the slide titled '{hit.title}': also cover — {brief.brief}"})
                    continue
                brief.brief = f"{brief.brief} (Requested by the user: {req['comment']})"
                body.append(write_one_slide(brief, outline, body[-1].title if body else "",
                                            ending.title if ending else "", model))
                existing_keys[norm_title_key(brief.title, topic_words)] = body[-1]
                slides = body + ([ending] if ending else [])
            deck = Deck(title=deck.title, slides=slides)
            slides = sorted(deck.slides, key=lambda s: s.number)
        except Exception:
            pass

    # 4) Everything else goes to the editor as instructions.
    text = format_comments([c for c in resolved if not c.get("was_add") or True])
    edited = edit_deck(
        outline, slides, model,
        review_issues=f"User comments — apply every one that names a slide in this chunk, and every whole-deck comment:\n{text}",
        topic=topic or deck.title,
        num_slides=len(slides),
        source_material=source_material,
    )
    # 5) A dropped slide must stay dropped even if the editor's fill
    # re-added a slide with the same title.
    if dropped_titles:
        kept = [s for s in edited.slides if s.title.strip().lower() not in dropped_titles]
        if len(kept) < len(edited.slides):
            for i, s in enumerate(sorted(kept, key=lambda x: x.number), start=1):
                s.number = i
            edited = Deck(title=edited.title, slides=sorted(kept, key=lambda x: x.number))
    return edited


def edit_deck_file(deck_path: str, outline_path: str | None = None, comments: str = "",
                   model: str | None = None, author: str = "", institute: str = "",
                   output_dir: str = "out", source_material: str = "") -> dict:
    """Edit an existing deck file in place (no slide regenerated from
    scratch): load deck.json (+ outline.json if present), apply the
    user's comments, re-render charts + LaTeX, review once, compile.
    Returns the same result dict shape as graph.run()."""
    import json as _json

    from .graph import compile_output, render_deck_charts, review_deck
    from .latex_utils import render_beamer
    from .models import SlideBrief

    deck = Deck.model_validate(_json.loads(Path(deck_path).read_text(encoding="utf-8")))
    if outline_path and Path(outline_path).exists():
        outline = Outline.model_validate(_json.loads(Path(outline_path).read_text(encoding="utf-8")))
    else:
        outline = Outline(
            title=deck.title, audience="", style_guide="",
            slides=[SlideBrief(number=s.number, title=s.title, brief="", key_points=list(s.bullets))
                    for s in deck.slides],
        )
    edited = apply_comments(deck, outline, parse_comments(text=comments), model,
                            topic=deck.title, source_material=source_material)
    edited, chart_count = render_deck_charts(edited, output_dir)
    latex = render_beamer(edited, author, institute)
    review = review_deck(edited, latex, chart_count, model)
    compiled = compile_output({
        "output_dir": output_dir, "latex": latex,
        "deck": edited.model_dump(), "outline": outline.model_dump(),
        "review": review.model_dump(),
    })
    return {
        "deck": edited.model_dump(), "outline": outline.model_dump(),
        "review": review.model_dump(), "chart_count": chart_count,
        "attempts": 1, **compiled,
    }
