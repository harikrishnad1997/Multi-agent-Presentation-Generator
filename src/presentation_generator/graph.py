"""LangGraph pipeline: planner -> parallel slide writers -> flow editor -> visuals -> LaTeX -> reviewer -> compiler.

Flow, in one paragraph: the Planner turns a topic into a structured
outline (one brief per slide). LangGraph ``Send`` fans those briefs
out to one Writer agent per slide, all running in parallel — each
writer only sees its own brief plus the neighbouring slide titles,
so it knows where it sits without drifting into the next slide's
content. Results are collected by slide number, not by finish order.
The Flow Editor then reads the whole deck in order: narrative arc,
transitions, duplicated points, ordering, tone and length. Only
after that does the deterministic Beamer renderer turn the deck
into LaTeX (never LLM-generated LaTeX), the Reviewer checks it, and
the Compiler writes the files and builds the PDF if an engine exists.

If the Reviewer finds critical issues, the process starts again from
the Planner — a fresh outline, fresh slides — with the reviewer's
findings fed in as feedback, instead of patching the failed deck in
place. Up to 3 full attempts; if the last one still fails, its deck
is compiled as-is with the failing review recorded honestly.

When image sourcing is enabled, a dedicated image agent checks the
topic article's images against slide content before any image is
downloaded or attached.


GPT is the LLM throughout (default gpt-4o-mini, overridable with
--model / OPENAI_MODEL). See llm.py for how the key is reached.
"""
from __future__ import annotations

import json
import operator
import re
import shutil
import subprocess
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .charts import render_chart, validate_chart
from .images import find_article_images, render_selected_images
from .latex_utils import render_beamer, validate_latex
from .llm import get_llm
from .models import ChartSpec, Deck, Outline, ReviewIssue, ReviewResult, SlideBrief, SlideContent

MAX_ATTEMPTS = 3  # full builds: first try + up to 2 restarts


class PresentationState(TypedDict, total=False):
    topic: str
    author: str
    institute: str
    num_slides: int
    model: str
    outline: dict
    slides: Annotated[list[dict], operator.add]
    deck: dict
    latex: str
    review: dict
    attempts: int
    build_attempt: int
    review_issues: str
    output_dir: str
    pdf_path: str | None
    compile_log: str
    compile_error: str
    charts_enabled: bool
    images_enabled: bool
    image_article_title: str
    image_article_url: str
    image_candidates: list[dict]
    image_selections: list[dict]
    chart_count: int
    image_count: int
    source_text: str
    references: list[str]
    comments: str
    # Per-branch Send payload for write_slide:
    brief: dict
    prev_title: str
    next_title: str


def _extract_json(text: str) -> dict:
    """Pull the first {...} JSON object out of model output."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise ValueError(f"No JSON object in model output: {text[:300]}")
        return json.loads(match.group(0))


def _chat_json(llm, system: str, user: str) -> dict:
    """Invoke the LLM and parse a JSON-only reply, with one retry."""
    last_error: Exception | None = None
    prompt_user = user
    for _ in range(2):
        msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt_user)])
        try:
            return _extract_json(str(msg.content))
        except Exception as exc:  # noqa: BLE001 - retry once, then raise
            last_error = exc
            prompt_user = user + "\n\nYour previous reply was not valid JSON. Reply with JSON only, no other text."
    raise RuntimeError(f"Model did not return valid JSON: {last_error}")


_ENDING_WORDS = ("conclusion", "summary", "takeaway", "final thought", "looking ahead", "wrap-up", "wrap up")


# Connectives as well as articles: v10 flagged 'Adoption Across Industries'
# and 'Collaboration Across Vendors' as overlapping just for sharing 'across'.
_TITLE_STOPWORDS = {"the", "a", "an", "and", "or", "of", "for", "in", "on", "to", "with", "how", "what", "why",
                    "across", "between", "beyond", "from", "into", "within", "through", "via", "vs", "versus",
                    "about", "at", "by", "as", "its", "their", "your", "our"}


def norm_title_key(title: str, stopwords: set[str] | None = None) -> str:
    """Title key for near-duplicate detection: lowercase words, minus
    generic words and the deck's own topic words. So 'Gemini Pricing
    Tiers' and 'Pricing Tiers' key the same — the planner/fill kept
    producing both, and the reviewer then failed the deck on the pair."""
    words = re.findall(r"[a-z0-9]+", title.lower())
    drop = _TITLE_STOPWORDS | (stopwords or set())
    return " ".join(sorted(set(words) - drop))


def _is_ending(title: str) -> bool:
    low = title.lower()
    return any(w in low for w in _ENDING_WORDS)


def ending_last(slides: list) -> list:
    """Return slides in number order with ending slides (Conclusion,
    Summary, ...) moved after every content slide. Works for SlideBrief
    and SlideContent. LLM steps renumber slides freely — the 30-slide A2A
    deck had 'A2A Scalability Considerations' after its Conclusion — so
    order is enforced here rather than trusted from their numbers."""
    ordered = sorted(slides, key=lambda s: s.number)
    return [s for s in ordered if not _is_ending(s.title)] + [s for s in ordered if _is_ending(s.title)]


def validate_outline(outline: Outline, n: int, visual_target: int) -> list[str]:
    """Check an outline against the hard rules. Empty list = valid.

    These are the rules the 30-slide run broke: the planner marked
    only 4 visuals instead of 9, and padded the ending with five
    conclusion-like slides (Conclusion, Summary, Looking Ahead,
    Final Thoughts, Takeaways). Validation + retry + deterministic
    repair below exist so that failure cannot silently reach the deck.
    """
    problems: list[str] = []
    if len(outline.slides) != n:
        problems.append(f"Expected {n} slides, got {len(outline.slides)}.")
    titles = [s.title.strip().lower() for s in outline.slides]
    if len(set(titles)) != len(titles):
        dupes = sorted({t for t in titles if titles.count(t) > 1})
        problems.append(f"Duplicate slide titles: {', '.join(dupes)}.")
    visuals = sum(1 for s in outline.slides if s.has_visual)
    if visuals != visual_target:
        problems.append(f"Expected {visual_target} visual slides, got {visuals}.")
    if any(s.has_visual for s in outline.slides if _is_ending(s.title)):
        problems.append("The conclusion slide must not have a visual.")
    endings = [s for s in outline.slides if _is_ending(s.title)]
    if len(endings) != 1:
        problems.append(f"Expected exactly 1 ending slide, got {len(endings)}: {[s.title for s in endings]}.")
    elif outline.slides and outline.slides[-1].title != endings[0].title:
        problems.append("The ending slide must be the last slide.")
    return problems


# Words that may repeat across titles without the slides overlapping
# ("Case Study: Finance" and "Case Study: Retail" are distinct slides).
_OVERLAP_IGNORE = {"case", "study", "studies", "key", "overview", "part", "continued", "consideration", "considerations"}
# Words for the protocol's introduction and backstory, counted as one theme:
# every such slide retells who launched it and when. v11 ('Introduction by
# Google' vs 'A2A Launch Timeline') and v13 ('Introduction to A2A History'
# vs 'Introduction to A2A Protocol') failed review on that repeat, and the
# titles shared no other word.
_ORIGIN_WORDS = {"introduction", "introducing", "intro", "history", "historical", "origin", "origins",
                 "launch", "launched", "timeline", "background", "genesis", "beginnings"}
_ORIGIN_THEME = "origins"
_PLACEHOLDER = re.compile(r"\b(?i:company|organi[sz]ation|client|vendor|firm|customer|enterprise|business)\s+[A-Z]\b|\b(?i:acme|xyz corp)\b")


_SUFFIXES = ("ations", "ation", "atives", "ative", "isons", "ison", "ities", "ity", "ments", "ment",
             "ings", "ing", "ions", "ion", "ives", "ive", "ed", "es", "s")


def _stem(word: str) -> str:
    """Crude suffix strip so 'Comparison' / 'Comparative' and 'Integration'
    / 'Integrating' count as the same theme (v6 shipped 'Comparison with
    MCP' next to 'Comparative Analysis of Protocols')."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def _content_words(title: str, topic_words: set[str]) -> set[str]:
    raw = {w for w in re.findall(r"[a-z0-9]+", title.lower()) if len(w) > 2}
    words = {_ORIGIN_THEME if w in _ORIGIN_WORDS else _stem(w) for w in raw}
    return words - {_stem(w) for w in _TITLE_STOPWORDS | _OVERLAP_IGNORE | topic_words}


SAME_TOPIC_ROUNDS = 3


def same_topic_slides(outline: Outline, llm) -> set[int]:
    """Ask the model which slides repeat an earlier slide's sub-topic under
    different words — the word checks cannot see that 'Industry
    Applications' repeats 'Use Cases of A2A' (v6 failed review on it).
    Returns the numbers of the LATER slide of each pair; best effort."""
    body = [s for s in outline.slides if not _is_ending(s.title)]
    system = (
        "You check a presentation outline for repeated sub-topics. Output JSON only. "
        "Schema: {\"pairs\": [{\"earlier\": int, \"later\": int, \"reason\": str}]}. "
        "List a pair only when the two slides would teach the audience substantially the same "
        "thing (same sub-topic under different words, e.g. 'Use Cases' and 'Industry Applications', "
        "or 'Future Directions' and 'Vision for the Future'). Slides on related but distinct aspects "
        "are not pairs. Return {\"pairs\": []} if there are none."
    )
    user = "\n".join(f"{s.number}. {s.title} — {s.brief}" for s in body)
    numbers = {s.number for s in body}
    # Deterministic judge: at the planner's temperature (0.7) three checks
    # of the same v8 outline flagged three different sets of slides.
    if "temperature" in getattr(type(llm), "model_fields", {}):
        llm = llm.model_copy(update={"temperature": 0})
    try:
        data = _chat_json(llm, system, user)
    except Exception:
        return set()
    later: set[int] = set()
    for pair in data.get("pairs", []) or []:
        try:
            a, b = int(pair["earlier"]), int(pair["later"])
        except (KeyError, TypeError, ValueError):
            continue
        if a != b and a in numbers and b in numbers:
            later.add(max(a, b))
    return later


def outline_flow_problems(outline: Outline, topic: str = "") -> list[str]:
    """Soft flow checks fed back to the planner on retry (never repaired
    deterministically). The 30-slide A2A deck shipped 'Communication
    Protocols' / 'Communication Standards', 'Security Considerations' /
    'Security Features', 'Integration Challenges' / 'Challenges Faced',
    and placeholder 'Case Study: Company A / Company B' slides."""
    topic_words = set(re.findall(r"[a-z0-9]+", f"{topic} {outline.title}".lower()))
    body = [s for s in outline.slides if not _is_ending(s.title)]
    problems: list[str] = []
    words = [(s, _content_words(s.title, topic_words)) for s in body]
    for i, (a, wa) in enumerate(words):
        for b, wb in words[i + 1:]:
            shared = wa & wb
            if shared:
                problems.append(
                    f"Slides '{a.title}' and '{b.title}' overlap (both about {', '.join(sorted(shared))}): "
                    "merge them or give each a clearly distinct sub-topic.")
    for slide in outline.slides:
        if _PLACEHOLDER.search(f"{slide.title} {slide.brief}"):
            problems.append(
                f"Slide '{slide.title}' uses a placeholder name (e.g. 'Company A'): name a real organization "
                "from the source material, or replace the slide with a different sub-topic.")
    return problems


def resolve_overlaps(outline: Outline, topic: str, llm, source_text: str = "",
                     same_topic: set[int] | None = None) -> Outline:
    """Replace each slide that overlaps an EARLIER slide (shared title
    theme) or uses a placeholder name with a fresh, distinct brief, in
    the same position and keeping its chart slot. Runs once the planner's
    own retries still leave flow problems — it ignored the feedback for
    'Future of A2A' / 'Future Challenges' / 'Potential for Future
    Protocols' and 'Community Support' / 'Community Contributions'.
    same_topic: slide numbers to replace as well (from same_topic_slides)."""
    same_topic = same_topic or set()
    topic_words = set(re.findall(r"[a-z0-9]+", f"{topic} {outline.title}".lower()))
    covered: set[str] = set()
    replace: list[SlideBrief] = []
    for slide in outline.slides:
        if _is_ending(slide.title):
            continue
        words = _content_words(slide.title, topic_words)
        if (words & covered) or slide.number in same_topic or _PLACEHOLDER.search(f"{slide.title} {slide.brief}"):
            replace.append(slide)
        else:
            covered |= words
    if not replace:
        return outline
    # Every current title, replaced ones included, so the fresh briefs
    # don't come back as the same sub-topic.
    all_titles = [s.title for s in outline.slides]
    try:
        # Ask for spares: one candidate per slot kept reusing a covered word
        # ('challenge'), so v10 shipped 'Challenges in Adoption',
        # 'Challenges of A2A Implementation' and 'Challenges in Implementation'.
        fresh = _fill_briefs(llm, topic, outline.title, all_titles, 2 * len(replace) + 2, 0,
                             avoid_words=sorted(covered), source_text=source_text)
    except Exception:
        return outline  # best effort: an overlapping deck beats a broken plan
    accepted: dict[int, SlideBrief] = {}
    candidates = iter(fresh)
    for old in replace:
        new = None
        for cand in candidates:
            words = _content_words(cand.title, topic_words)
            if words and not words & covered and not _is_ending(cand.title):
                new = cand
                break
        if new is None:
            break  # out of candidates: keep the remaining originals
        covered |= words
        new.number, new.has_visual = old.number, old.has_visual
        new.visual_hint = f"Quantitative comparison or trend for: {new.title}" if old.has_visual else ""
        accepted[old.number] = new
    slides = [accepted.get(s.number, s) for s in outline.slides]
    return outline.model_copy(update={"slides": slides})


def _fill_briefs(llm, topic: str, title: str, existing_titles: list[str], count: int, start_number: int,
                 avoid_words: list[str] | None = None, source_text: str = "") -> list[SlideBrief]:
    """Ask the planner for `count` extra content briefs (never endings).
    avoid_words: themes already covered, which new titles must not reuse.
    source_text: research/source material the new briefs must stay within."""
    system = (
        "You plan presentations. Output JSON only. "
        "Schema: {\"slides\": [{\"number\": int, \"title\": str, \"brief\": str, \"key_points\": [str], "
        "\"has_visual\": false, \"visual_hint\": \"\"}]}. "
        f"Produce exactly {count} new content slides for the presentation '{title}'. "
        "Titles must not duplicate any existing title and must not be conclusion/summary/takeaway slides. "
        "Keep titles short (<= 8 words), briefs to 1-2 sentences."
    )
    user = f"Topic: {topic}\nExisting titles (do not repeat): {', '.join(existing_titles)}"
    if avoid_words:
        user += ("\nThese themes are already covered — each new slide must be about something else, and its "
                 f"title must not use any of these words: {', '.join(avoid_words)}")
    if source_text:
        user += f"\n\nBase every new slide on facts in this source material only:\n{source_text}"
    data = _chat_json(llm, system, user)
    briefs = [SlideBrief.model_validate(b) for b in data.get("slides", [])]
    for i, b in enumerate(briefs):
        b.number = start_number + i
        b.has_visual = False
        b.visual_hint = ""
    return briefs[:count]


def _repair_outline(outline: Outline, n: int, visual_target: int, llm=None, topic: str = "") -> Outline:
    """Deterministic repair when the planner still fails validation.

    Keeps the first occurrence of a duplicated title, keeps only the
    last ending slide (moved to the end), fixes the visual count by
    marking evenly-spaced non-ending slides, and fills any shortfall
    with fresh content briefs so the count stays at exactly n.
    """
    seen: set[str] = set()
    kept: list[SlideBrief] = []
    for s in sorted(outline.slides, key=lambda b: b.number):
        key = s.title.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(s)
    # Exactly one ending slide, at the end.
    endings = [s for s in kept if _is_ending(s.title)]
    if endings:
        # Exactly one ending, and it goes last — also when there is only
        # one but it sits mid-deck.
        kept = [s for s in kept if not _is_ending(s.title)] + [endings[-1]]
    elif kept:
        kept[-1].title = "Conclusion and Summary"
    # Visual count: turn off extras, then turn on evenly-spaced slides.
    visuals = [s for s in kept if s.has_visual and not _is_ending(s.title)]
    for s in kept:
        s.has_visual = False
        s.visual_hint = ""
    if visual_target:
        candidates = [s for s in kept[:-1]]  # never the last slide
        step = max(1, len(candidates) // visual_target) if candidates else 1
        chosen = candidates[::step][:visual_target]
        # If stepping skipped too many, fill from the front.
        for s in candidates:
            if len(chosen) >= visual_target:
                break
            if s not in chosen:
                chosen.append(s)
        for s in chosen:
            s.has_visual = True
            s.visual_hint = f"Quantitative comparison or trend for: {s.title}"
    # Fill any shortfall with fresh briefs, inserted before the ending.
    if len(kept) < n and llm is not None:
        try:
            extra = _fill_briefs(llm, topic, outline.title, [s.title for s in kept], n - len(kept), len(kept))
            ending = kept[-1] if kept and _is_ending(kept[-1].title) else None
            body = kept[:-1] if ending else kept
            kept = body + extra + ([ending] if ending else [])
        except Exception:
            pass
    for i, s in enumerate(kept, start=1):
        s.number = i
    return Outline(title=outline.title, audience=outline.audience, style_guide=outline.style_guide, slides=kept[:n])


# ---------------------------------------------------------------- planner
def plan_outline(topic: str, author: str, num_slides: int, model: str | None,
                 charts_enabled: bool, source_text: str = "", review_feedback: str = "") -> Outline:
    """Plan + validate + repair. Pure function, shared by the LangGraph
    node and by the A2A Planner agent — one implementation, two transports.
    review_feedback is the previous build's reviewer findings, passed in
    on a restart so the new outline cannot repeat the same failure."""
    llm = get_llm(model)
    n = num_slides
    visual_target = max(0, (n - 1) // 3) if charts_enabled else 0
    system = (
        "You plan presentations. Output JSON only, no other text. "
        "Schema: {\"title\": str, \"audience\": str, \"style_guide\": str, "
        "\"slides\": [{\"number\": int, \"title\": str, \"brief\": str, \"key_points\": [str], "
        "\"has_visual\": bool, \"visual_hint\": str}]}. "
        f"Produce exactly {n} slides. "
        "The last slide, and ONLY the last slide, must be a conclusion / summary. "
        "Do not add any other ending-like slides (no separate Summary, Takeaways, Final Thoughts, or Looking Ahead slides). "
        "All other slides must be distinct content slides with distinct titles, each covering a different "
        "sub-topic: never two slides on the same theme (e.g. not both 'Security Features' and "
        "'Security Considerations' — make one 'Security' slide). Order slides so each builds on the last. "
        "Never use placeholder names like 'Company A' or 'Organization X'; a case-study slide must name "
        "a real organization from the source material, and if there is none, choose another sub-topic. "
        "Keep titles short (<= 8 words) and briefs to 1-2 sentences. "
        f"Mark exactly {visual_target} slides has_visual=true, spread through the deck, never the last slide. "
        "Each visual slide needs a visual_hint naming a real quantitative comparison, trend, or share "
        "that can be charted in 3-8 numbers (e.g. 'adoption by year', 'share by sector'). "
        "No visual on a slide that has no numbers to show."
    )
    if source_text:
        system += (
            " SOURCE RULE: base the outline ONLY on facts in the source material below. "
            "Do not invent facts, numbers, names, or claims that are not in the source. "
            "If the source is thin on a sub-topic, say so in the brief rather than inventing it."
        )
    user = f"Topic: {topic}\nAuthor: {author}\nAudience: general, curious non-experts unless the topic says otherwise."
    if source_text:
        user += f"\n\nSource material (ground every slide in this):\n{source_text}"
    if review_feedback:
        user += (
            "\n\nA previous build of this deck FAILED review for these reasons:\n"
            f"{review_feedback}\nFix every issue in this new outline — change titles, briefs "
            "and visuals so the failure cannot repeat."
        )
    outline: Outline | None = None
    problems: list[str] = []
    flow: list[str] = []
    base_user = user
    for _attempt in range(3):
        if problems or flow:
            # Keep the source material and review feedback on retries: a
            # retry without them planned from model knowledge alone, which
            # is how ungrounded 'Company A' slides reached the deck.
            user = (
                base_user + "\n\nYour previous outline had these problems:\n"
                + "\n".join(f"- {p}" for p in problems + flow)
                + "\nFix every problem above and return a corrected outline, JSON only."
            )
        outline = Outline.model_validate(_chat_json(llm, system, user))
        problems = validate_outline(outline, n, visual_target)
        flow = outline_flow_problems(outline, topic)
        if not problems and not flow:
            break
    assert outline is not None
    # Only the hard rules are repaired; leftover flow problems are left to
    # the editor and reviewer rather than forcing a mechanical fix.
    if problems:
        outline = _repair_outline(outline, n, visual_target, llm=llm, topic=topic)
    # Check again after each replacement: fresh briefs are only screened by
    # words, and in v8 they brought back 'Overview of A2A Protocol' next to
    # the Introduction and a second comparison slide.
    for _round in range(SAME_TOPIC_ROUNDS):
        same_topic = same_topic_slides(outline, llm)
        if not same_topic and not outline_flow_problems(outline, topic):
            break
        outline = resolve_overlaps(outline, topic, llm, source_text=source_text, same_topic=same_topic)
    return outline


def planner(state: PresentationState) -> dict:
    # On a restart this is a fresh build: the reviewer's findings become
    # planner feedback, and every artifact of the failed build is cleared
    # so nothing stale (old deck, old LaTeX, old issues) leaks into it.
    feedback = state.get("review_issues", "")
    outline = plan_outline(
        state["topic"], state.get("author", ""), state.get("num_slides", 6),
        state.get("model"), bool(state.get("charts_enabled", True)),
        source_text=state.get("source_text", ""),
        review_feedback=feedback,
    )
    return {
        "outline": outline.model_dump(), "attempts": 0, "slides": [],
        "deck": {}, "latex": "", "chart_count": 0, "review_issues": "", "compile_error": "",
        "build_attempt": state.get("build_attempt", 0) + 1,
    }


def route_planner(state: PresentationState):
    """Fan out: one Send per slide brief -> one writer agent per slide, in parallel."""
    outline = Outline.model_validate(state["outline"])
    briefs = sorted(outline.slides, key=lambda b: b.number)
    sends = []
    for i, brief in enumerate(briefs):
        sends.append(
            Send(
                "write_slide",
                {
                    **state,
                    "brief": brief.model_dump(),
                    "prev_title": briefs[i - 1].title if i > 0 else "",
                    "next_title": briefs[i + 1].title if i + 1 < len(briefs) else "",
                },
            )
        )
    return sends


# ------------------------------------------------------------- slide writer
def write_one_slide(brief: SlideBrief, outline: Outline, prev_title: str, next_title: str,
                    model: str | None, source_text: str = "") -> SlideContent:
    """Write one slide. Pure function, shared by the LangGraph node and
    by the A2A Writer agent — one implementation, two transports."""
    llm = get_llm(model)
    chart_schema = (
        "\"chart\": {\"chart_type\": \"bar|line|pie\", \"title\": str, \"x_label\": str, \"y_label\": str, "
        "\"labels\": [str, 3-8 items], \"values\": [number, same length as labels], "
        "\"source_note\": str} or null. "
        "Chart values must be plausible, illustrative numbers for the topic — never invent a precise "
        "source or citation. Keep source_note as 'Illustrative data — verify before presenting.' "
        "unless the numbers are well-known facts."
    )
    system = (
        "You write one presentation slide. Output JSON only, no other text. "
        "Schema: {\"number\": int, \"title\": str, \"bullets\": [str], \"notes\": str, " + chart_schema + "}. "
        "Bullets: 3-5 items, each one idea, plain text, no LaTeX, no markdown. "
        "Notes: speaker notes, 1-3 sentences, plain text. "
        "Stay strictly on your brief — do not cover the neighbouring slides' topics. "
        + ("Include a chart for this slide, matching the visual hint. " if brief.has_visual else "Set chart to null — no visual on this slide. ")
    )
    user = (
        f"Presentation: {outline.title}\nStyle: {outline.style_guide}\n"
        f"Previous slide: {prev_title or '(none — this is first)'}\n"
        f"Next slide: {next_title or '(none — this is last)'}\n"
        f"Your slide #{brief.number}: {brief.title}\nBrief: {brief.brief}\n"
        f"Key points to cover: {', '.join(brief.key_points) or '(use your judgement)'}"
        + (f"\nVisual hint: {brief.visual_hint}" if brief.has_visual else "")
        + (f"\n\nSource material — use only facts from here, never invent: \n{source_text}" if source_text else "")
    )
    if source_text:
        system += " SOURCE RULE: every bullet must be traceable to the source material. Do not invent facts or numbers not in it." 
    try:
        slide = SlideContent.model_validate(_chat_json(llm, system, user))
    except Exception:
        # One failed slide must not kill a 30-slide deck: fall back to the brief.
        slide = SlideContent(
            number=brief.number,
            title=brief.title,
            bullets=(brief.key_points or [brief.brief])[:5],
            notes=brief.brief,
            chart=None,
        )
    slide.number = brief.number
    slide.title = brief.title
    if not brief.has_visual:
        slide.chart = None
    if slide.chart is not None and validate_chart(slide.chart):
        slide.chart = None  # invalid chart spec fails safe: no chart, bullets stay
    return slide


def write_slide(state: PresentationState) -> dict:
    brief = SlideBrief.model_validate(state["brief"])
    outline = Outline.model_validate(state["outline"])
    slide = write_one_slide(brief, outline, state.get("prev_title") or "", state.get("next_title") or "",
                          state.get("model"), source_text=state.get("source_text", ""))
    # Tag the slide with its build: the slides channel is an
    # operator.add accumulator, so a restart would otherwise leave the
    # failed build's slides in state and the editor would see both
    # builds at once. The tag is dropped by SlideContent validation.
    slide_dict = slide.model_dump()
    slide_dict["build_attempt"] = state.get("build_attempt", 1)
    return {"slides": [slide_dict]}


# -------------------------------------------------------------- flow editor
def _slide_text(s: SlideContent) -> str:
    text = f"Slide {s.number}: {s.title}\n" + "\n".join(f"- {b}" for b in s.bullets) + f"\nNotes: {s.notes}"
    if s.chart:
        text += f"\nChart ({s.chart.chart_type}): {s.chart.title} — {', '.join(f'{l}={v:g}' for l, v in zip(s.chart.labels, s.chart.values))}"
    return text


_BULLET_STOPWORDS = _TITLE_STOPWORDS | {"is", "are", "be", "by", "as", "its", "it", "this", "that", "can", "include", "includes", "including", "such", "like", "use", "uses"}
BULLET_OVERLAP = 0.5  # token Jaccard at which two bullets count as the same point


def _bullet_tokens(text: str, topic_words: set[str]) -> set[str]:
    words = {w.rstrip("s") for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}
    return words - {w.rstrip("s") for w in _BULLET_STOPWORDS | topic_words}


def drop_repeated_bullets(slides: list[SlideContent], topic_words: set[str], min_bullets: int = 3) -> list[SlideContent]:
    """Drop a bullet that repeats a point made on an EARLIER slide (token
    Jaccard >= BULLET_OVERLAP, topic words ignored), as long as the slide
    keeps min_bullets. The 30-slide A2A deck failed review on 'signed
    agent cards for identity verification' appearing on slides 20 and 21,
    even though the chunk editor saw both."""
    seen: list[set[str]] = []
    for slide in sorted(slides, key=lambda s: s.number):
        if slide.title.strip().lower().startswith("references"):
            continue
        kept: list[str] = []
        tokens_kept: list[set[str]] = []
        for i, bullet in enumerate(slide.bullets):
            tokens = _bullet_tokens(bullet, topic_words)
            repeat = bool(tokens) and any(len(tokens & t) / len(tokens | t) >= BULLET_OVERLAP for t in seen)
            remaining = len(slide.bullets) - i - 1
            if repeat and len(kept) + remaining >= min_bullets:
                continue  # drop: the point was already made, and the slide stays full enough
            kept.append(bullet)
            tokens_kept.append(tokens)
        slide.bullets = kept
        seen.extend(tokens_kept)
    return slides


def edit_deck(outline: Outline, slides: list[SlideContent], model: str | None, review_issues: str = "", topic: str = "", num_slides: int | None = None, source_material: str = "", source_text: str = "") -> Deck:
    """Edit a deck in order; fix flow, duplication, ordering, tone.

    Chunked in groups of 6 slides: a 30-slide deck regenerated in one
    LLM reply would be ~7k tokens and risks truncation. Each chunk is
    edited with its neighbours' full text as context (not just titles —
    the 30-slide duplicate sat across a chunk boundary and titles alone
    did not expose it). The editor MAY drop a slide that duplicates
    another; if that leaves the deck short, fresh content slides are
    written to fill it before the ending slide. Pure function, shared
    by the LangGraph node and by the A2A Editor agent.
    """
    llm = get_llm(model)
    slides = sorted(slides, key=lambda s: s.number)
    issues = review_issues
    system = (
        "You are the flow editor for a presentation. Output JSON only, no other text. "
        "Schema: {\"title\": str, \"slides\": [{\"number\": int, \"title\": str, \"bullets\": [str], \"notes\": str, "
        "\"chart\": {\"chart_type\": str, \"title\": str, \"x_label\": str, \"y_label\": str, \"labels\": [str], \"values\": [number], \"source_note\": str} | null}]}. "
        "Keep the same slides and titles in this chunk unless a title is misleading. "
        "You MAY drop a slide in this chunk if it duplicates another slide's title or bullets — "
        "a duplicate must be removed, not rewritten. Otherwise keep every slide. "
        "Your jobs: (1) cut or rewrite any point duplicated within the chunk or with the neighbouring slides; "
        "(2) add a transition where a jump feels abrupt — fold it into the next slide's first bullet or its notes; "
        "(3) even out tone and bullet length; never keep a point that the rest of the deck already makes; (4) every slide keeps 3-5 bullets and 1-3 sentences of notes; "
        "(5) preserve each chart exactly by default, but obey an explicit user request to change a chart's type, title, labels, or placement. "
        "Never invent or alter numeric values unless the user supplied them in the request or source material; "
        "when a requested trend needs missing time-series values, keep the existing values and say what data is needed. "
        "Return the full edited chunk, not just the changes."
    )
    edited: list[SlideContent] = []
    chunk_size = 6
    for start in range(0, len(slides), chunk_size):
        chunk = slides[start : start + chunk_size]
        chunk_text = "\n\n".join(_slide_text(s) for s in chunk)
        # Full neighbour text, not just titles: the 30-slide duplicate
        # sat across a chunk boundary and titles alone did not expose it.
        neighbours = []
        if start > 0:
            neighbours.append("Previous slide (for duplication check):\n" + _slide_text(slides[start - 1]))
        if start + chunk_size < len(slides):
            neighbours.append("Next slide (for duplication check):\n" + _slide_text(slides[start + chunk_size]))
        # The rest of the deck, compactly: a chunk editor that only sees its
        # neighbours cannot tell a point is already made 10 slides earlier.
        others = [x for x in slides if x not in chunk]
        deck_map = "\n".join(f"Slide {x.number}: {x.title} — " + " | ".join(x.bullets) for x in others)
        user = (f"Presentation: {outline.title}\nStyle: {outline.style_guide}\n"
                + (f"Rest of the deck (do not repeat any point made there):\n{deck_map}\n\n" if deck_map else "")
                + "\n\n".join(neighbours) + f"\n\n{chunk_text}")
        if issues:
            user += f"\n\nReviewer issues from the previous round — fix all that apply to this chunk:\n{issues}"
        if source_material:
            user += ("\n\nAdditional source material supplied for this revision. Apply it only to the slide(s) "
                     "targeted by the user's comments. Ground factual changes in this material and do not "
                     "invent facts:\n" + source_material)
        try:
            fragment = Deck.model_validate(_chat_json(llm, system, user))
            # Only this chunk's slides: the editor sees the rest of the deck
            # and in v11 the last chunk returned slides 8-12 (dropped from
            # their own chunk), which moved them to the end of the deck.
            chunk_keys = {norm_title_key(s.title) for s in chunk}
            foreign = {norm_title_key(s.title) for s in slides if s not in chunk} - chunk_keys
            own = [s for s in fragment.slides if norm_title_key(s.title) not in foreign]
            if len(own) < len(chunk) - 1:
                raise ValueError("chunk editor dropped more than one slide")  # keep the chunk as written
            # Trust the chunk's order, not its numbers: the LLM often
            # renumbers a chunk from 1, and sorting the whole deck by those
            # numbers interleaved chunks (the 30-slide A2A deck put its
            # Conclusion at slide 19).
            edited.extend(sorted(own, key=lambda x: x.number))
        except Exception:
            edited.extend(chunk)  # editor failure must not lose slides
    for position, slide in enumerate(edited, start=1):
        slide.number = position
    # Deterministic dedupe: same title twice (normalised — "Gemini
    # Pricing Tiers" and "Pricing Tiers" are the same slide), or
    # near-identical bullets.
    topic_words = set(re.findall(r"[a-z0-9]+", f"{topic} {outline.title}".lower()))
    edited = drop_repeated_bullets(edited, topic_words)
    deduped: list[SlideContent] = []
    seen_titles: set[str] = set()
    for s in sorted(edited, key=lambda x: x.number):
        key = norm_title_key(s.title, topic_words)
        if key in seen_titles:
            continue
        seen_titles.add(key)
        deduped.append(s)
    edited = deduped
    # Chart preservation: the editor must not silently drop a chart.
    # The A2A run lost 2 of 9 charts here (slides 5 and 7) because the
    # chunk LLM returned chart=null. Restore any chart the input had.
    input_charts = {s.title.strip().lower(): s.chart for s in slides if s.chart is not None}
    for s in edited:
        if s.chart is None and s.title.strip().lower() in input_charts:
            s.chart = input_charts[s.title.strip().lower()]
    # Fill any shortfall from dropped duplicates, before the ending slide.
    target = num_slides or len(slides)
    if len(edited) < target and topic:
        try:
            ending = edited[-1] if edited and _is_ending(edited[-1].title) else None
            body = edited[:-1] if ending else edited
            # Grounded in the research and screened against covered themes:
            # the ungrounded fill invented v10's off-outline 'Performance
            # Metrics for A2A' after the dedupe dropped a duplicate slide.
            covered = sorted(set().union(*(_content_words(s.title, topic_words) for s in edited)))
            extra_briefs = _fill_briefs(llm, topic, outline.title, [s.title for s in edited], target - len(edited),
                                        len(body) + 1, avoid_words=covered, source_text=source_text)
            for brief in extra_briefs:
                prev_t = body[-1].title if body else ""
                next_t = ending.title if ending else ""
                body.append(write_one_slide(brief, outline, prev_t, next_t, model, source_text=source_text))
            edited = body + ([ending] if ending else [])
        except Exception:
            pass
    # The fill above can itself add a near-duplicate of an existing
    # title, so dedupe once more before renumbering.
    _seen: set[str] = set()
    _final: list[SlideContent] = []
    for s in sorted(edited, key=lambda x: x.number):
        _key = norm_title_key(s.title, topic_words)
        if _key in _seen:
            continue
        _seen.add(_key)
        _final.append(s)
    edited = _final
    # Hard cap: the deck must have exactly `target` slides. The chunk
    # editor can ADD slides (it returned 9 for a 6-slide deck on a
    # restart) and nothing downstream checks the count, so the extra
    # slides used to reach the PDF silently. Trim deterministically:
    # first (target-1) content slides plus the ending slide last.
    if len(edited) > target:
        endings = [s for s in edited if _is_ending(s.title)]
        ending = endings[-1] if endings else None
        body = [s for s in edited if not _is_ending(s.title)]
        edited = (body[: target - 1] + [ending]) if ending is not None else body[:target]
    # Normalise numbering in final order, the ending slide always last.
    edited = ending_last(edited)
    for i, slide in enumerate(edited, start=1):
        slide.number = i
    return Deck(title=outline.title, slides=edited)


def editor(state: PresentationState) -> dict:
    """LangGraph node. If a deck is already in state (only on a
    restart, and the planner has cleared it by then) start from it;
    otherwise start from the raw writer slides."""
    outline = Outline.model_validate(state["outline"])
    if state.get("deck"):
        slides = [SlideContent.model_validate(s) for s in Deck.model_validate(state["deck"]).slides]
    else:
        # Only this build's slides: the accumulator still holds the
        # failed attempts' slides (see write_slide's build_attempt tag).
        current = state.get("build_attempt", 1)
        # Strict match: an untagged slide is never this build's. The old
        # default (untagged == current) let every earlier build's edited
        # slides back in after a restart, mixing builds in one deck.
        raw = [s for s in state.get("slides", []) if s.get("build_attempt") == current]
        slides = [SlideContent.model_validate(s) for s in raw]
    comments_raw = state.get("comments", "")
    if comments_raw:
        from .comments import apply_comments, parse_comments

        deck = apply_comments(
            Deck(title=outline.title, slides=slides), outline,
            parse_comments(text=comments_raw), state.get("model"),
            topic=state.get("topic", ""),
        )
    else:
        deck = edit_deck(
            outline, slides, state.get("model"),
            review_issues=state.get("review_issues", ""),
            topic=state.get("topic", ""),
            num_slides=state.get("num_slides"),
            source_text=state.get("source_text", ""),
        )
    # Only the deck: the slides channel is an append-only accumulator of
    # writer output, and writing edited (untagged) slides back into it
    # leaked them into the next build after a restart.
    return {"deck": deck.model_dump()}


# ------------------------------------------------------------------ charts
def render_deck_charts(deck: Deck, output_dir: str) -> tuple[Deck, int]:
    """Render every valid chart spec to PNG. Pure function, shared by
    the LangGraph node and by the A2A Chart agent."""
    out_dir = Path(output_dir or "out")
    count = 0
    slides = []
    for slide in sorted(deck.slides, key=lambda s: s.number):
        if slide.chart is not None and not validate_chart(slide.chart):
            path = render_chart(slide.chart, out_dir / "charts" / f"slide_{slide.number:02d}.png")
            if path is not None:
                slide.chart_file = f"charts/slide_{slide.number:02d}.png"
                count += 1
            else:
                slide.chart = None
                slide.chart_file = None
        else:
            slide.chart = None
            slide.chart_file = None
        slides.append(slide.model_dump())
    return Deck(title=deck.title, slides=[SlideContent.model_validate(s) for s in slides]), count


def render_charts(state: PresentationState) -> dict:
    """Render every valid chart spec to PNG, deterministically."""
    deck, count = render_deck_charts(Deck.model_validate(state["deck"]), state.get("output_dir") or "out")
    return {"deck": deck.model_dump(), "chart_count": count}


def image_agent(state: PresentationState) -> dict:
    """Find topic-article images and let the image agent match them to slides."""
    if not state.get("images_enabled", False):
        return {"image_candidates": [], "image_selections": [], "image_count": 0}
    try:
        article_title, article_url, candidates = find_article_images(state.get("topic", ""))
    except Exception:
        article_title, article_url, candidates = "", "", []
    if not candidates:
        return {
            "image_article_title": article_title, "image_article_url": article_url,
            "image_candidates": [], "image_selections": [], "image_count": 0,
        }
    from .image_agent import select_relevant_images

    selections = select_relevant_images(state["deck"], candidates, state.get("model"))
    return {
        "image_article_title": article_title, "image_article_url": article_url,
        "image_candidates": candidates, "image_selections": selections,
    }


def render_images(state: PresentationState) -> dict:
    """Download only slide/image assignments selected by the image agent."""
    deck = Deck.model_validate(state["deck"])
    deck, count = render_selected_images(
        deck, state.get("output_dir") or "out",
        state.get("image_article_title", ""), state.get("image_article_url", ""),
        state.get("image_candidates", []), state.get("image_selections", []),
    )
    return {"deck": deck.model_dump(), "image_count": count}


REFS_PER_SLIDE = 6  # rendered \\scriptsize; at 4, 15 sources took 4 slides


def _is_references(slide: SlideContent) -> bool:
    return slide.title.strip().lower().startswith("references")


def add_references(state: PresentationState) -> dict:
    """Append a references slide, including research, supplied, and image sources."""
    deck = Deck.model_validate(state["deck"])
    refs = list(state.get("references", []))
    for slide in deck.slides:
        if slide.image_source:
            refs.append(slide.image_source)
    # Stable dedupe while retaining the source order.
    refs = list(dict.fromkeys(ref.strip() for ref in refs if ref.strip()))
    if not refs:
        refs = ["No external sources supplied; content generated from model knowledge."]
    deck.slides = [s for s in deck.slides if not _is_references(s)]
    # Keep each citation frame short enough to fit while preserving every source.
    starts = range(0, len(refs), REFS_PER_SLIDE)
    for chunk_index, start in enumerate(starts):
        title = "References" if len(starts) == 1 else f"References ({chunk_index + 1} of {len(starts)})"
        deck.slides.append(SlideContent(
            number=len(deck.slides) + 1,
            title=title,
            bullets=refs[start:start + REFS_PER_SLIDE],
            notes="Sources used to prepare this presentation.",
        ))
    return {"deck": deck.model_dump()}


# ------------------------------------------------------------------- latex
def render_latex(state: PresentationState) -> dict:
    # The post-edit slide content is authoritative: render each final slide
    # as its own Beamer frame (latex_utils.render_slide) and assemble them.
    deck = Deck.model_validate(state["deck"])
    return {"latex": render_beamer(deck, state.get("author", ""), state.get("institute", ""))}


# ----------------------------------------------------------------- reviewer
def review_deck(deck: Deck, latex: str, chart_count: int, model: str | None,
                compile_error: str = "") -> ReviewResult:
    """Review a deck + its LaTeX. Pure function, shared by the LangGraph
    node and by the A2A Reviewer agent."""
    llm = get_llm(model)
    problems = validate_latex(latex, len(deck.slides))
    if compile_error:
        # The tail of the log holds the error; the full log would swamp the prompt.
        problems.append(f"LaTeX compiler failed: {compile_error[-1500:]}")
    system = (
        "You review a presentation deck and its LaTeX. Output JSON only. "
        "Schema: {\"passed\": bool, \"issues\": [{\"severity\": str, \"slide\": int|null, \"issue\": str, \"fix\": str}], \"summary\": str}. "
        "Check: each slide has 3-5 bullets, notes present, no duplicated points, titles match content, "
        "no factual howlers, no empty slides. Severity is one of critical, major, minor. "
        "Use critical ONLY for duplicated content, empty slides, wrong titles, or invalid LaTeX/charts. "
        "'Lacks detail' or 'lacks timelines' is major at most, never critical. "
        "Set passed=false only if there is at least one critical issue."
    )
    chart_problems: list[str] = []
    for s in deck.slides:
        if s.chart is not None:
            chart_problems.extend(f"Slide {s.number} chart: {p}" for p in validate_chart(s.chart))
    problems = problems + chart_problems
    user = (
        f"Title: {deck.title}\nCharts: {chart_count} rendered\n"
        + "\n\n".join(
            f"Slide {s.number}: {s.title}\n" + "\n".join(f"- {b}" for b in s.bullets)
            + (f"\nChart: {s.chart.title} ({s.chart.chart_type})" if s.chart else "")
            for s in deck.slides if not _is_references(s)
        )
    )
    try:
        result = ReviewResult.model_validate(_chat_json(llm, system, user))
    except Exception:
        result = ReviewResult(passed=not problems, summary="LLM review failed; relying on structural checks.")
    # References slides are built deterministically from the source list, so
    # they are left out of the LLM review (it read the split as duplication).
    reference_numbers = {s.number for s in deck.slides if _is_references(s)}
    issues = [i for i in result.issues if i.slide not in reference_numbers]
    for problem in problems:
        issues.insert(0, ReviewIssue(severity="critical", issue=problem, fix="Fix the LaTeX rendering."))
    if problems:
        result.passed = False
    result.issues = issues
    # Blocking rule: only critical issues fail the deck. The LLM's own
    # passed flag is advisory — it failed a 30-slide deck 3 rounds in a
    # row on 'lacks detail' / 'lacks timelines' alone.
    result.passed = not any(i.severity == "critical" for i in issues)
    return result


def reviewer(state: PresentationState) -> dict:
    result = review_deck(
        Deck.model_validate(state["deck"]), state.get("latex", ""),
        state.get("chart_count", 0), state.get("model"), state.get("compile_error", ""),
    )
    attempts = state.get("attempts", 0) + 1
    return {
        "review": result.model_dump(),
        "attempts": attempts,
        "review_issues": json.dumps([i.model_dump() for i in result.issues], indent=2),
    }


def route_reviewer(state: PresentationState) -> str:
    """Reviewer verdict: a passing deck compiles. A failing deck
    starts the process again from the planner — with the reviewer's
    findings as feedback — up to MAX_ATTEMPTS full builds. After the
    last attempt the deck compiles as-is and the failing review is
    recorded honestly in review.json."""
    review = ReviewResult.model_validate(state["review"])
    if review.passed:
        return "compile"
    if state.get("build_attempt", 1) < MAX_ATTEMPTS:
        return "restart"
    return "compile"


# ----------------------------------------------------------------- compiler
def _compile_tex(out_dir: Path) -> tuple[str | None, str, str]:
    """Compile out_dir/presentation.tex. Returns (pdf_path, log, error).

    Shared by the pre-review check and the final compile so both use the
    same engine choice, pass count and failure handling. pdflatex runs
    twice so the Madrid footline's frame totals and navigation resolve.
    """
    tex_file = out_dir / "presentation.tex"
    pdf_file = out_dir / "presentation.pdf"
    pdf_file.unlink(missing_ok=True)  # never mistake a previous build for this one
    if shutil.which("tectonic"):
        commands = [["tectonic", "-X", "compile", str(tex_file), "--outdir", str(out_dir)]]
    elif shutil.which("pdflatex"):
        commands = [["pdflatex", "-interaction=nonstopmode", "-output-directory", str(out_dir), str(tex_file)]] * 2
    else:
        return None, "No LaTeX engine found (tried tectonic, pdflatex). .tex written and validated; PDF not built.", ""
    try:
        for command in commands:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=300, cwd=str(out_dir))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, str(exc), str(exc)
    log = (proc.stdout + proc.stderr)[-4000:]
    pdf_path = str(pdf_file) if pdf_file.exists() else None
    error = (log or f"Compiler exited with status {proc.returncode}.") if proc.returncode or not pdf_path else ""
    return pdf_path, log, error


def compile_check(state: PresentationState) -> dict:
    """Compile the assembled document before review and retain its diagnostics."""
    out_dir = Path(state.get("output_dir") or "out").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "presentation.tex").write_text(state.get("latex", ""), encoding="utf-8")
    pdf_path, log, error = _compile_tex(out_dir)
    return {"compile_error": error, "compile_log": log, "pdf_path": pdf_path}


def compile_output(state: PresentationState) -> dict:
    # Absolute paths: the earlier relative -output-directory + cwd
    # combination made pdflatex look in out/<out>/ and fail to write.
    out_dir = Path(state.get("output_dir") or "out").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "presentation.tex").write_text(state.get("latex", ""), encoding="utf-8")
    (out_dir / "deck.json").write_text(json.dumps(state.get("deck", {}), indent=2), encoding="utf-8")
    (out_dir / "outline.json").write_text(json.dumps(state.get("outline", {}), indent=2), encoding="utf-8")
    (out_dir / "review.json").write_text(json.dumps(state.get("review", {}), indent=2), encoding="utf-8")

    # In the graph, compile_check already compiled this exact LaTeX (the
    # reviewer routes straight here), so reuse its result instead of
    # compiling again. Callers without a pre-check (comments revision)
    # compile now.
    if state.get("compile_log") is not None:
        return {"pdf_path": state.get("pdf_path"), "compile_log": state["compile_log"]}
    pdf_path, log, _ = _compile_tex(out_dir)
    return {"pdf_path": pdf_path, "compile_log": log}


def build_graph():
    builder = StateGraph(PresentationState)
    builder.add_node("planner", planner)
    builder.add_node("write_slide", write_slide)
    builder.add_node("editor", editor)
    builder.add_node("render_charts", render_charts)
    builder.add_node("image_agent", image_agent)
    builder.add_node("render_images", render_images)
    builder.add_node("add_references", add_references)
    builder.add_node("render_latex", render_latex)
    builder.add_node("compile_check", compile_check)
    builder.add_node("reviewer", reviewer)
    builder.add_node("compile", compile_output)

    builder.add_edge(START, "planner")
    # The path map names where the Send fan-out goes; without it the drawn
    # graph (Opik, get_graph) shows planner -> __end__ and nothing after.
    builder.add_conditional_edges("planner", route_planner, ["write_slide"])
    builder.add_edge("write_slide", "editor")
    builder.add_edge("editor", "render_charts")
    builder.add_edge("render_charts", "image_agent")
    builder.add_edge("image_agent", "render_images")
    builder.add_edge("render_images", "add_references")
    builder.add_edge("add_references", "render_latex")
    builder.add_edge("render_latex", "compile_check")
    builder.add_edge("compile_check", "reviewer")
    builder.add_conditional_edges("reviewer", route_reviewer, {"compile": "compile", "restart": "planner"})
    builder.add_edge("compile", END)
    return builder.compile()


def run(topic: str, author: str = "", institute: str = "", num_slides: int = 6,
        model: str | None = None, output_dir: str = "out", charts_enabled: bool = True,
        source_text: str = "", comments: str = "", search: bool = True,
        shallow: bool = False, images_enabled: bool = False,
        references: list[str] | None = None) -> dict:
    from .tracing import flush, prepare_graph

    graph = build_graph()
    graph, trace_config = prepare_graph(graph)
    grounding = {"mode": "source", "sources": []}
    source_references = list(references or [])
    if not source_text and search:
        # No --source: research first, then the deck. research_topic
        # plans sub-queries, searches the web, fetches pages, and
        # synthesizes a research brief; the brief becomes the source
        # the planner/writers must stick to.
        from .research import research_topic

        found = research_topic(topic, model, shallow=shallow, num_slides=num_slides)
        if found.brief:
            source_text = found.brief
            grounding = {"mode": found.mode, "sources": found.sources, "queries": found.queries}
            source_references.extend(found.sources)
        else:
            grounding = {"mode": "none", "sources": []}
    elif not source_text:
        grounding = {"mode": "none", "sources": []}
    initial: PresentationState = {
        "topic": topic,
        "author": author,
        "institute": institute,
        "num_slides": num_slides,
        "model": model or "",
        "output_dir": output_dir,
        "slides": [],
        "attempts": 0,
        "build_attempt": 0,
        "charts_enabled": charts_enabled,
        "images_enabled": images_enabled,
        "chart_count": 0,
        "source_text": source_text,
        "references": source_references,
        "comments": comments,
    }
    try:
        # Opik captures both the execution spans and compiled graph definition
        # when OPIK_API_KEY is set; prepare_graph is a no-op otherwise.
        result = graph.invoke(initial, config=trace_config)
        result["grounding"] = grounding
        return result
    finally:
        flush()
