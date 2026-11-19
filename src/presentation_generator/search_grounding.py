"""Ground a topic build in web search, so slides are not made up.

When the user builds a deck from a topic (no --source), the Planner
and Writers get this search text as their source — the same grounding
machinery sources.py uses, with the same hard rule: base every bullet
on a fact in it, never invent one.

Sources, in order (all free, no API key, stdlib urllib only):
1. Wikipedia — search the topic, take the top 2-3 articles, fetch
   their plain-text extracts (intro + sections, capped per article).
   This is where dates, names, and definitions come from.
2. DuckDuckGo instant answer — a short abstract when Wikipedia has
   nothing (often empty; treated as a bonus, never required).

Fails soft: if search is unreachable or finds nothing, it returns
found=False and the run proceeds from model knowledge — and says so
in the run summary, instead of pretending to be grounded.
"""
from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

UA = {"User-Agent": "MAPG/0.3 (multi-agent-presentation-generator; contact: harikrishnad1997)"}
TIMEOUT = 15
MAX_GROUNDING_CHARS = 6_000
MAX_ARTICLE_CHARS = 2_500
# A page is on-topic only if every distinctive topic word appears at least
# this often. Measured on "Google Gemini": real articles score 9+ per word,
# the zodiac "Gemini" article mentions "google" twice, google.com never
# mentions "gemini".
MIN_TERM_HITS = 3


@dataclass
class GroundingResult:
    found: bool
    text: str = ""
    sources: list[str] = field(default_factory=list)  # human-readable source labels


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", errors="ignore"))


def _clean(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    return text


def _topic_words(topic: str) -> set[str]:
    drop = {"the", "a", "an", "of", "for", "and", "with", "to", "on", "in",
            "about", "overview", "introduction", "guide", "what", "is", "are"}
    return {w for w in re.findall(r"[a-z0-9]+", topic.lower()) if len(w) >= 3 and w not in drop}


def is_relevant(words: set[str], *texts: str) -> bool:
    """True when every topic word appears MIN_TERM_HITS+ times in the texts."""
    haystack = " ".join(texts).lower()
    return all(len(re.findall(rf"\b{re.escape(w)}\b", haystack)) >= MIN_TERM_HITS for w in words)


def _article_text(title: str) -> str:
    """Best text for one article: the full extracts API text when it
    has any (deep sections), else the REST summary lead (always has
    something for a real article — the extracts API returns empty for
    some pages, e.g. Agent2Agent)."""
    text = ""
    try:
        q = urllib.parse.quote(title)
        data = _get_json(
            "https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
            f"&explaintext=1&exsectionformat=plain&format=json&titles={q}"
        )
        for page in (data.get("query", {}).get("pages", {}) or {}).values():
            text = _clean(page.get("extract", ""))
            break
    except Exception:
        text = ""
    if not text:
        try:
            q = urllib.parse.quote(title.replace(" ", "_"))
            data = _get_json(f"https://en.wikipedia.org/api/rest_v1/page/summary/{q}")
            text = _clean(data.get("extract", ""))
        except Exception:
            text = ""
    return text[:MAX_ARTICLE_CHARS]


def _wikipedia(topic: str) -> list[tuple[str, str, str]]:
    """Return [(title, url, extract)] for the top Wikipedia hits that
    actually match the topic. Search ranking alone is not enough —
    it once surfaced 'Agent-based commerce' for the A2A protocol —
    so every candidate is scored on distinct topic words in its text
    and only matches are kept."""
    q = urllib.parse.quote(topic)
    search_url = (
        "https://en.wikipedia.org/w/api.php?action=query&list=search"
        f"&srsearch={q}&srlimit=5&format=json"
    )
    try:
        data = _get_json(search_url)
    except Exception:
        return []
    hits = data.get("query", {}).get("search", [])
    titles = [h["title"] for h in hits if h.get("title")]
    # Skip disambiguation pages — they name the right article but say
    # nothing about it. Only if every hit is one do we keep the first.
    real = [x for x in titles if "(disambiguation)" not in x.lower()]
    titles = (real or titles[:1])[:3]
    words = _topic_words(topic)
    scored = []
    for i, title in enumerate(titles):
        text = _article_text(title)
        if not text:
            continue
        if not is_relevant(words, title, text):
            continue
        low = text.lower()
        score = sum(len(re.findall(rf"\b{re.escape(w)}\b", low)) for w in words)
        url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
        scored.append((-score, i, title, url, text))
    scored.sort()
    return [(title, url, text) for _, _, title, url, text in scored]


def _duckduckgo(topic: str) -> tuple[str, str] | None:
    q = urllib.parse.quote(topic)
    data = _get_json(f"https://api.duckduckgo.com/?q={q}&format=json&no_html=1&skip_disambig=1")
    abstract = _clean(data.get("AbstractText") or "")
    url = data.get("AbstractURL") or ""
    if abstract:
        return (url or "DuckDuckGo", abstract[:1500])
    return None


def ground_topic(topic: str) -> GroundingResult:
    parts: list[str] = []
    sources: list[str] = []
    try:
        articles = _wikipedia(topic)
    except Exception:
        articles = []
    for title, url, extract in articles[:3]:
        parts.append(f"--- Wikipedia: {title} ({url}) ---\n{extract}")
        sources.append(f"Wikipedia — {title}: {url}")
    try:
        ddg = _duckduckgo(topic)
    except Exception:
        ddg = None
    if ddg and (not articles or ddg[0] not in str(sources)):
        parts.append(f"--- DuckDuckGo abstract ({ddg[0]}) ---\n{ddg[1]}")
        sources.append(f"DuckDuckGo — {ddg[0]}")
    text = _clean("\n\n".join(parts))[:MAX_GROUNDING_CHARS]
    return GroundingResult(found=bool(text), text=text, sources=sources)
