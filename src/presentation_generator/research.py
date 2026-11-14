"""Deep research first, then a deck.

A topic build used to be grounded only in a Wikipedia search
(~6k chars, one angle). This phase goes deeper before the Planner
 ever runs:

1. Plan — the LLM turns the topic into 5 research questions
   (origins/history, key facts & numbers, how it works, adoption
   and use cases, challenges & latest developments).
2. Search — each question is run against Bing web search
   (no API key; DuckDuckGo's HTML endpoint is unreachable from
   here) and the top fresh pages are fetched and read as text.
3. Base — the Wikipedia grounding from search_grounding.py is
   added as the encyclopedic base, so the brief always has the
   definitions and dates even when the open web is thin.
4. Synthesize — the LLM writes a structured research brief from
   the gathered excerpts only: overview, key facts with dates and
   numbers, timeline, how it works, use cases & adoption,
   challenges. The brief becomes the source the Planner and
   Writers must stick to, under the same "never invent a fact"
   rule as --source.

Fails soft at every step: a dead search backend, an unfetchable
page, or a failed synthesis never breaks the build — the phase
falls back to the Wikipedia-only grounding, and if even that
finds nothing it reports mode "none" honestly.
"""
from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from .llm import get_llm

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
TIMEOUT = 15
MAX_PAGES = 8
MAX_PAGE_CHARS = 3_000
MAX_BRIEF_CHARS = 10_000
MAX_RAW_CHARS = 14_000
SKIP_DOMAINS = ("youtube.com", "youtu.be", "facebook.com", "instagram.com", "x.com", "twitter.com")


@dataclass
class ResearchResult:
    brief: str = ""
    mode: str = "none"  # "research" | "search" (Wikipedia fallback) | "none"
    sources: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)


def _strip_tags(text: str) -> str:
    text = re.sub(r"(?is)<(script|style|nav|header|footer|noscript)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _unwrap_bing_url(href: str) -> str:
    """Bing result links are /ck/a redirects; the real destination is
    base64 in the u= parameter. Direct links pass through."""
    href = html.unescape(href)
    if "/ck/a" in href:
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
        enc = (qs.get("u") or [""])[0]
        if not enc:
            return ""
        # Bing prefixes the payload with "a1" (a version marker);
        # without stripping it the length is 1 mod 4 and undecodable.
        if enc.startswith("a1"):
            enc = enc[2:]
        try:
            import base64

            decoded = base64.urlsafe_b64decode(enc + "=" * (-len(enc) % 4)).decode("utf-8", errors="ignore")
        except Exception:
            return ""
        return decoded if decoded.startswith("http") else ""
    return href


def _bing_search(query: str, max_results: int = 5) -> list[tuple[str, str, str]]:
    """Bing web search, no key. Returns [(url, title, snippet)]."""
    url = "https://www.bing.com/search?q=" + urllib.parse.quote(query)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        page = resp.read().decode("utf-8", errors="ignore")
    out: list[tuple[str, str, str]] = []
    for block in re.findall(r'(?s)<li class="b_algo".*?</li>', page):
        # The anchor carries attributes before href (target="_blank"),
        # and href is usually a Bing /ck/a redirect, not the target.
        m = re.search(r'<h2><a[^>]*href="([^"]+)"[^>]*>(.*?)</a></h2>', block)
        if not m:
            continue
        link = _unwrap_bing_url(m.group(1))
        if not link.startswith("http") or any(d in link for d in ("bing.com", "microsoft.com")) or any(d in link for d in SKIP_DOMAINS):
            continue
        title = _strip_tags(m.group(2))
        snip = re.search(r"(?s)<p[^>]*>(.*?)</p>", block)
        snippet = _strip_tags(snip.group(1)) if snip else ""
        out.append((link, title, snippet))
        if len(out) >= max_results:
            break
    return out


def _fetch_page(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        raw = resp.read(200_000).decode("utf-8", errors="ignore")
    text = _strip_tags(raw)
    # Pages carry menus/cookie banners; the first chunk is usually boilerplate.
    return text[:MAX_PAGE_CHARS]


def _chat_json(llm, system: str, user: str) -> dict:
    resp = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    match = re.search(r"\{.*\}", content, re.S)
    return json.loads(match.group(0)) if match else {}


def _plan_queries(topic: str, model: str | None) -> list[str]:
    try:
        llm = get_llm(model)
        data = _chat_json(
            llm,
            "You plan web research. Output JSON only: {\"queries\": [str, 5 items]}.",
            f"Topic: {topic}\nGive 5 short web-search queries that together cover: "
            "origins and history; key facts, dates and numbers; how it works; "
            "adoption, use cases and key players; challenges, limitations and latest developments.",
        )
        queries = [q.strip() for q in data.get("queries", []) if isinstance(q, str) and q.strip()]
        if len(queries) >= 3:
            return queries[:6]
    except Exception:
        pass
    return [
        f"{topic} history origins",
        f"{topic} key facts dates numbers",
        f"how {topic} works",
        f"{topic} adoption use cases",
        f"{topic} challenges latest developments",
    ]


def _synthesize(topic: str, excerpts: str, model: str | None) -> str:
    llm = get_llm(model)
    system = (
        "You are a research analyst. Write a structured research brief from "
        "the excerpts ONLY — every fact must come from them, with the source "
        "in brackets after it where it matters. Never invent a date, number, "
        "name, or claim. If the excerpts disagree or are thin on a point, "
        "say so. Plain text, no markdown headers, sections as: Overview / "
        "Key facts / Timeline / How it works / Use cases and adoption / "
        "Challenges and limitations."
    )
    resp = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": f"Topic: {topic}\n\nExcerpts (each headed by its source):\n{excerpts}"},
    ])
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    return content.strip()[:MAX_BRIEF_CHARS]


def research_topic(topic: str, model: str | None = None, shallow: bool = False) -> ResearchResult:
    from .search_grounding import ground_topic

    if shallow:
        found = ground_topic(topic)
        return ResearchResult(
            brief=found.text, mode="search" if found.found else "none", sources=found.sources,
        )
    queries = _plan_queries(topic, model)
    seen: set[str] = set()
    excerpts: list[str] = []
    sources: list[str] = []
    for query in queries:
        try:
            results = _bing_search(query)
        except Exception:
            continue
        for url, title, snippet in results:
            if url in seen or len(excerpts) >= MAX_PAGES:
                break
            seen.add(url)
            try:
                text = _fetch_page(url)
            except Exception:
                continue
            if len(text) < 200:
                continue
            domain = urllib.parse.urlparse(url).netloc
            excerpts.append(f"--- {title} ({domain}) ---\n{url}\n{text}")
            sources.append(f"{title[:100]} — {url}")
    # Wikipedia base: definitions and dates even when the open web is thin.
    try:
        found = ground_topic(topic)
        if found.found:
            excerpts.append(f"--- Wikipedia grounding ---\n{found.text}")
            sources.extend(found.sources)
    except Exception:
        found = None
    if not excerpts:
        if found and found.found:
            return ResearchResult(brief=found.text, mode="search", sources=found.sources, queries=queries)
        return ResearchResult(brief="", mode="none", sources=[], queries=queries)
    raw = "\n\n".join(excerpts)[:MAX_RAW_CHARS]
    try:
        brief = _synthesize(topic, raw, model)
    except Exception:
        brief = raw[:MAX_BRIEF_CHARS]
    if not brief.strip():
        brief = raw[:MAX_BRIEF_CHARS]
    return ResearchResult(brief=brief, mode="research", sources=sources, queries=queries)
