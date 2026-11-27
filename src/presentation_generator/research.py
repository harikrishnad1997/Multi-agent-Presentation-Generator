"""Deep research first, then a deck.

A topic build used to be grounded only in a Wikipedia search
(~6k chars, one angle). This phase goes deeper before the Planner
 ever runs:

1. Plan — the LLM turns the topic into 5 research questions
   (origins/history, key facts & numbers, how it works, adoption
   and use cases, challenges & latest developments).
2. Search — each question is run against Tavily when
   TAVILY_API_KEY is set (it returns page text directly), else
   against keyless Bing web search, whose pages are fetched and
   read as text. Keyless Bing matches loosely (it effectively keys
   on the first word, so "Google Gemini ..." returns google.com), so
   every page must pass a topic-relevance check before it is used
   or cited.
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
import os
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
# Larger decks need more distinct material: 8 sources gave a 30-slide
# deck too little, and the planner padded it with repeated 'Overview' /
# 'Future' / 'Community' slides (v8, v9). Budgets grow with the slide
# count up to these caps; decks of up to 15 slides keep the defaults.
MAX_QUERIES_CAP = 12
MAX_PAGES_CAP = 20
BRIEF_CHARS_PER_SLIDE = 600


@dataclass(frozen=True)
class ResearchBudget:
    queries: int = 5
    pages: int = MAX_PAGES
    raw_chars: int = MAX_RAW_CHARS
    brief_chars: int = MAX_BRIEF_CHARS


def research_budget(num_slides: int | None) -> ResearchBudget:
    n = num_slides or 0
    pages = min(MAX_PAGES_CAP, max(MAX_PAGES, -(-n // 2)))
    return ResearchBudget(
        queries=min(MAX_QUERIES_CAP, max(5, -(-n // 3))),
        pages=pages,
        raw_chars=MAX_RAW_CHARS * pages // MAX_PAGES,
        brief_chars=max(MAX_BRIEF_CHARS, n * BRIEF_CHARS_PER_SLIDE),
    )


TAVILY_URL = "https://api.tavily.com/search"
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


def _tavily_search(query: str, api_key: str, max_results: int = 5) -> list[tuple[str, str, str]]:
    """Tavily search. Returns [(url, title, page_text)]; the text is the
    page's extracted content, so no separate fetch is needed."""
    body = json.dumps({
        "query": query, "max_results": max_results, "search_depth": "basic",
        "include_raw_content": True,
    }).encode("utf-8")
    req = urllib.request.Request(TAVILY_URL, data=body, method="POST", headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {api_key}",
    })
    with urllib.request.urlopen(req, timeout=TIMEOUT * 2) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="ignore"))
    out: list[tuple[str, str, str]] = []
    for item in data.get("results", []):
        url = item.get("url") or ""
        if not url.startswith("http") or any(d in url for d in SKIP_DOMAINS):
            continue
        text = re.sub(r"\s+", " ", item.get("raw_content") or item.get("content") or "").strip()
        out.append((url, item.get("title") or url, text[:MAX_PAGE_CHARS]))
    return out


def _search(query: str) -> list[tuple[str, str, str, str | None]]:
    """[(url, title, snippet, page_text or None)] — Tavily when keyed, else
    Bing (whose pages still need fetching, signalled by None)."""
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if api_key:
        try:
            return [(url, title, "", text) for url, title, text in _tavily_search(query, api_key)]
        except Exception:
            pass  # fall back to keyless Bing
    return [(url, title, snippet, None) for url, title, snippet in _bing_search(query)]


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


def _plan_queries(topic: str, model: str | None, count: int = 5) -> list[str]:
    try:
        llm = get_llm(model)
        aspects = ("origins and history; key facts, dates and numbers; how it works; "
                   "adoption, use cases and key players; challenges, limitations and latest developments")
        if count > 5:
            # More queries must mean more distinct ground, not rephrasings.
            aspects += ("; technical specification details; architecture and components; tooling, SDKs "
                        "and implementations; security; comparisons with alternatives; named case "
                        "studies; governance and community; roadmap")
        data = _chat_json(
            llm,
            f"You plan web research. Output JSON only: {{\"queries\": [str, {count} items]}}.",
            f"Topic: {topic}\nGive {count} short web-search queries, each on a different aspect, "
            f"that together cover: {aspects}.",
        )
        queries = [q.strip() for q in data.get("queries", []) if isinstance(q, str) and q.strip()]
        if len(queries) >= 3:
            return queries[:count]
    except Exception:
        pass
    return [
        f"{topic} history origins",
        f"{topic} key facts dates numbers",
        f"how {topic} works",
        f"{topic} adoption use cases",
        f"{topic} challenges latest developments",
    ]


def _synthesize(topic: str, excerpts: str, model: str | None,
                num_slides: int | None = None, max_chars: int = MAX_BRIEF_CHARS) -> str:
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
    if num_slides and num_slides > 16:
        # A long deck needs one distinct sub-topic per slide in the brief.
        system += (
            f" The brief feeds a {num_slides}-slide deck: after those sections, add a section "
            f"'Distinct sub-topics' listing up to {num_slides} specific, non-overlapping sub-topics the "
            "excerpts support, each with its supporting facts. Never pad: list only what the "
            "excerpts cover."
        )
    resp = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": f"Topic: {topic}\n\nExcerpts (each headed by its source):\n{excerpts}"},
    ])
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    return content.strip()[:max_chars]


def research_topic(topic: str, model: str | None = None, shallow: bool = False,
                   num_slides: int | None = None) -> ResearchResult:
    """num_slides sizes the research: more queries, sources and brief for
    longer decks (see research_budget)."""
    from .search_grounding import _topic_words, ground_topic, is_relevant

    if shallow:
        found = ground_topic(topic)
        return ResearchResult(
            brief=found.text, mode="search" if found.found else "none", sources=found.sources,
        )
    budget = research_budget(num_slides)
    queries = _plan_queries(topic, model, budget.queries)
    terms = _topic_words(topic)
    seen: set[str] = set()
    excerpts: list[str] = []
    sources: list[str] = []
    for query in queries:
        if len(excerpts) >= budget.pages:
            break
        try:
            results = _search(query)
        except Exception:
            continue
        for url, title, snippet, text in results:
            if len(excerpts) >= budget.pages:
                break
            if url in seen:
                continue  # already read via an earlier query; the rest of this query's results still count
            seen.add(url)
            if text is None:
                try:
                    text = _fetch_page(url)
                except Exception:
                    continue
            if len(text) < 200 or not is_relevant(terms, title, snippet, text):
                continue
            domain = urllib.parse.urlparse(url).netloc
            excerpts.append(f"--- {title} ({domain}) ---\n{url}\n{text}")
            sources.append(f"{title[:100]} — {url}")
    # Wikipedia base: definitions and dates even when the open web is thin.
    try:
        found = ground_topic(topic)
        if found.found:
            excerpts.append(f"--- Wikipedia grounding ---\n{found.text}")
            # Labels end in the URL; skip articles the web search already cited.
            cited = {urllib.parse.unquote(u) for u in seen}
            sources.extend(src for src in found.sources
                           if urllib.parse.unquote(src.rsplit(" ", 1)[-1]) not in cited)
    except Exception:
        found = None
    if not excerpts:
        if found and found.found:
            return ResearchResult(brief=found.text, mode="search", sources=found.sources, queries=queries)
        return ResearchResult(brief="", mode="none", sources=[], queries=queries)
    raw = "\n\n".join(excerpts)[:budget.raw_chars]
    try:
        brief = _synthesize(topic, raw, model, num_slides=num_slides, max_chars=budget.brief_chars)
    except Exception:
        brief = raw[:budget.brief_chars]
    if not brief.strip():
        brief = raw[:budget.brief_chars]
    return ResearchResult(brief=brief, mode="research", sources=sources, queries=queries)
