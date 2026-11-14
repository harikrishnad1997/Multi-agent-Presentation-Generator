# Multi-agent Presentation Generator

Generate a Beamer (LaTeX) presentation from a topic, with LangChain
agents orchestrated by LangGraph — and GPT as the LLM throughout.

This replaces the original June 2024 stub, which was a single
hardcoded Gemini deck in `example.py` with no LLM call at all. That
file also had a real bug: it did `text.replace('\\n', '\n')` on LLM
output, which rewrote the `\n` inside `\note{...}` into a newline +
`ote{...}`. This version never does string-level newline replacement
on LaTeX — see `latex_utils.py`.

## Example output

- [A2A Agent Protocol, 30 slides, 9 charts (PDF)](examples/a2a-agent-protocol-30-slides.pdf) —
  built by the A2A run (`run_a2a.py --topic "A2A Agent Protocol (Google Agent2Agent)" --slides 30`,
  search-grounded on Wikipedia): 30 slides, 9 charts (the full 9/9 visuals),
  review passed round 3.

## Flow

```
topic
  -> Planner: structured outline, one brief per slide
  -> Slide Writers (parallel, one agent per slide, via LangGraph Send)
  -> Flow Editor: reads the whole deck in order — narrative arc,
     transitions, duplicated points, ordering, tone
  -> LaTeX: deterministic Beamer rendering from the structured deck
  -> Reviewer: LLM review + structural LaTeX checks, max 3 rounds
  -> Compiler: writes .tex / .json, builds PDF if tectonic/pdflatex exists
```

## A2A version

The same pipeline over Google's Agent2Agent protocol
(`run_a2a.py`, `src/presentation_generator/a2a_agents.py`,
`src/presentation_generator/a2a_client.py`,
`src/presentation_generator/a2a_server.py`,
`src/presentation_generator/graph_a2a.py`):

- Each agent — Planner, Writer, Editor, Chart, Reviewer — is a
  real A2A agent (`a2a-sdk==0.3.26`, the classic Starlette
  implementation; 1.x moved AgentCard to protobuf types and dropped
  the app factory, the wire protocol is the same). It publishes an
  Agent Card at `/.well-known/agent-card.json` and serves
  JSON-RPC `message/send` on its own port: planner 8101, writer
  8102, editor 8103, chart 8104, reviewer 8105.
- LangGraph stays as the orchestrator, but its nodes are A2A
  clients: they resolve the Card, send a Message with the JSON
  payload, and read the Task artifact back. Agents share the core
  functions in `graph.py`, so there is one implementation and two
  transports (in-process, and A2A over HTTP).
- `python run_a2a.py --topic "Google Gemini" --slides 30 --out out/gemini30-a2a`
  starts all five servers, waits for their Cards, runs the
  pipeline, and shuts them down.

## 30-slide fixes (v0.2)

The first 30-slide run failed review on a duplicated Conclusion
(slides 24+25) and produced only 4 charts instead of 9. Fixed:
planner outlines are now validated (exact slide count, exact
visual count, exactly one ending slide, last) with retry +
deterministic repair; the editor may drop a duplicated slide and
fills any shortfall with fresh content slides; revise rounds start
from the edited deck, never from raw writer slides; the compiler
now uses absolute output paths (the relative `-output-directory`
+ cwd combination made pdflatex fail to write).

Why parallel writers + a separate editor: writers run in parallel
(writing N slides takes roughly the time of the slowest single
slide, not N times as long) and each one stays strictly on its
brief, but no single writer can see the whole story. The editor is
the agent that can — it reorders, cuts duplicates, and smooths
transitions. Results are collected by slide number, not finish
order, so the deck never scrambles; if one slide's LLM call fails,
that slide falls back to its brief (key points as bullets) instead
of killing the whole deck.

Why LaTeX is rendered, not generated: an LLM writing raw LaTeX is
where decks break (unescaped `%`, missing `\end{frame}`). The deck
is structured JSON until the last step, then a fixed template renders
it with proper escaping.

## Usage

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

python main.py --topic "Google Gemini" --author "Harikrishna Dev" \
  --institute "University of Texas at Dallas" --slides 6 --out out/gemini
```

Output in the `--out` folder: `presentation.tex`, `deck.json`,
`outline.json`, `review.json`, and `presentation.pdf` if a LaTeX
engine (`tectonic` or `pdflatex`) is installed.

## Editing a deck with your comments

```bash
# Edit an existing deck — no slide is regenerated from scratch.
# The Flow Editor applies each comment, then the deck is re-rendered
# and re-compiled.
python main.py --edit out/gemini/deck.json --outline out/gemini/outline.json \
  --comments $'Slide 5: shorten the bullets to one line each\nDrop slide 12, it repeats slide 9' \
  --out out/gemini-v2

# Or apply comments on a fresh build (same flags work on run_a2a.py too).
python main.py --topic "Google Gemini" --slides 6 \
  --comments "overall: keep every bullet under 12 words" --out out/gemini
```

Comments are one per line, optionally aimed at a slide
(`Slide 5: ...`, `overall: ...`); a JSON file works too
(`--comments-file comments.json`, `{"comments": [{"slide": 5, "comment": "..."}]}`).
A comment can rewrite, shorten, or drop a slide, or ask to
`add a slide about X` — the new slide is written before the ending slide.
Slide numbers are resolved to titles against the deck as you saw
it, so adding or dropping a slide never shifts a later comment's
target; a near-duplicate title (e.g. "Gemini Pricing Tiers" when
"Pricing Tiers" already exists) is folded into the existing slide
instead of added as a second one.

## Building a deck from a source

```bash
# From a PDF — every slide is grounded in the document, not invented.
python main.py --source ~/Downloads/paper.pdf --slides 10 --out out/paper

# From a public GitHub repo (shallow-cloned: README + docs + file tree)
# or a local folder/repo — topic defaults to the repo name.
python main.py --source https://github.com/harikrishnad1997/Multi-agent-Presentation-Generator --slides 10 --out out/repo-deck
```

Sources: `.pdf`, `.txt`/`.md`, a local folder, or a public GitHub
repo URL. The source text (capped at ~12k chars, README/docs first)
goes to the Planner and to each Writer with one hard rule: base every
bullet on a fact in the source, never invent one. For a private repo,
clone it locally and pass the folder. Same flags work on `run_a2a.py`.

## Search grounding (slides are not made up)

Every topic build is grounded in web search by default — no flag
needed. Before the Planner runs, the topic is searched on Wikipedia
(top articles, disambiguation pages skipped, each candidate scored
on the topic's own words) plus a DuckDuckGo abstract, and that text
becomes the source the Planner and Writers must stick to, under the
same "never invent a fact" rule as `--source`. The run prints what
it was grounded in, e.g. `Grounded in web search: Wikipedia —
Agent2Agent`. If search finds nothing reliable, the deck still
builds from model knowledge but says so plainly instead of
pretending to be grounded. `--no-search` turns it off; `--source`
already grounds the deck, so search is skipped then.

## Tracing (Opik)

One Opik trace per deck build — project `multi-agent-presentation-generator`,
workspace `harikrishnad1997` (see `src/presentation_generator/tracing.py`,
same pattern as Reel Chat). The LangChain `OpikTracer` callback traces
every LLM call and LangGraph node in both the in-process and A2A runs,
and each A2A agent also records its own span server-side.

Tracing is on when `OPIK_API_KEY` is set (`PRESENTATION_OPIK=off` to
disable it); without a key it is a strict no-op and a deck build is
never affected. Tracing failures can never break a build — everything
fails soft. The project is created automatically when the first trace
lands. `opik` is pinned to `==2.2.138` (a newer unpinned Opik rejected
the `llm` span type at import and crashed Reel Chat in Nov 2026).

## LLM — GPT

Default model is `gpt-4o-mini` (`--model` / `OPENAI_MODEL` to change).
The key is resolved in `src/presentation_generator/llm.py`:

- `OPENAI_API_KEY` set → LangChain `ChatOpenAI` (the normal path for
  anyone cloning this repo).
- Otherwise → a small LangChain `BaseChatModel` that shells out to
  Muse's secure OpenAI connector helper, which injects the key at
  request time. The key never appears in an env var, a file, the code,
  or any output.

Structured output is JSON-only prompting + parsing, not
tool-calling, so it works identically on both backends.

## Layout

- `main.py` — CLI
- `run_a2a.py` — CLI for the A2A transport (starts the five agent servers)
- `src/presentation_generator/graph.py` — LangGraph pipeline + agents
- `src/presentation_generator/graph_a2a.py` — same pipeline over A2A
- `src/presentation_generator/a2a_agents.py` — the five A2A agents
- `src/presentation_generator/a2a_client.py` — A2A client (Card → message/send → artifact)
- `src/presentation_generator/a2a_server.py` — A2A server entry point
- `src/presentation_generator/llm.py` — GPT backends (ChatOpenAI / helper)
- `src/presentation_generator/models.py` — Pydantic schemas agents speak
- `src/presentation_generator/latex_utils.py` — Beamer rendering + validation
- `src/presentation_generator/charts.py` — deterministic matplotlib charts
- `src/presentation_generator/comments.py` — user comments that edit a deck
- `src/presentation_generator/sources.py` — build a deck from a PDF / file / repo
- `src/presentation_generator/search_grounding.py` — Wikipedia + DuckDuckGo grounding
- `src/presentation_generator/tracing.py` — Opik tracing (no-op without a key)
