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

- [Google Gemini, 30 slides, 8 charts (PDF)](examples/google-gemini-30-slides.pdf) —
  built by the A2A run (`run_a2a.py --topic "Google Gemini" --slides 30`):
  30 slides, 8 charts, review passed round 1.

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
(`run_a2a.py`, `src/presentation_generator/a2a_*.py`,
`graph_a2a.py`):

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

Why parallel writers + a separate editor: writers are fast in
parallel (a 7-slide deck takes roughly the time of 1 slide) and each
one stays strictly on its brief, but no single writer can see the
whole story. The editor is the agent that can — it reorders, cuts
duplicates, and smooths transitions. Results are collected by slide
number, not finish order, so the deck never scrambles; if one slide
fails, only that slide retries.

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
- `src/presentation_generator/graph.py` — LangGraph pipeline + agents
- `src/presentation_generator/llm.py` — GPT backends (ChatOpenAI / helper)
- `src/presentation_generator/models.py` — Pydantic schemas agents speak
- `src/presentation_generator/latex_utils.py` — Beamer rendering + validation
