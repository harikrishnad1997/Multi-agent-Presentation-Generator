"""Build a deck from a source document instead of from the model's
own knowledge: a PDF, a text/markdown file, a local folder (e.g. a
checked-out repo), or a public GitHub repo URL.

The source text is passed to the Planner (and to each Writer) as
grounding, with one hard rule: base every slide on facts in the
source and never invent facts that are not in it. Source text is
capped (~12k chars) so it fits comfortably in every prompt; for a
repo the README and docs come first, then the file tree.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

MAX_SOURCE_CHARS = 12_000


@dataclass
class SourceDoc:
    title: str
    text: str
    kind: str  # "pdf" | "text" | "repo" | "folder"
    ref: str   # path or URL the text came from


def _cap(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    return text[:MAX_SOURCE_CHARS]


def _from_pdf(path: Path) -> SourceDoc:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    title = path.stem.replace("-", " ").replace("_", " ").title()
    return SourceDoc(title=title, text=_cap("\n".join(pages)), kind="pdf", ref=str(path))


def _from_text_file(path: Path) -> SourceDoc:
    title = path.stem.replace("-", " ").replace("_", " ").title()
    return SourceDoc(title=title, text=_cap(path.read_text(encoding="utf-8", errors="ignore")),
                     kind="text", ref=str(path))


def _from_folder(path: Path) -> SourceDoc:
    """A local folder (usually a checked-out repo): README + docs first,
    then the file tree, so the planner sees what the project is and
    what it contains."""
    parts: list[str] = []
    for name in ("README.md", "README", "README.txt"):
        f = path / name
        if f.exists():
            parts.append(f.read_text(encoding="utf-8", errors="ignore"))
            break
    docs = sorted(p for p in path.rglob("*.md") if ".git" not in p.parts and p.name.upper() != "README.MD")
    for doc in docs[:5]:
        try:
            parts.append(f"\n--- {doc.relative_to(path)} ---\n" + doc.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            pass
    tree = []
    for p in sorted(path.rglob("*")):
        if ".git" in p.parts or "__pycache__" in p.parts or "node_modules" in p.parts:
            continue
        if p.is_file():
            tree.append(str(p.relative_to(path)))
    parts.append("\n--- file tree ---\n" + "\n".join(tree[:200]))
    return SourceDoc(title=path.name.replace("-", " ").replace("_", " ").title(),
                     text=_cap("\n".join(parts)), kind="folder", ref=str(path))


def _from_github_url(url: str) -> SourceDoc:
    """Public GitHub repo URL: shallow clone to a temp dir, then read it
    as a folder. Private repos fail here with a clear error — clone a
    local copy (or pass the folder) instead."""
    match = re.match(r"https?://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?$", url.strip())
    if not match:
        raise ValueError(f"Not a GitHub repo URL: {url}")
    owner, repo = match.groups()
    with tempfile.TemporaryDirectory(prefix="mapg-source-") as tmp:
        dest = str(Path(tmp) / repo)
        proc = subprocess.run(
            ["git", "clone", "--depth", "1", f"https://github.com/{owner}/{repo}.git", dest],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Could not clone https://github.com/{owner}/{repo} "
                "(is it public? for a private repo, clone it locally and pass the folder). "
                f"git said: {(proc.stderr or '').strip()[-200:]}"
            )
        doc = _from_folder(Path(dest))
        doc.kind = "repo"
        doc.ref = url
        return doc


def load_source(source: str) -> SourceDoc:
    """Load a source from a path or a GitHub URL. Auto-detects the kind."""
    source = source.strip()
    if source.startswith("http://") or source.startswith("https://"):
        if "github.com" in source:
            return _from_github_url(source)
        raise ValueError(f"Only GitHub repo URLs are supported as remote sources: {source}")
    path = Path(os.path.expanduser(source))
    if not path.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    if path.is_dir():
        return _from_folder(path)
    if path.suffix.lower() == ".pdf":
        return _from_pdf(path)
    return _from_text_file(path)
