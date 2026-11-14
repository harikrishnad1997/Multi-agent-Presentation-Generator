"""Streamlit editor for the presentation generator API."""
from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import requests
import streamlit as st

st.set_page_config(page_title="Presentation Studio", page_icon="▤", layout="wide")
API_BASE = os.getenv("PRESENTATION_API_URL", "http://localhost:8000").rstrip("/")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
[data-testid="stAppViewContainer"] {background: #101319; color: #f2f4f7;}
[data-testid="stSidebar"] {background: #171b24;}
[data-testid="stHeader"] {background: rgba(16, 19, 25, .92);}
[data-testid="stChatMessage"] {background: #1b202b; border: 1px solid #303847; border-radius: 12px;}
[data-testid="stVerticalBlockBorderWrapper"] {border-color: #303847 !important;}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {color: #e4e7ec;}
[data-testid="stCaptionContainer"] {color: #aab2c0;}
[data-testid="stTextInput"] input, [data-testid="stChatInput"] textarea,
[data-testid="stNumberInput"] input {background: #202631; color: #f2f4f7; border-color: #414b5b;}
[data-testid="stFileUploader"] section {background: #1b202b; border-color: #414b5b;}
.hero {padding: 1rem 0 .4rem;}
.hero h1 {font-size: 2rem; letter-spacing: -.04em; margin-bottom: .15rem;}
.hero p {color: #aab2c0; margin: 0;}
</style>
""", unsafe_allow_html=True)

for key, default in {"job_id": None, "result": None, "messages": [], "pdf_bytes": None}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown('<div class="hero"><h1>Presentation Studio</h1><p>Generate a research deck, then refine it through conversation.</p></div>', unsafe_allow_html=True)

def api_json(method: str, path: str, **kwargs):
    response = requests.request(method, f"{API_BASE}{path}", timeout=60, **kwargs)
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise RuntimeError(f"API error ({response.status_code}): {detail}")
    return response.json()

def wait_for_job(job_id: str):
    status_box = st.empty()
    while True:
        job = api_json("GET", f"/api/generations/{job_id}")
        status = job.get("status")
        status_box.info({"queued": "Waiting for a worker…", "running": "Building your presentation…", "completed": "Presentation ready.", "failed": "The job failed."}.get(status, status))
        if status == "completed":
            status_box.empty()
            return api_json("GET", f"/api/generations/{job_id}/result")
        if status == "failed":
            raise RuntimeError(job.get("error", "Generation failed"))
        time.sleep(2)

def run_job(job):
    with st.spinner("Working on your presentation…"):
        result = wait_for_job(job["id"])
    st.session_state.job_id = job["id"]
    st.session_state.result = result
    pdf_path = result.get("pdf_path")
    pdf_artifact = next((name for name in result.get("artifacts", []) if name.lower().endswith(".pdf")), None)
    if pdf_path and pdf_artifact:
        artifact = "/".join(Path(pdf_artifact).parts)
        url = f"{API_BASE}/api/generations/{job['id']}/artifacts/{artifact}"
        pdf_response = requests.get(url, timeout=60)
        if pdf_response.ok:
            st.session_state.pdf_bytes = pdf_response.content
    else:
        st.session_state.pdf_bytes = None

with st.sidebar:
    st.subheader("Create a presentation")
    with st.form("generation_form"):
        topic = st.text_input("Topic", placeholder="e.g. The future of grid scale storage")
        openai_key = st.text_input("OpenAI API key (optional)", type="password", key="openai_key", help="Used for this presentation and its edits. If blank, the API server's OPENAI_API_KEY is used.")
        opik_key = st.text_input("Opik API key (optional)", type="password", key="opik_api_key", help="Used for this generation's traces. If blank, the API server's OPIK_API_KEY is used.")
        source_file = st.file_uploader("Or add a source", type=["pdf", "txt", "md"], key="source_file")
        slides = st.slider("Content slides", min_value=10, max_value=40, value=30)
        author = st.text_input("Author", value="Harikrishna Dev")
        institute = st.text_input("Organization", value="University of Texas at Dallas")
        charts = st.toggle("Include charts", value=True)
        search = st.toggle("Research the topic", value=True)
        images = st.toggle("Add sourced images", value=True, help="Find relevant images from Wikimedia Commons and include creator, license, and source attribution.")
        submitted = st.form_submit_button("Generate presentation", type="primary", use_container_width=True)
    if submitted:
        try:
            if source_file:
                response = requests.post(f"{API_BASE}/api/generations/upload", files={"file": (source_file.name, source_file.getvalue(), source_file.type)}, data={"topic": topic, "slides": slides, "author": author, "institute": institute, "charts": charts, "search": search, "images": images, "source_name": source_file.name, "api_key": openai_key, "opik_api_key": opik_key}, timeout=60)
                if not response.ok:
                    raise RuntimeError(response.text)
                job = response.json()
            else:
                if not topic.strip():
                    st.warning("Enter a topic or upload a source file.")
                    st.stop()
                job = api_json("POST", "/api/generations", json={"topic": topic, "slides": slides, "author": author, "institute": institute, "charts": charts, "search": search, "images": images, "api_key": openai_key, "opik_api_key": opik_key})
            st.session_state.messages = []
            run_job(job)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

if st.session_state.result:
    left, right = st.columns([0.9, 1.35], gap="large")
    with left:
        st.subheader("Edit with chat")
        st.caption(f"Current deck: {st.session_state.result.get('topic') or 'Presentation'}")
        chat_area = st.container(height=470, border=True)
        with chat_area:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        retry_last = st.button(
            "Retry last edit",
            disabled=not any(message["role"] == "user" for message in st.session_state.messages),
        )
        prompt = st.chat_input("Describe a change, e.g. 'Make slide 3 more concise'")
        if retry_last:
            prompt = next(message["content"] for message in reversed(st.session_state.messages) if message["role"] == "user")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            context = "\n".join(f"{m['role'].title()}: {m['content']}" for m in st.session_state.messages[:-1])[-12_000:]
            try:
                job = api_json("POST", f"/api/generations/{st.session_state.job_id}/edits", json={"comments": prompt, "conversation_context": context, "api_key": st.session_state.get("openai_key", "")})
                run_job(job)
                if "line graph" in prompt.lower() or "line chart" in prompt.lower():
                    slides = (st.session_state.result.get("deck") or {}).get("slides", [])
                    has_line = any((slide.get("chart") or {}).get("chart_type") == "line" for slide in slides)
                    reply = ("Updated the presentation with a line chart. The chart keeps the available values; provide year-by-year figures or a source if you want a factual before/after trend." if has_line else "I couldn't create the requested line graph from the current deck data, so I left the chart unchanged. Add a source or year-by-year values and ask again.")
                else:
                    reply = "Updated the presentation to reflect your request."
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.rerun()
            except Exception as exc:
                st.session_state.messages.append({"role": "assistant", "content": f"I couldn’t apply that edit: {exc}"})
                st.rerun()
        with st.expander("Slide outline"):
            deck = st.session_state.result.get("deck") or {}
            for slide in deck.get("slides", []):
                st.markdown(f"**{slide.get('number', '')}. {slide.get('title', '')}**")
                bullets = slide.get("bullets", [])
                if bullets:
                    st.caption(" · ".join(str(b) for b in bullets))
    with right:
        st.subheader("Presentation preview")
        if st.session_state.pdf_bytes:
            pdf = st.session_state.pdf_bytes
            encoded = base64.b64encode(pdf).decode("ascii")
            st.markdown(f'<iframe src="data:application/pdf;base64,{encoded}" width="100%" height="760" style="border:1px solid #e4e7ec;border-radius:12px;background:white"></iframe>', unsafe_allow_html=True)
            st.download_button("Download PDF", data=pdf, file_name="presentation.pdf", mime="application/pdf")
        else:
            st.info("The API did not produce a PDF. Install tectonic or pdflatex on the API host to enable PDF preview and download.")
        with st.expander("Generation details"):
            st.write(f"Job ID: `{st.session_state.job_id}`")
            st.write(f"Charts: {st.session_state.result.get('chart_count', 0)}")
            st.write(f"Sourced images: {st.session_state.result.get('image_count', 0)}")
            st.write("Artifacts:", st.session_state.result.get("artifacts", []))
else:
    st.info("Create a presentation from the left panel. The generated PDF will appear here.")
