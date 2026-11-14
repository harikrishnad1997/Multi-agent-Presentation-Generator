"""LLM image-selection agent: match article images to presentation slides."""
from __future__ import annotations

import json
import re

from .graph import _chat_json
from .llm import get_llm


def select_relevant_images(deck: dict, candidates: list[dict], model: str | None = None) -> list[dict]:
    """Return up to two confident, unique article-image/slide assignments."""
    if not candidates:
        return []
    slides = [
        {"number": s.get("number"), "title": s.get("title"), "bullets": s.get("bullets", [])}
        for s in deck.get("slides", [])
        if s.get("bullets") and not s.get("chart_file")
        and not re.search(r"conclusion|summary|takeaway|references", s.get("title", ""), re.I)
    ]
    candidate_view = [
        {
            "image_id": image["image_id"],
            "file": image.get("title", ""),
            "description": (image.get("description") or "")[:500],
            "license": image.get("license", ""),
        }
        for image in candidates[:16]
    ]
    system = (
        "You are the presentation image-selection agent. Match only the supplied Wikipedia article images "
        "to slides whose subject they clearly illustrate. Do not infer image contents beyond the file title "
        "and description. Skip weak or generic matches; a slide without an image is better than a misleading one. "
        "Choose at most two images total, each image at most once, and at most one image per slide. "
        "Return JSON only with schema: {\"selections\":[{\"image_id\":int,\"slide_number\":int,\"confidence\":number,\"reason\":str}]}. "
        "Only include matches with confidence >= 0.8."
    )
    user = json.dumps({
        "presentation": deck.get("title", ""),
        "slides": slides,
        "article_images": candidate_view,
    }, ensure_ascii=False)
    try:
        data = _chat_json(get_llm(model), system, user)
    except Exception:
        return []
    slide_ids = {int(slide["number"]) for slide in slides}
    image_ids = {int(image["image_id"]) for image in candidate_view}
    chosen_images: set[int] = set()
    chosen_slides: set[int] = set()
    selections = []
    for item in data.get("selections", []):
        try:
            image_id = int(item["image_id"])
            slide_number = int(item["slide_number"])
            confidence = float(item.get("confidence", 0))
        except (KeyError, TypeError, ValueError):
            continue
        if (image_id not in image_ids or slide_number not in slide_ids or confidence < 0.8
                or image_id in chosen_images or slide_number in chosen_slides):
            continue
        chosen_images.add(image_id)
        chosen_slides.add(slide_number)
        selections.append({"image_id": image_id, "slide_number": slide_number})
        if len(selections) == 2:
            break
    return selections
