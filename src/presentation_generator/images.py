"""Source a small number of images embedded in the topic's Wikipedia article.

Selecting from the article itself keeps visuals tied to the presentation topic.
Only files with identifiable licensing metadata are used; each slide and the
references section retain the article, creator, license, and file page.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import httpx

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "MultiAgentPresentationGenerator/1.0 (presentation image sourcing)"
_STOPWORDS = {
    "the", "and", "with", "from", "into", "about", "what", "when", "where", "why", "how",
    "overview", "introduction", "intro", "guide", "key", "major", "role", "impact", "future",
    "history", "summary", "explained", "understanding", "applications", "use", "uses", "case", "cases",
}


def _terms(value: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", value.lower()) if len(word) > 2 and word not in _STOPWORDS}


def _plain(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", value or "")).strip()


def find_article_images(topic: str) -> tuple[str, str, list[dict]]:
    # Wikimedia requires a descriptive User-Agent. Keep discovery and download
    # requests consistent; an undefined `headers` here silently disabled the
    # entire image stage because the graph catches discovery errors.
    with httpx.Client(
        timeout=20, follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        search = client.get(API, params={
            "action": "query", "list": "search", "srsearch": topic,
            "srnamespace": 0, "srlimit": 5, "format": "json", "formatversion": 2,
        })
        search.raise_for_status()
        topic_terms = _terms(topic)
        hits = search.json().get("query", {}).get("search", [])
        required = max(1, (len(topic_terms) + 1) // 2)
        ranked = sorted(
            [hit for hit in hits if len(topic_terms & _terms(hit.get("title", ""))) >= required],
            key=lambda hit: len(topic_terms & _terms(hit.get("title", ""))),
            reverse=True,
        )
        if not ranked:
            return "", "", []
        article_title = ranked[0]["title"]
        article_url = "https://en.wikipedia.org/wiki/" + article_title.replace(" ", "_")

        article = client.get(API, params={
            "action": "query", "prop": "images|pageimages", "titles": article_title,
            "imlimit": 30, "piprop": "name|thumbnail", "pilicense": "free",
            "pithumbsize": 1200, "format": "json", "formatversion": 2,
        })
        article.raise_for_status()
        pages = article.json().get("query", {}).get("pages", [])
        if not pages:
            return article_title, article_url, []
        page = pages[0]
        file_names = []
        if page.get("pageimage"):
            file_names.append("File:" + page["pageimage"])
        file_names.extend(item.get("title", "") for item in page.get("images", []))
        file_names = list(dict.fromkeys(name for name in file_names if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".svg"))))[:30]
        if not file_names:
            return article_title, article_url, []

        details = client.get(API, params={
            "action": "query", "prop": "imageinfo", "titles": "|".join(file_names),
            "iiprop": "url|mime|extmetadata", "iiurlwidth": 1200,
            "format": "json", "formatversion": 2,
        })
        details.raise_for_status()
        candidates = []
        for image_page in details.json().get("query", {}).get("pages", []):
            info = (image_page.get("imageinfo") or [{}])[0]
            metadata = info.get("extmetadata", {})
            thumbnail = info.get("thumburl")
            mime = info.get("mime", "")
            if not thumbnail or not mime.startswith("image/"):
                continue
            license_name = _plain(metadata.get("LicenseShortName", {}).get("value", ""))
            license_lower = license_name.lower()
            if not license_name or "fair use" in license_lower or "non-free" in license_lower:
                continue
            description = _plain(metadata.get("ImageDescription", {}).get("value", ""))
            candidates.append({
                "image_id": len(candidates) + 1,
                "url": thumbnail,
                "page": info.get("descriptionurl", ""),
                "title": image_page.get("title", ""),
                "description": description,
                "artist": _plain(metadata.get("Artist", {}).get("value", "")),
                "license": license_name,
            })
        return article_title, article_url, candidates


def render_selected_images(deck, output_dir: str, article_title: str, article_url: str,
                           candidates: list[dict], selections: list[dict]) -> tuple[object, int]:
    """Download only image-to-slide assignments approved by the image agent."""
    by_id = {int(image["image_id"]): image for image in candidates}
    by_slide = {slide.number: slide for slide in deck.slides}
    for slide in deck.slides:
        slide.image_file = None
        slide.image_source = None
    out_dir = Path(output_dir or "out") / "images"
    count = 0
    with httpx.Client(timeout=30, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
        used_ids: set[int] = set()
        used_slides: set[int] = set()
        for selection in selections[:5]:
            try:
                image_id = int(selection["image_id"])
                slide_number = int(selection["slide_number"])
                image = by_id[image_id]
                slide = by_slide[slide_number]
                if image_id in used_ids or slide_number in used_slides:
                    continue
                used_ids.add(image_id)
                used_slides.add(slide_number)
                if slide.chart_file or not slide.bullets or re.search(r"conclusion|summary|takeaway|references", slide.title, re.I):
                    continue
                response = client.get(image["url"])
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").split(";")[0].lower()
                suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}.get(content_type)
                if not suffix or len(response.content) > 12 * 1024 * 1024:
                    continue
                out_dir.mkdir(parents=True, exist_ok=True)
                filename = f"slide_{slide.number:02d}{suffix}"
                (out_dir / filename).write_bytes(response.content)
                slide.image_file = f"images/{filename}"
                slide.image_source = (
                    f"Wikipedia article: {article_title} ({article_url}); "
                    f"{image['artist']}; {image['license']}; {image['page']}"
                )
                count += 1
            except (httpx.HTTPError, ValueError, OSError, KeyError):
                continue
    return deck, count
