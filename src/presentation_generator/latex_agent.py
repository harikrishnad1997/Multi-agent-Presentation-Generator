"""Per-slide LaTeX agent implementation.

The agent receives one validated slide and creates one Beamer frame fragment.
Rendering is deterministic so content is safely escaped and all environments
are balanced before the frame is assembled into the full document.
"""
from __future__ import annotations

from .latex_utils import render_slide
from .models import SlideContent


def create_slide_latex(slide: SlideContent | dict) -> str:
    """Create the LaTeX part owned by one slide's LaTeX agent."""
    validated = SlideContent.model_validate(slide)
    return render_slide(validated)
