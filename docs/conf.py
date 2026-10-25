"""Sphinx configuration for the ATESTool API reference site."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

project = "ATESTool"
copyright = "2026, Wouter Deleersnyder"
author = "Wouter Deleersnyder"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

napoleon_numpy_docstring = True
napoleon_google_docstring = False

autosummary_generate = True
templates_path = ["_templates"]
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_typehints_format = "fully-qualified"
autodoc_member_order = "bysource"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"

latex_engine = "xelatex"
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "11pt",
    # The default two-column genindex layout overlaps unreadably for this codebase's
    # long, deeply-nested qualified names (many repeated __init__/attribute names
    # across ~30 classes). Drop it from the PDF only; HTML's index page doesn't have
    # this overlap problem and keeps it.
    "printindex": "",
}
latex_documents = [
    ("index", "ATESTool.tex", "ATESTool Documentation", author, "manual"),
]
# The Python Module Index has the same two-column overlap problem; drop it from the
# PDF only (see the "printindex" note above).
latex_domain_indices = False
