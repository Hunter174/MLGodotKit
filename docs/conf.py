from __future__ import annotations

import os
import sys
from datetime import datetime

# -- Project information -----------------------------------------------------

project = "MLGodotKit"
author = "Hunter Paxton"
copyright = f"{datetime.now().year}, {author}"

# The full version, including alpha/beta/rc tags
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",      # Python autodoc (safe to keep)
    "sphinx.ext.napoleon",     # Google / NumPy style docstrings
    "sphinx.ext.mathjax",      # LaTeX math
    "sphinx.ext.viewcode",     # Add links to source code
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "en"

# -- HTML output -------------------------------------------------------------

html_logo = "_static/MLGodotKit_logo.png"

html_favicon = "_static/MLGodotKit_logo.png"

html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]

html_css_files = [
    "custom.css",
]

# RTD-friendly defaults
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}

