from __future__ import annotations

from datetime import datetime

# -- Project information -----------------------------------------------------

project = "MLGodotKit"
author = "Hunter Paxton"
copyright = f"{datetime.now().year}, {author}"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
language = "en"

# -- HTML output -------------------------------------------------------------

html_theme = "basic"

html_logo = "_static/MLGodotKit_logo.png"
html_favicon = "_static/Thumbnail_MLGodotKit_logo.png"

html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Will expand later
html_theme_options = {
}
