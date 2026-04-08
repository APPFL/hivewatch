import datetime
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../src"))

project = "HiveWatch"
copyright = f"2026-{datetime.date.today().year}, APPFL Authors"
author = "APPFL Authors"

extensions = [
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autosectionlabel_prefix_document = True
myst_enable_extensions = ["colon_fence"]

html_theme = "furo"
html_title = "HiveWatch documentation"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_theme_options = {
    "footer_icons": [],
}
