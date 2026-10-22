import shutil
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

from sphinxcontrib import katex

HERE = Path(__file__).parent

info = metadata("mantispy")
project = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}"
version = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
source_repository = urls["Source"]

release = info["Version"]

bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
templates_path = ["_templates"]
nitpicky = True
needs_sphinx = "4.0"

extensions = [
    "scverse_doc",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.katex",
    "sphinx_autodoc_typehints",
    "sphinx_design",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxext.opengraph",
    "scverse_misc.sphinx_ext",  # documents the scverse_misc.Settings object mantispy re-exports
]

autosummary_generate = True
autodoc_member_order = "groupwise"
default_role = "literal"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
myst_heading_anchors = 6
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto")
nb_output_stderr = "remove"
# Outputs are committed; CI's "Tutorials run" job re-executes the tutorials (-D nb_execution_mode=cache), except the ~3 GB dataset pages.
nb_execution_mode = "off"
nb_execution_excludepatterns = ["datasets/*", "case_studies/*"]
nb_execution_timeout = 900
nb_execution_raise_on_error = True
nb_merge_streams = True
typehints_defaults = "braces"
always_use_bars_union = True

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

from scverse_doc.registry import intersphinx as _scverse_intersphinx  # noqa: E402

# scverse packages come from the shared registry; the rest are mantispy-specific. Entries here win.
intersphinx_mapping = _scverse_intersphinx() | {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# The scverse theme (pydata-sphinx-theme) is registered by the scverse_doc extension.
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_js_files = ["https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"]

html_title = project
html_logo = "_static/mantispy-logo.png"
# accent sets the theme fallback; custom.css owns the full teal/orange palette.
html_theme_options = {"accent": "#1a878a"}
html_show_sphinx = False
ogp_image = "_static/mantispy-logo.png"

pygments_style = "default"
katex_prerender = shutil.which(katex.NODEJS_BINARY) is not None

nitpick_ignore = [
    # scverse-misc 0.1.6 renders `optional` as a type in the Settings.reset signature it generates
    ("py:class", "optional"),
    # Python 3.14 moved pathlib internals into pathlib._local, so this xref cannot resolve
    ("py:class", "pathlib._local.Path"),
]
