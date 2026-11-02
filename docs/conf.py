from datetime import datetime
from importlib.metadata import metadata

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

extensions = [
    "scverse_doc",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "IPython.sphinxext.ipython_console_highlighting",
    "scverse_misc.sphinx_ext",  # documents the scverse_misc.Settings object mantispy re-exports
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
myst_url_schemes = ("http", "https", "mailto")
# Outputs are committed; CI's "Tutorials run" job re-executes the tutorials (-D nb_execution_mode=cache), except the ~3 GB dataset pages.
nb_execution_excludepatterns = ["datasets/*", "case_studies/*"]
nb_execution_timeout = 900
nb_execution_raise_on_error = True
typehints_defaults = "braces"
always_use_bars_union = True

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_js_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js",
    "js/plotly_resize.js",  # re-fit plotly figures to their container (pydata grid + cross-browser)
]

html_logo = "_static/mantispy-logo.png"
html_theme_options = {"secondary_accent": True}
html_show_sphinx = False
ogp_image = "_static/mantispy-logo.png"

nitpick_ignore = [
    # scverse-misc 0.1.6 renders `optional` as a type in the Settings.reset signature it generates
    ("py:class", "optional"),
    # Python 3.14 moved pathlib internals into pathlib._local, so this xref cannot resolve
    ("py:class", "pathlib._local.Path"),
]
