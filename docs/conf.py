import shutil
import sys
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

from sphinxcontrib import katex

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))

info = metadata("mantispy")
project = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}"
version = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
repository_url = urls["Source"]

release = info["Version"]

bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
templates_path = ["_templates"]
nitpicky = True
needs_sphinx = "4.0"

html_context = {
    "display_github": True,
    "github_user": "scverse",
    "github_repo": project,
    "github_version": "main",
    "conf_py_path": "/docs/",
}

extensions = [
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
    "scverse_misc.sphinx_ext",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
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

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "anndata": ("https://anndata.scverse.org/en/stable/", None),
    "scanpy": ("https://scanpy.scverse.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "spatialdata": ("https://spatialdata.scverse.org/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

html_theme = "scanpydoc"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_js_files = ["https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"]

html_title = project

html_theme_options = {
    "repository_url": repository_url,
    "repository_branch": "main",
    "use_repository_button": True,
    "use_issues_button": True,
    "path_to_docs": "docs/",
    "navigation_with_keys": False,
    "accent_color": "#1a878a",
    "logo": {
        "image_light": "_static/mantispy-logo.png",
        "image_dark": "_static/mantispy-logo.png",
        "alt_text": "mantispy",
    },
    "show_toc_level": 2,
}
html_show_sphinx = False
ogp_image = "_static/mantispy-logo.png"

pygments_style = "default"
katex_prerender = shutil.which(katex.NODEJS_BINARY) is not None

nitpick_ignore = [
    # scverse-misc 0.1.6 renders `optional` as a type in the Settings.reset signature it generates
    ("py:class", "optional"),
    # Python 3.14 moved pathlib internals into pathlib._local, so this xref cannot resolve
    ("py:class", "pathlib._local.Path"),
    # plottable ships no intersphinx inventory, so its Table type in evaluate_integration cannot resolve
    ("py:class", "plottable.Table"),
]
